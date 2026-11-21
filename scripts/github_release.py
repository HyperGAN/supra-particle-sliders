"""Validate a reviewed Supra model manifest and publish its GitHub release."""

import argparse
import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path
from urllib.request import urlopen

REPOSITORY = "HyperGAN/supra-particle-sliders"
HUB = "https://huggingface.co/ntc-ai/supra-particle-sliders"
MANIFEST = "release-manifest.json"
ASSET_PATHS = (
    "catalog.json",
    "source-provenance.json",
    "distilled/final-boss-1600-rank8.safetensors",
    "distilled/final-boss-converged-rank8.safetensors",
    "weights/final-boss-1600.safetensors",
    "weights/final-boss-converged.safetensors",
)


def sha256(path):
    with path.open("rb") as contents:
        return hashlib.file_digest(contents, "sha256").hexdigest()


def load_request(path):
    request = json.loads(path.read_bytes())
    formats = {
        "tag": r"v(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)",
        "source_commit": r"[0-9a-f]{40}",
        "hub_revision": r"[0-9a-f]{40}",
        "manifest_sha256": r"[0-9a-f]{64}",
    }
    if not isinstance(request, dict) or request.keys() != formats.keys():
        raise ValueError("release.json must contain tag, source_commit, hub_revision and manifest_sha256")
    for key, pattern in formats.items():
        if not isinstance(request[key], str) or not re.fullmatch(pattern, request[key]):
            raise ValueError(f"Invalid release field {key}")
    return request


def command(*args):
    return subprocess.check_output(args, text=True).strip()


def verify_source(commit):
    if command("git", "cat-file", "-t", commit) != "commit":
        raise ValueError("Release source is not a Git commit")
    subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"], check=True)


def fetch_asset(base, name, path, max_bytes):
    with urlopen(f"{base}/{name}", timeout=60) as response, path.open("wb") as out:
        remaining = max_bytes
        while chunk := response.read(min(remaining + 1, 1024 * 1024)):
            remaining -= len(chunk)
            if remaining < 0:
                raise ValueError(f"Download exceeded expected size for {name}")
            out.write(chunk)


def asset_record(path):
    return {"bytes": path.stat().st_size, "sha256": sha256(path)}


def load_asset_manifest(request, directory):
    manifest_path = directory / MANIFEST
    manifest = asset_record(manifest_path)
    if manifest["sha256"] != request["manifest_sha256"]:
        raise ValueError("Hub release manifest does not match the reviewed SHA-256")
    payload = json.loads(manifest_path.read_bytes())
    if payload["source_commit"] != request["source_commit"]:
        raise ValueError("Hub release manifest identifies a different source commit")
    records = {}
    for name in ASSET_PATHS:
        entry = payload["files"][name]
        if (type(entry["bytes"]) is not int or entry["bytes"] <= 0
                or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])):
            raise ValueError(f"Invalid size or SHA-256 for {name}")
        records[Path(name).name] = entry
    records[MANIFEST] = manifest
    return records


def verify_payload(request, directory):
    records = load_asset_manifest(request, directory)
    for name, expected in records.items():
        if asset_record(directory / name) != expected:
            raise ValueError(f"Content does not match the release manifest: {name}")
    provenance = json.loads((directory / "source-provenance.json").read_bytes())
    if (provenance["repository"] != f"https://github.com/{REPOSITORY}"
            or provenance["commit"] != request["source_commit"]):
        raise ValueError("Source provenance does not match the reviewed repository and commit")
    catalog = json.loads((directory / "catalog.json").read_bytes())
    weights = {
        slider[kind]: slider[f"{kind}_sha256"]
        for slider in catalog["sliders"]
        for kind in ("original", "distill")
    }
    if weights != {
        name: records[Path(name).name]["sha256"]
        for name in ASSET_PATHS if name.endswith(".safetensors")
    }:
        raise ValueError("Catalog adapter checksums do not match the release manifest")
    return records


def prepare(request, directory):
    base = f"{HUB}/resolve/{request['hub_revision']}"
    with tempfile.TemporaryDirectory(dir=directory.parent) as staging:
        temporary = Path(staging)
        fetch_asset(base, MANIFEST, temporary / MANIFEST, 1024 * 1024)
        records = load_asset_manifest(request, temporary)
        for name in ASSET_PATHS:
            filename = Path(name).name
            fetch_asset(base, name, temporary / filename, records[filename]["bytes"])
        verify_payload(request, temporary)
        directory.mkdir(parents=True, exist_ok=True)
        for filename in records:
            (temporary / filename).replace(directory / filename)
    print(f"Verified {len(records)} assets at Hub revision {request['hub_revision']}")


def tag_commit(tag):
    ref = f"refs/tags/{tag}"
    lines = command("git", "ls-remote", "origin", ref, f"{ref}^{{}}")
    refs = {}
    for line in lines.splitlines():
        commit, name = line.split("\t")
        refs[name] = commit
    return refs.get(f"{ref}^{{}}", refs.get(ref))


def checked_tag(request, *, required=False):
    existing = tag_commit(request["tag"])
    if (existing is not None or required) and existing != request["source_commit"]:
        raise ValueError("GitHub release tag does not resolve to the reviewed source commit")


def get_release(tag):
    pages = command(
        "gh", "api", "--paginate", f"repos/{REPOSITORY}/releases",
        "--jq", f".[] | select(.tag_name == \"{tag}\") | @json",
    )
    matches = [json.loads(line) for line in pages.splitlines()]
    if len(matches) > 1:
        raise ValueError(f"Multiple GitHub releases use {tag}")
    return matches[0] if matches else None


def missing_assets(release, records):
    uploaded = set()
    for asset in release["assets"]:
        name = asset["name"]
        expected = records.get(name)
        if (expected is None or name in uploaded or asset["state"] != "uploaded"
                or asset["size"] != expected["bytes"]
                or asset["digest"] != f"sha256:{expected['sha256']}"):
            raise ValueError(f"GitHub release contains unexpected asset bytes or state: {name}")
        uploaded.add(name)
    return sorted(records.keys() - uploaded)


def publish(request, directory):
    records = verify_payload(request, directory)
    tag = request["tag"]
    checked_tag(request)
    release = get_release(tag)
    notes = (
        f"Reviewed Supra model bundle from source commit `{request['source_commit']}`.\n\n"
        f"Verified Hub revision: {HUB}/tree/{request['hub_revision']}\n\n"
        f"Manifest SHA-256: `{request['manifest_sha256']}`"
    )
    if release is None:
        command(
            "gh", "release", "create", tag, "--repo", REPOSITORY, "--draft",
            "--target", request["source_commit"], "--title", f"Supra {tag}", "--notes", notes,
        )
        release = get_release(tag)
    if not release["draft"]:
        checked_tag(request, required=True)
        if release["prerelease"] or missing_assets(release, records):
            raise ValueError("Published release does not match the reviewed bundle")
        print(f"Supra {tag} is already published with the exact verified assets")
        return
    if (release["target_commitish"] != request["source_commit"]
            or release["prerelease"] or release["body"].strip() != notes):
        raise ValueError("Existing draft does not match the reviewed Supra release")

    for filename in missing_assets(release, records):
        command("gh", "release", "upload", tag, str(directory / filename), "--repo", REPOSITORY)
    draft = get_release(tag)
    if draft["id"] != release["id"] or missing_assets(draft, records):
        raise ValueError("GitHub did not verify all reviewed assets on the release draft")
    checked_tag(request)
    if draft["draft"]:
        command(
            "gh", "api", "--method", "PATCH", f"repos/{REPOSITORY}/releases/{draft['id']}",
            "-f", f"tag_name={tag}", "-f", f"target_commitish={request['source_commit']}",
            "-F", "draft=false", "-f", "make_latest=legacy",
        )
    checked_tag(request, required=True)
    release = get_release(tag)
    if (not release or release["id"] != draft["id"] or release["draft"]
            or release["prerelease"] or missing_assets(release, records)):
        raise ValueError("Public release readback does not match the verified draft")
    print(f"Published {release['html_url']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "publish"))
    parser.add_argument("--request", type=Path, default=Path("release.json"))
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    request = load_request(args.request)
    verify_source(request["source_commit"])
    if args.action == "prepare":
        args.directory.parent.mkdir(parents=True, exist_ok=True)
        prepare(request, args.directory)
    else:
        publish(request, args.directory)


if __name__ == "__main__":
    main()

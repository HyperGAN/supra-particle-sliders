import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

source = Path(__file__).resolve().parents[1] / "scripts/github_release.py"
spec = importlib.util.spec_from_file_location("supra_github_release", source)
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


class GitHubReleaseTest(unittest.TestCase):
    def setUp(self):
        staging = tempfile.TemporaryDirectory()
        self.addCleanup(staging.cleanup)
        self.directory = Path(staging.name)
        self.request = {
            "tag": "v1.2.3",
            "source_commit": "a" * 40,
            "hub_revision": "b" * 40,
        }
        weights = {}
        for name in release.ASSET_PATHS:
            if name.endswith(".safetensors"):
                path = self.directory / Path(name).name
                path.write_bytes(name.encode())
                weights[name] = release.sha256(path)
        self.write_json("catalog.json", {"sliders": [
            {"original": original, "original_sha256": weights[original],
             "distill": distill, "distill_sha256": weights[distill]}
            for original, distill in zip(
                sorted(p for p in weights if p.startswith("weights/")),
                sorted(p for p in weights if p.startswith("distilled/")),
            )
        ]})
        self.write_json("source-provenance.json", {
            "repository": f"https://github.com/{release.REPOSITORY}",
            "commit": self.request["source_commit"],
        })
        self.write_json(release.MANIFEST, {
            "source_commit": self.request["source_commit"],
            "files": {name: release.asset_record(self.directory / Path(name).name)
                      for name in release.ASSET_PATHS},
        })
        self.request["manifest_sha256"] = release.sha256(self.directory / release.MANIFEST)

    def write_json(self, name, data):
        path = self.directory / name
        path.write_text(json.dumps(data))
        return path

    def uploaded_assets(self):
        records = release.verify_payload(self.request, self.directory)
        return [{"name": name, "state": "uploaded", "size": record["bytes"],
                 "digest": f"sha256:{record['sha256']}"} for name, record in records.items()]

    def test_check_bundle(self):
        path = self.write_json("request.json", self.request)
        self.assertEqual(release.load_request(path), self.request)
        self.assertEqual(len(release.verify_payload(self.request, self.directory)), 7)

    def test_invalid_manifest_digest(self):
        self.request["manifest_sha256"] = "c" * 64
        with self.assertRaisesRegex(ValueError, "reviewed SHA-256"):
            release.verify_payload(self.request, self.directory)

    def test_invalid_weight_digest(self):
        path = self.directory / "final-boss-1600.safetensors"
        path.write_bytes(b"x" * path.stat().st_size)
        with self.assertRaisesRegex(ValueError, "final-boss-1600.safetensors"):
            release.verify_payload(self.request, self.directory)

    def test_source_commit_mismatch(self):
        self.request["source_commit"] = "d" * 40
        with self.assertRaisesRegex(ValueError, "different source commit"):
            release.verify_payload(self.request, self.directory)

    def test_conflicting_tag_stops_before_release_mutation(self):
        with (
            patch.object(release, "tag_commit", return_value="e" * 40),
            patch.object(release, "command") as command,
            self.assertRaisesRegex(ValueError, "tag does not resolve"),
        ):
            release.publish(self.request, self.directory)
        command.assert_not_called()

    def test_public_release_is_idempotent(self):
        existing = {"draft": False, "prerelease": False, "assets": self.uploaded_assets()}
        with (
            patch.object(release, "tag_commit", return_value=self.request["source_commit"]),
            patch.object(release, "get_release", return_value=existing),
            patch.object(release, "command") as command,
        ):
            release.publish(self.request, self.directory)
        command.assert_not_called()

    def test_retry_uploads_only_missing_asset_and_verifies_before_publish(self):
        notes = (
            f"Reviewed Supra model bundle from source commit `{self.request['source_commit']}`.\n\n"
            f"Verified Hub revision: {release.HUB}/tree/{self.request['hub_revision']}\n\n"
            f"Manifest SHA-256: `{self.request['manifest_sha256']}`"
        )
        assets = self.uploaded_assets()
        draft = {"id": 123, "draft": True, "target_commitish": self.request["source_commit"],
                 "prerelease": False, "body": notes, "assets": assets[:-1]}
        verified = dict(draft, assets=assets)
        public = dict(verified, draft=False, html_url="https://github.com/example/releases/tag/v1.2.3")
        with (
            patch.object(release, "tag_commit", side_effect=[None, None, self.request["source_commit"]]),
            patch.object(release, "get_release", side_effect=[draft, verified, public]),
            patch.object(release, "command") as command,
        ):
            release.publish(self.request, self.directory)
        upload, publish = (call.args for call in command.call_args_list)
        self.assertEqual(upload[:4], ("gh", "release", "upload", self.request["tag"]))
        self.assertEqual(Path(upload[4]).name, assets[-1]["name"])
        self.assertEqual(publish[:4], ("gh", "api", "--method", "PATCH"))

    def test_unverified_draft_can_never_go_public(self):
        draft = {"assets": self.uploaded_assets()}
        draft["assets"][-1]["digest"] = None
        with self.assertRaisesRegex(ValueError, "unexpected asset bytes or state"):
            release.missing_assets(draft, release.verify_payload(self.request, self.directory))

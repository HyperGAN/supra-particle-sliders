#!/usr/bin/env python3
"""Audit completed Supra comparison receipts on CPU, without model evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess

import torch
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = Path("/ml2/hypergan/supra-concept-sliders")


class ReviewError(ValueError):
    pass


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def canonical(value):
    return json.loads(json.dumps(value))


def digest_state(value):
    """Independent implementation of the recorded native state digest format."""
    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(str(key).encode())
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode())
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode())
    visit(value)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ReviewError(message)


def close(actual, expected):
    return math.isfinite(actual) and math.isfinite(expected) and math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-12)


def audit_pool(pool, count, *, preservation=False):
    arms = ("old_lora", "new_particlegan")
    require(set(pool) == set(arms), "evaluation pool must contain exactly the two declared arms")
    for arm in arms:
        report = pool[arm]
        records = report["records"]
        require(report["contexts"] == len(records) == count, f"{arm} evaluation context count differs")
        mse = sum(row["mse"] for row in records) / count
        power = sum(row["base_power"] for row in records) / count
        require(all(math.isfinite(row["mse"]) and row["mse"] >= 0 for row in records), f"{arm} invalid recorded MSE")
        require(close(report["mse"], mse) and close(report["rmse"], mse ** .5), f"{arm} MSE/RMSE aggregation differs")
        require(close(report["base_power"], power) and close(report["relative_rms"], (mse / max(power, 1e-12)) ** .5), f"{arm} relative RMS aggregation differs")
        if not preservation:
            gap = sum(row["base_gap_mse"] for row in records) / count
            require(close(report["base_gap_mse"], gap) and close(report["velocity_ratio"], mse / max(gap, 1e-12)), f"{arm} velocity ratio differs")
        groups = {}
        for row in records:
            groups.setdefault(row["prompt"], []).append(row)
        require(set(groups) == set(report["per_subject"]), f"{arm} per-subject identities differ")
        for prompt, group in groups.items():
            subject = report["per_subject"][prompt]
            require(subject["contexts"] == len(group) and close(subject["mse"], sum(row["mse"] for row in group) / len(group)), f"{arm} per-subject aggregation differs")
            if not preservation:
                require(close(subject["velocity_ratio"], sum(row["mse"] for row in group) / max(sum(row["base_gap_mse"] for row in group), 1e-12)), f"{arm} per-subject velocity ratio differs")
        panel = report["common_new_critic"]
        require(panel["draws"] == len(panel["per_draw_generator_loss"]) == 4 and panel["contexts"] == count,
                f"{arm} common critic panel count differs")
        require(close(panel["generator_loss"], sum(panel["per_draw_generator_loss"]) / 4), f"{arm} common critic draw mean differs")
        require(panel["evaluation_only"] is True, "game panels must be evaluation only")
    old, new = (pool[arm] for arm in arms)
    for left, right in zip(old["records"], new["records"]):
        require(all(left[key] == right[key] for key in ("index", "prompt", "t", "base_power", "base_gap_mse")),
                "old/new contexts or live target/base quantities differ")
    for key in ("critic_state_sha256", "paired_noise_sha256", "output_sigma", "noise_rng", "law"):
        require(old["common_new_critic"][key] == new["common_new_critic"][key], f"common critic {key} differs between arms")


def audit_endpoints(endpoints):
    for arm in ("old_lora", "new_particlegan"):
        report = endpoints[arm]
        rows = report["records"]
        require(report["trajectories"] == len(rows) == 12, f"{arm} endpoint count differs")
        require(close(report["mse"], sum(row["mse"] for row in rows) / 12) and
                close(report["mean_endpoint_ratio"], sum(row["ratio"] for row in rows) / 12), f"{arm} endpoint aggregation differs")
        require(all(close(row["ratio"], row["mse"] / max(row["base_gap_mse"], 1e-12)) for row in rows), f"{arm} endpoint ratios differ")
    for old, new in zip(endpoints["old_lora"]["records"], endpoints["new_particlegan"]["records"]):
        require(all(old[key] == new[key] for key in ("name", "prompt", "seed", "base_gap_mse", "initial_latent_sha256")),
                "old/new endpoint inputs differ")


def audit_probe_receipt(report, baseline, probes):
    path = baseline.parent / "validation.json"
    published = read_json(path).get("velocity_probes") if path.is_file() else None
    if path.is_file():
        require(report.get("receipt_sha256") == sha(path), "original probe receipt hash differs")
    if not published:
        require(report.get("available", False) is False and "maximum_absolute_metric_difference" not in report,
                "missing original probes must not claim zero-error reproduction")
        return dict(available=False, reason=report.get("reason", "original probes unavailable"))
    require(report.get("available", True) is True and len(published) == len(probes), "original probe count differs")
    key = lambda row: (row["name"], round(row["t"], 7))
    current = {key(row): row for row in probes}
    require(len(current) == len(probes) and {key(row) for row in published} == set(current), "original probe identities differ")
    differences = [abs(value - current[key(row)][name]) for row in published for name, value in row.items()
                   if isinstance(value, (int, float))]
    require(differences and close(report["maximum_absolute_metric_difference"], max(differences)), "original probe reproduction aggregate differs")
    return dict(available=True, maximum_absolute_metric_difference=max(differences))


def build_review(run_dir, *, allow_source_changes=False):
    run_dir = Path(run_dir).resolve()
    if run_dir.is_file():
        run_dir = run_dir.parent
    folder = run_dir / "evaluation"
    require((folder / "status.json").is_file() and read_json(folder / "status.json").get("phase") == "complete",
            "evaluation is incomplete; review performs no evaluation itself")
    run, comparison = read_json(run_dir / "run.json"), read_json(folder / "comparison.json")
    horizon = run["fixed_updates"]
    require(type(horizon) is int and horizon > 0 and comparison["completed_steps"] == horizon, "fixed final update horizon differs")
    require(not allow_source_changes or horizon == 1600, "historical source changes may be allowed only for the original 1600-step receipt")
    require(comparison.get("declared_updates", horizon) == comparison.get("baseline_completed_steps", horizon) == horizon,
            "declared baseline/new training budgets differ")
    baseline = Path(run.get("original_baseline_path", ORIGINAL / "outputs/final-boss-supra-1600/final-boss-supra.safetensors"))
    require(comparison["baseline_sha256"] == run["original_baseline_sha256"] == sha(baseline), "original baseline hash differs")
    with safe_open(str(baseline), framework="pt", device="cpu") as handle:
        require(json.loads(handle.metadata()["step"]) == horizon, "original baseline embedded step differs")
    require(comparison["checkpoint_sha256"] == sha(run_dir / "final.pt") and comparison["data_sha256"] == sha(run_dir / "data.pt"),
            "native checkpoint or dataset hash differs")
    require(comparison["prompts_sha256"] == run["prompts_sha256"] == sha(ROOT / "configs/supra/prompts-final-boss.yaml"), "prompt recipe hash differs")
    require(comparison["evaluation_only"] is True and comparison["live_target_batch_size"] == 4,
            "evaluation boundary or paired batch shape differs")
    require(comparison["training_state_unchanged"] is True and comparison["strength_zero_exact"] is True,
            "native training state or strength-zero check failed")
    require(canonical(comparison["task"]) == canonical(run["config"]), "evaluation task differs from declared training configuration")
    require(len(comparison["task"]["sites"]) == 71 and comparison["task"]["frozen_published_loras"] == 0,
            "comparison does not cover all 71 fresh particle branches")
    changed_sources = [name for name, expected in comparison["source_sha256"].items() if sha(ROOT / name) != expected]
    require(not changed_sources or allow_source_changes, f"evaluation checkout sources changed: {', '.join(changed_sources)}")
    archive = run_dir / "source/sha256.json"
    archive_report = dict(available=archive.is_file())
    if archive.is_file():
        hashes = read_json(archive)
        def archived_source(name):
            path = run_dir / "source" / name
            return run_dir / "source" / Path(name).name if horizon == 1600 and not path.exists() else path
        require(all(sha(archived_source(name)) == expected for name, expected in hashes.items()), "archived training sources changed")
        changed_training = [name for name, expected in hashes.items() if sha(ROOT / name) != expected]
        require(not changed_training or allow_source_changes, "training checkout sources changed")
        archive_report.update(files_verified=len(hashes), manifest_sha256=sha(archive),
                              changed_training_checkout_sources=changed_training)
    elif horizon != 1600:
        raise ReviewError("continued training requires its source archive")
    pg = Path(run["particlegan_root"])
    commit = subprocess.check_output(["git", "-C", str(pg), "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "-C", str(pg), "status", "--porcelain"], text=True).strip())
    sources = {path.name: sha(path) for path in sorted((pg / "particlegan").glob("*.py"))}
    require(commit == run["particlegan_commit"] and dirty is False and run["particlegan_dirty"] is False and
            digest_state(sources) == run["particlegan_source_digest"], "ParticleGAN clean commit/source pin differs")
    lineage = run.get("resumed_from")
    if lineage:
        parent = Path(lineage["checkpoint"]).parent
        require(lineage["checkpoint_sha256"] == sha(lineage["checkpoint"]) and lineage["run_sha256"] == sha(parent / "run.json"),
                "continuation checkpoint/run lineage hash differs")
        require(lineage["data_sha256"] == sha(parent / "data.pt") == comparison["data_sha256"], "continuation dataset lineage differs")
        require(read_json(parent / "run.json")["fixed_updates"] == lineage["completed_steps"] < horizon,
                "continuation does not start from the previous fixed horizon")
    audit_pool(comparison["validation"], 240)
    audit_pool(comparison["preservation"], 60, preservation=True)
    audit_endpoints(comparison["endpoints"])
    probe_report = audit_probe_receipt(comparison["published_baseline_reproduction"], baseline,
                                      comparison["published_probes"]["old_lora"])
    state = torch.load(run_dir / "final.pt", map_location="cpu", weights_only=False, mmap=True)
    require(state["policy"]["completed_steps"] == horizon and canonical(state["config"]) == canonical(run["config"]),
            "actual native checkpoint step/config differs")
    selected = state["policy"]["averages" if comparison["served_source"] == "averaged" else "models"]
    critic_digest = digest_state(selected["critic"])
    require(all(comparison[pool][arm]["common_new_critic"]["critic_state_sha256"] == critic_digest
                for pool in ("validation", "preservation") for arm in ("old_lora", "new_particlegan")), "scoring critic differs from the selected native checkpoint")
    export = comparison["adapter_export"]
    adapter = run_dir / Path(export["path"]).name
    require(sha(adapter) == export["sha256"] and adapter.stat().st_size == export["bytes"], "lean adapter hash/size differs")
    require(export["reload_exact"] is True and export["reload_max_absolute_difference"] == 0. and export["strength_zero_exact"] is True,
            "lean adapter native BF16 reload failed")
    with safe_open(str(adapter), framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
        exported_config = json.loads(metadata["config"])
        pins = json.loads(metadata["native_pins"])
        require(metadata["format"] == "supra_particlegan_clean_v1" and exported_config["completed_steps"] == horizon and
                exported_config["served_source"] == comparison["served_source"], "lean adapter law/step differs")
        require(pins["model_revision"] == run["model_revision"] and pins["text_encoder_revision"] == run["text_revision"] and
                pins["vae_revision"] == run["vae_revision"] and pins["backend_sha256"] == read_json(ROOT / "backend.lock.json")["sha256"],
                "lean adapter native model/backend pins differ")
        require(exported_config["sites"] == run["config"]["sites"] and canonical(exported_config["extra"]["training"]) == canonical(run) and
                exported_config["extra"]["checkpoint_sha256"] == comparison["checkpoint_sha256"] and
                exported_config["extra"]["data_sha256"] == comparison["data_sha256"], "lean adapter declared training lineage differs")
    exported = load_file(str(adapter), device="cpu")
    expected = {"generator." + key: value for key, value in selected["generator"].items()
                if any(key.startswith(f"model.{site}.{part}.") for site in run["config"]["sites"] for part in ("down", "bridge", "up"))}
    expected.update({"router." + key: value for key, value in selected["router"].items() if key != "log_mass"})
    expected.update({"bank.log_mass": selected["router"]["log_mass"],
                     "bank.table": state["policy"]["averaged_table" if comparison["served_source"] == "averaged" else "table"]})
    require(set(expected) == set(exported) and len(exported) == export["tensors"] and
            sum(value.numel() for value in exported.values()) == export["parameters"] and
            all(torch.equal(value, exported[key]) for key, value in expected.items()), "lean export tensors differ from the selected native checkpoint")
    del state, selected, expected, exported
    samples = read_json(folder / "samples/metadata.json")["samples"]
    require(comparison["renders"]["images"] == len(samples) == 32 and comparison["renders"]["matched_initial_latents"] is True,
            "render count or matched latent check differs")
    groups = {}
    for sample in samples:
        path = folder / "samples" / sample["file"]
        require(sha(path) == sample["image_sha256"], "render image hash differs")
        with Image.open(path) as picture:
            require(picture.size == (256, 256) and picture.mode == "RGB", "render is not native 256px RGB")
        groups.setdefault((sample["name"], sample["seed"]), []).append(sample)
    require(len(groups) == 8 and all(len(rows) == 4 and {row["arm"] for row in rows} ==
            {"off", "old_lora", "new_particlegan", "positive_teacher"} and
            len({row["initial_latent_sha256"] for row in rows}) == 1 for rows in groups.values()), "render arm/initial-latent matching differs")
    metrics = {}
    for name, pool, key in (("validation_velocity_rmse", "validation", "rmse"),
                            ("mean_endpoint_ratio", "endpoints", "mean_endpoint_ratio"),
                            ("preservation_relative_rms", "preservation", "relative_rms")):
        old, new = (comparison[pool][arm][key] for arm in ("old_lora", "new_particlegan"))
        metrics[name] = dict(original=old, new=new, new_relative_change_percent=100 * (new / old - 1))
    return dict(schema="supra-full-comparison-cpu-review-v1", all_checks_passed=True, fixed_updates=horizon,
                comparison_sha256=sha(folder / "comparison.json"), run_sha256=sha(run_dir / "run.json"),
                metrics=metrics, training_source_archive=archive_report,
                historical_source_changes_allowed=allow_source_changes, changed_evaluation_checkout_sources=changed_sources,
                continuation_lineage_verified=bool(lineage), original_probe_reproduction=probe_report,
                native_export_tensors_verified=export["tensors"], rendered_images_verified=32,
                scope="One ordered initialization; six subjects and two validation draws with correlated trajectory contexts. Full architecture, initialization, game and optimizer differ. Output metrics are reporting only.",
                game_panel_limit="One common critic trained with the new model; four paired-noise draws are diagnostics, not optimizer replications or independent image-quality evidence.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="New review receipt; defaults to RUN/evaluation/review.json")
    parser.add_argument("--allow-source-changes", action="store_true", help="Explicit historical 1600-step receipt audit; record changed checkout sources")
    args = parser.parse_args()
    torch.set_num_threads(8)
    run_dir = args.run.parent if args.run.is_file() else args.run
    output = args.output or run_dir / "evaluation/review.json"
    try:
        require(not output.exists(), "review receipt already exists; choose a new output to preserve it")
        result = build_review(run_dir, allow_source_changes=args.allow_source_changes)
    except ReviewError as error:
        print(json.dumps(dict(event="review_rejected", reason=str(error)), sort_keys=True), flush=True)
        raise SystemExit(2) from None
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(event="review_complete", path=str(output), **result), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

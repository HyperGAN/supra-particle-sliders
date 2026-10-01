#!/usr/bin/env python3
"""Read-only CPU qualification of the fixed Supra V1/V2 particle comparison."""
from argparse import ArgumentParser
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.review_e22_supra_full import canonical, digest_state, require
from scripts.review_e22_supra_training_controls import evaluate_check, finite, paired_pool

PIN = "f459cb6d6aaaabeb1af076ec53ad7a963618de90"
ARCHITECTURE = "linear_modulated_v2"
UPDATES = 1600


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frozen_parameters(state):
    return {role: {name: state["policy"]["models"][role][name]
                   for name, trainable in state["policy"]["requires_grad"][role].items() if not trainable}
            for role in ("generator", "encoder")}


def audit_trace(rows, old_rows, data):
    require(len(rows) == len(old_rows) == UPDATES, "both traces must contain the complete fixed horizon")
    require([row["step"] for row in rows] == list(range(1, UPDATES + 1)), "native update clock changed")
    finite(rows)
    stream = torch.Generator().manual_seed(7)
    paired_keys = ("step", "hold", "batch_indices", "base_noise_sums", "paired_rng_digest")
    for row, old in zip(rows, old_rows):
        step = row["step"]
        hold = step % 5 == 0
        require(row["hold"] is hold and row["game_weight"] == (.1 if hold else 1.), "preservation schedule changed")
        pool = data["holds" if hold else "fit"]["context"]
        expected_indices = torch.randint(len(pool), (4,), generator=stream).tolist()
        require(row["batch_indices"] == expected_indices, f"original CPU data stream changed at {step}")
        require(all(row[key] == old[key] for key in paired_keys), f"paired training draws differ at {step}")
        require(row["penalty_calls"] == step, "native critic penalty clock changed")
        require(row["dense_gradient_rows"] == (0 if step == 1 else 128), f"bank gradients are not dense at {step}")
        require(row["bank_grad_norm"] >= 0 and row["output_sigma"] > 0, "invalid training gradient or output sigma")
    return dict(updates=UPDATES, preservation_updates=sum(row["hold"] for row in rows),
                all_rows_finite=True, dense_bank_after_update_two=True,
                data_and_paired_streams_equal_archived_trace=True,
                data_rng=stream.get_state(),
                dv12_rng_digests_equal_steps=sum(row["dv12_rng_digest"] == old["dv12_rng_digest"]
                                                 for row, old in zip(rows, old_rows)))


def audit_export(path, receipt, state, run):
    export = receipt["export"]
    require(path.stat().st_size == export["bytes"], "export byte count differs")
    source = receipt["controls"]["served_source"]
    require(source in ("fast", "averaged"), "invalid served source")
    selected = state["policy"]["models" if source == "fast" else "averages"]
    trainable = {name for name, flag in state["policy"]["requires_grad"]["generator"].items() if flag}
    expected = {"generator." + name: selected["generator"][name] for name in trainable}
    expected.update({"router." + name: value for name, value in selected["router"].items() if name != "log_mass"})
    expected.update({"bank.table": state["policy"]["table" if source == "fast" else "averaged_table"],
                     "bank.log_mass": selected["router"]["log_mass"]})
    require(len(expected) == export["tensors"] == 428, "adapter does not contain the complete 71-site schema")
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
        config = json.loads(metadata["config"])
        require(metadata["format"] == "supra_particlegan_clean_v2" and config["architecture"] == ARCHITECTURE,
                "export formula/version metadata differs")
        require(config == export["config"], "export receipt config differs")
        require(config["sites"] == run["config"]["sites"] and len(config["sites"]) == 71, "export sites differ")
        require((config["rank"], config["z_dim"], config["num_particles"], config["cfg"]) == (16, 4, 128, 3.),
                "export dimensions or CFG differ")
        require(config["sampling"] == "clean" and config["routed_geometry"] == "mass_atoms_v1", "export routing law differs")
        require(config["completed_steps"] == UPDATES and config["served_source"] == source, "export horizon/source differs")
        require(config["extra"]["selection"] == "fixed 1600-update horizon; output metrics evaluation only", "export selection changed")
        pins = json.loads(metadata["native_pins"])
        for key, original in (("model_revision", "model_revision"), ("text_encoder_revision", "text_revision"),
                              ("vae_revision", "vae_revision")):
            require(pins[key] == run[original], "export native model revision differs")
        require(pins["backend_sha256"] == read(ROOT / "backend.lock.json")["sha256"], "export backend pin differs")
        require(set(handle.keys()) == set(expected), "export tensor names differ")
        for name, reference in expected.items():
            value = handle.get_tensor(name)
            require(value.dtype == torch.float32 and torch.equal(value, reference), f"export tensor differs: {name}")
            valid = ((torch.isfinite(value) | torch.isneginf(value)).all() and torch.isfinite(value).any()
                     if name == "bank.log_mass" else torch.isfinite(value).all())
            require(bool(valid), f"invalid export tensor: {name}")
    parameters = sum(value.numel() for value in expected.values())
    require(parameters == export["parameters"], "export parameter count differs")
    return dict(sha256=sha(path), tensors=len(expected), parameters=parameters,
                metadata_and_schema_valid=True, all_values_equal_selected_native_checkpoint=True)


def build_review(directory):
    directory = Path(directory).resolve()
    require((directory / "receipt.json").is_file(), "experiment receipt is not complete")
    require(read(directory / "status.json")["phase"] == "complete", "experiment status is not complete")
    receipt, plan, run = (read(directory / name) for name in ("receipt.json", "plan.json", "run.json"))
    require(receipt["plan"] == run["architecture_experiment"] == plan, "experiment plan differs between receipts")
    require(plan["fixed_updates"] == run["fixed_updates"] == UPDATES, "fixed horizon changed")
    require(plan["architectures"] == ["nonlinear_v1", ARCHITECTURE], "architecture comparison changed")
    require(plan["output_metrics"] == "evaluation only; never optimization, guard or structural criterion", "metric boundary changed")
    require(run["config"]["output_error_guard"] is False and run["config"]["max_feature_context_harm"] == 0.,
            "structural criterion is not the unchanged native feature game")
    require(run["config"]["checkpoint_selection"] == "final fixed update horizon; evaluation only", "checkpoint selection changed")
    manifest = read(directory / "source/sha256.json")
    require(manifest == plan["source_sha256"], "source manifest differs from predeclared plan")
    for name, expected in manifest.items():
        require(sha(directory / "source" / name) == expected and sha(ROOT / name) == expected, f"frozen source changed: {name}")
    reference = Path(plan["reference_run"])
    old_run = read(reference / "run.json")
    require(plan["particlegan_commit"] == run["particlegan_commit"] == old_run["particlegan_commit"] == PIN,
            "ParticleGAN source pin differs")
    package = Path(run["particlegan_root"]) / "particlegan"
    pg_hashes = {path.name: sha(path) for path in sorted(package.glob("*.py"))}
    require(digest_state(pg_hashes) ==
            plan["particlegan_source_digest"] == run["particlegan_source_digest"] == old_run["particlegan_source_digest"],
            "ParticleGAN archived source digest differs")
    committed = subprocess.check_output(["git", "-C", old_run["particlegan_root"], "archive", PIN, "particlegan"])
    with tarfile.open(fileobj=io.BytesIO(committed)) as archive:
        committed_hashes = {Path(member.name).name: hashlib.sha256(archive.extractfile(member).read()).hexdigest()
                            for member in archive.getmembers() if member.isfile() and
                            Path(member.name).parent == Path("particlegan") and member.name.endswith(".py")}
    require(pg_hashes == committed_hashes, "archived ParticleGAN package differs from the actual pinned git tree")
    require(sha(reference / "final.pt") == plan["reference_checkpoint_sha256"], "V1 comparator checkpoint changed")
    require(sha(reference / "smoke-resume.pt") == plan["reference_step_two_sha256"], "V1 step-two comparator changed")
    require(sha(directory / "data.pt") == sha(reference / "data.pt") == plan["data_sha256"], "training data bytes differ")
    original = torch.load(reference / "final.pt", map_location="cpu", weights_only=False, mmap=True)
    old_two = torch.load(reference / "smoke-resume.pt", map_location="cpu", weights_only=False, mmap=True)
    parity, smoke = read(directory / "legacy-parity.json"), read(directory / "smoke.json")
    require(parity["native_step_two_exact"] is True and parity["trace_rows_exact"] is True and
            parity["state_digest"] == digest_state(old_two), "V1 step-two replay witness differs from archived state")
    require(original["policy"]["completed_steps"] == UPDATES and old_two["policy"]["completed_steps"] == 2,
            "archived comparator horizons differ")
    for flag in ("initial_native_state_identical", "legacy_step_two_exact", "strength_zero_exact", "exact_resume"):
        require(smoke[flag] is True, f"native preflight witness failed: {flag}")
    require(smoke["dense_gradient_rows"] == 128, "preflight bank gradient count differs")
    for flag in ("full_initial_native_state_identical", "legacy_step_two_replay_exact", "native_resume_exact",
                 "frozen_unchanged", "export_reload_exact", "evaluation_state_unchanged", "data_and_paired_streams_matched"):
        require(receipt[flag] is True, f"native runtime witness failed: {flag}")
    data = torch.load(directory / "data.pt", map_location="cpu", weights_only=False)
    require(digest_state(data) == original["config"]["dataset_digest"] == run["config"]["dataset_digest"], "data content digest differs")
    state = torch.load(directory / "final.pt", map_location="cpu", weights_only=False, mmap=True)
    require(state["policy"]["completed_steps"] == UPDATES and canonical(state["config"]) == run["config"], "final native step/config differs")
    require(state["config"]["architecture"] == ARCHITECTURE and
            {key: value for key, value in state["config"].items() if key != "architecture"} == original["config"],
            "architecture is not the only native configuration change")
    require(digest_state(state) == receipt["final_state_digest"], "final native state differs from evaluation witness")
    require(digest_state(frozen_parameters(state)) == digest_state(frozen_parameters(original)), "frozen host/teacher changed")
    rows = [json.loads(line) for line in (directory / "train.jsonl").read_text().splitlines() if line.strip()]
    old_rows = [json.loads(line) for line in (reference / "train.jsonl").read_text().splitlines() if line.strip()]
    trace_report = audit_trace(rows, old_rows, data)
    require(torch.equal(trace_report.pop("data_rng"), state["data_rng"]), "final CPU data RNG differs from exact replay")
    require(torch.equal(state["paired_noise_rng"], original["paired_noise_rng"]), "final paired-noise RNG differs from V1")
    evaluations = {architecture: read(directory / f"evaluation-{architecture}.json") for architecture in ("v1", "v2")}
    for architecture, evaluation in evaluations.items():
        finite(evaluation, architecture)
        evaluate_check(evaluation, {"metrics": receipt["metrics"][architecture]}, data)
    pairing = {pool: paired_pool(evaluations["v1"][pool], evaluations["v2"][pool], data["prompts"])
               for pool in ("fit", "test", "holds", "preservation")}
    metrics = {pool: {architecture: {"rmse": evaluation[pool]["rmse"], "relative_rms": evaluation[pool]["relative_rms"],
                                     "archived_v1_final_critic_g_loss": evaluation[pool]["fixed_start_D"]["g_game"],
                                     "new_v2_final_critic_g_loss": evaluation[pool]["arm_final_D"]["g_game"]}
                     for architecture, evaluation in evaluations.items()} for pool in pairing}
    return dict(schema="supra-particle-architecture-cpu-qualification-v1", all_checks_passed=True,
                run=str(directory), fixed_updates=UPDATES, source_files_verified=len(manifest),
                initial_native_state_digest_recorded=parity["initial_native_digest"],
                archived_step_two_digest_verified=parity["state_digest"], final_native_state_digest_verified=receipt["final_state_digest"],
                trace=trace_report, export=audit_export(directory / "final.safetensors", receipt, state, run),
                metrics=metrics, paired_context_scores=pairing,
                critic_semantics={"fixed_start_D": "Frozen archived V1 FINAL step-1600 critic; name is inherited from the continuation helper.",
                                  "arm_final_D": "Frozen new V2 FINAL step-1600 critic, shared by both evaluation arms."},
                scoring_critic_state_digests={"archived_v1_final_1600": digest_state(original["policy"]["models"]["critic"]),
                                             "new_v2_final_1600": digest_state(state["policy"]["models"]["critic"])},
                evidence_limits=["CPU review checks persisted state, tensors, sources and receipts. Initial equality, GPU resume, export forward parity and V1 restore are recorded runtime witnesses, not rerun here.",
                                 "V1 final checkpoint is reused after exact fresh two-update replay; the complete V1 training trajectory is not rerun.",
                                 "One task and one deterministic training stream; held-out trajectory contexts are correlated. Critic scores are endogenous diagnostics, not independent image-quality evidence.",
                                 "No endpoints or rendered images are evaluated by this fixed architecture experiment. Output metrics are reporting only."],
                file_hashes={name: sha(directory / name) for name in ("plan.json", "run.json", "receipt.json", "final.pt", "train.jsonl",
                                                                      "evaluation-v1.json", "evaluation-v2.json")})


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "outputs/e22-particle-v2-1600")
    parser.add_argument("--output", type=Path, help="Default: RUN/qualification-review.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    output = args.output or args.run / "qualification-review.json"
    require(not output.exists(), "qualification output already exists; preserve the prior evidence")
    report = build_review(args.run)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(all_checks_passed=True, output=str(output), metrics=report["metrics"])))


if __name__ == "__main__":
    main()

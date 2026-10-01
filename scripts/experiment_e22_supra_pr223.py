#!/usr/bin/env python3
"""Fixed-budget PR223 shared-profile comparison against qualified old Supra.

The new run continues its qualified fresh step-two initialization. The control
is the archived f459 V2 trajectory, admitted by exact common-owner equality at
step two. Output metrics are final diagnostics; they never select a variant.
"""
import argparse
from copy import deepcopy
import gc
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PIN = "bdf05d1be0f68cfdb0c71e81e7e0d3cce477572f"
OLD_PIN = "f459cb6d6aaaabeb1af076ec53ad7a963618de90"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_tensor(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(type(value).__name__)


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=json_tensor) + "\n")
    temporary.replace(path)


def emit(**row):
    print(json.dumps(row, allow_nan=False, default=json_tensor), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/e22-pr223-shared-1600")
    parser.add_argument("--control", type=Path, default=ROOT / "outputs/e22-particle-v2-1600")
    parser.add_argument("--initial", type=Path, default=ROOT / "outputs/e22-pr223-shared-profile-smoke")
    parser.add_argument("--judge", type=Path, default=ROOT / "outputs/e22-particle-noise-update-256/latest")
    parser.add_argument("--particlegan-root", type=Path, default=ROOT.parent / "ParticleGAN-pr223-supra")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("choose a fresh output directory")
    pg_root = args.particlegan_root.resolve()
    if subprocess.check_output(["git", "-C", str(pg_root), "rev-parse", "HEAD"], text=True).strip() != PIN:
        raise RuntimeError("unexpected ParticleGAN checkout")
    if subprocess.check_output(["git", "-C", str(pg_root), "status", "--porcelain", "--", "particlegan"], text=True).strip():
        raise RuntimeError("ParticleGAN source is dirty")
    sys.path.insert(0, str(pg_root))
    sys.path.insert(1, str(ROOT))
    import torch
    import particlegan
    from supra.runtime import TARGETS, model_module
    from supra.particle_adapter import LINEAR_MODULATED_V2
    from supra.particle_game import ConditionalTokenCritic
    from supra.particle_pilot import checkpoint, frozen_digest, restore, state_digest
    from supra.particle_training import LEGACY_ROUTED_PROFILE, SHARED_ROUTED_PROFILE, make_training_loop, training_update
    from monitor_e22_supra_particle_convergence import GameProgressMonitor, rolling_losses
    from diagnose_e22_supra_training_controls import evaluate_final

    if Path(particlegan.__file__).resolve().parent.parent != pg_root:
        raise RuntimeError("unexpected ParticleGAN import")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.set_num_threads(8)
    started = time.perf_counter()
    args.output.mkdir(parents=True)
    checks = {}

    def require(name, condition):
        checks[name] = bool(condition)
        if not checks[name]:
            write(args.output / "failed-checks.json", checks)
            raise RuntimeError(name)

    initial_result = json.loads((args.initial / "result.json").read_text())
    initial_review = json.loads((args.initial / "independent-review.json").read_text())
    old_run = json.loads((args.control / "run.json").read_text())
    old_review = json.loads((args.control / "qualification-review.json").read_text())
    require("qualified_shared_initialization", initial_result["qualified"] and all(initial_result["checks"].values())
            and initial_review.get("qualified", initial_review.get("all_checks_passed", False)))
    require("qualified_historical_control", old_review["all_checks_passed"]
            and old_run["particlegan_commit"] == OLD_PIN and old_run["fixed_updates"] == 1600)
    data = torch.load(args.control / "data.pt", map_location="cpu", weights_only=False)
    initial = torch.load(args.initial / "checkpoint-00002.pt", map_location="cpu", weights_only=False, mmap=True)
    old_initial = torch.load(args.control / "checkpoint-00002.pt", map_location="cpu", weights_only=False, mmap=True)
    require("shared_start_digest_qualified", state_digest(initial) == initial_result["checkpoint_native_digest"])
    require("identical_cached_data", state_digest(data) == initial["config"]["dataset_digest"]
            == old_initial["config"]["dataset_digest"])
    require("architecture_and_initial_step", initial["config"]["architecture"] == LINEAR_MODULATED_V2
            and initial["policy"]["completed_steps"] == old_initial["policy"]["completed_steps"] == 2)
    common_owners = {key: state_digest(initial["policy"][key]) == state_digest(value)
                     for key, value in old_initial["policy"].items() if key != "recipe"}
    require("all_common_step_two_owners_exact", all(common_owners.values()))
    require("application_rngs_exact", all(torch.equal(initial[key], old_initial[key]) for key in ("data_rng", "paired_noise_rng")))
    old_rows = [json.loads(line) for line in (args.control / "train.jsonl").read_text().splitlines()]
    rows = initial_result["rows"][:2]
    require("prefix_trace_exact", rows == old_rows[:2] and len(old_rows) == 1600)
    sources = [Path(__file__), ROOT / "scripts/monitor_e22_supra_particle_convergence.py",
               ROOT / "scripts/diagnose_e22_supra_training_controls.py", ROOT / "supra/runtime.py"]
    sources += sorted((ROOT / "supra").glob("particle*.py"))
    app_hashes = {str(path.relative_to(ROOT)): sha(path) for path in sources}
    pg_hashes = {str(path.relative_to(pg_root)): sha(path) for path in sorted((pg_root / "particlegan").rglob("*.py"))}
    for path in sources:
        target = args.output / "source" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    for name in pg_hashes:
        target = args.output / "source/particlegan" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pg_root / name, target)
    input_paths = {"data": args.control / "data.pt", "old_trace": args.control / "train.jsonl",
                   "old_run": args.control / "run.json", "old_review": args.control / "qualification-review.json",
                   "shared_initial": args.initial / "checkpoint-00002.pt", "shared_review": args.initial / "independent-review.json",
                   "judge": args.judge / "final.pt"}
    input_paths.update({f"old_{step}": args.control / f"checkpoint-{step:05d}.pt" for step in (400, 800, 1200, 1600)})
    input_hashes = {key: sha(path) for key, path in input_paths.items()}
    plan = dict(schema="supra_pr223_matched_1600_v1", fixed_updates=1600, start_step=2,
                profile=SHARED_ROUTED_PROFILE, particlegan_commit=PIN, control_commit=OLD_PIN,
                control_directory=str(args.control.resolve()), initialization="qualified fresh public deterministic_orthogonal_",
                initial_common_owners_exact=common_owners, input_paths={key: str(path.resolve()) for key, path in input_paths.items()},
                input_sha256=input_hashes, application_source_sha256=app_hashes, particlegan_source_sha256=pg_hashes,
                training="same cached contexts, paired noise, DV12, shared128x4 bank, 71 sites, V2; preservation each fifth update",
                structural="native learned critic features, zero context feature harm, no output guard",
                selection="predeclared1600 horizon; no output-based stopping, hyperparameter or checkpoint selection",
                comparison_steps=[400, 800, 1200, 1600], progress_every=200,
                common_judge="frozen V2 cabe step1856 for all curves and both final endpoints",
                final_judges="same frozen step1856 and same new final1600 critic for both endpoints",
                perturbation_panels="same private CPU72 Gaussian panels and native DV12 stream fixed at common step2",
                output_metrics="final evaluation only", limitation="one deterministic task/stream; routed guard integration, Atlas cells inactive")
    write(args.output / "plan.json", plan)
    write(args.output / "run.json", dict(plan, config=initial["config"]))
    write(args.output / "status.json", dict(phase="preparing_control", step=2, steps=1600, backend="routed", particle_profile=SHARED_ROUTED_PROFILE))
    emit(event="plan", fixed_updates=1600, control_commit=OLD_PIN, profile=SHARED_ROUTED_PROFILE)
    with torch.random.fork_rng(devices=[device.index or 0]):
        mod = model_module()
        with torch.device("meta"):
            base = mod.SupraDiT()
            mod.attach_supra_lora(base, rank=16, alpha=16, targets=TARGETS)
        base.load_state_dict({key.removeprefix("teacher."): value for key, value in
                              initial["policy"]["models"]["encoder"].items() if key.startswith("teacher.")}, strict=True, assign=True)
        base.to(device).eval().requires_grad_(False)
        fixed = ConditionalTokenCritic(data["coordinate_scale"]).to(device)
        judge = torch.load(args.judge / "final.pt", map_location="cpu", weights_only=False, mmap=True)
        fixed.load_state_dict(judge["policy"]["models"]["critic"], strict=True)
        fixed.eval().requires_grad_(False)
        require("judge_qualified_horizon", judge["policy"]["completed_steps"] == 1856)
        plan["fixed_critic_digest"] = state_digest(fixed.state_dict())
        del judge
        legacy = make_training_loop(base, data, device=device, architecture=LINEAR_MODULATED_V2,
                                    profile=LEGACY_ROUTED_PROFILE)
    common_dv12 = initial["policy"]["streams"]["noise_generator"].clone()
    control_progress = []
    for step in plan["comparison_steps"]:
        saved = torch.load(input_paths[f"old_{step}"], map_location="cpu", weights_only=False, mmap=True)
        restore(legacy, saved)
        require(f"control_{step}_strict_restore", state_digest(checkpoint(legacy)) == state_digest(saved))
        monitor = GameProgressMonitor(legacy, data, fixed, args.output / f"control-probes-{step}")
        monitor.dv12_state = common_dv12.clone()
        probe = monitor.evaluate(legacy, old_rows[:step])
        control_progress.append(probe)
        write(args.output / "comparison-control-progress.json", control_progress)
        emit(event="control_progress", step=step, probes=probe["probes"])
        del saved, monitor
    del legacy, old_initial
    gc.collect()
    torch.cuda.empty_cache()
    with torch.random.fork_rng(devices=[device.index or 0]):
        loop = make_training_loop(base, data, device=device, architecture=LINEAR_MODULATED_V2, profile=SHARED_ROUTED_PROFILE)
    restore(loop, initial)
    require("shared_initial_full_state_exact", state_digest(checkpoint(loop)) == state_digest(initial))
    frozen = frozen_digest(loop)
    monitor = GameProgressMonitor(loop, data, fixed, args.output)
    monitor.dv12_state = common_dv12.clone()
    write(args.output / "plan.json", plan)
    emit(event="game_progress", **monitor.evaluate(loop, rows))
    del initial
    gc.collect()
    training_started = time.perf_counter()
    exact_trace_count = 2

    def activity():
        policy = loop.policy
        def diagnostic(value):
            # Native acquisition statistics may be undefined. Report these as
            # missing observations; never rewrite native checkpoint tensors.
            if isinstance(value, torch.Tensor):
                return diagnostic(value.detach().cpu().tolist())
            if isinstance(value, float) and not math.isfinite(value):
                return None
            if isinstance(value, dict):
                return {key: diagnostic(item) for key, item in value.items()}
            if isinstance(value, (tuple, list)):
                return [diagnostic(item) for item in value]
            return value
        return diagnostic(dict(optimizer_surprise_fires=policy.surprise.fires, anchor_release_events=policy.surprise.anchor_events,
                    epoch_rebases=policy.reopen_guard.epoch_rebases, contracted_network=policy._contracted_network(),
                    accepted_moves=policy.routed_control.counters["moves"], reopen_guard=policy.reopen_guard.state_dict(),
                    lr_settle=policy.lr_settle.diagnostics()))

    def save(path):
        temporary = path.with_suffix(".tmp")
        torch.save(checkpoint(loop), temporary)
        temporary.replace(path)

    with (args.output / "train.jsonl").open("w") as trace:
        for row in rows:
            trace.write(json.dumps(row) + "\n")
        for step in range(3, 1601):
            row = training_update(loop)
            control = old_rows[step - 1]
            for key in ("batch_indices", "base_noise_sums", "paired_rng_digest", "hold", "game_weight"):
                require(f"stream_match_{step}_{key}", row[key] == control[key])
            exact_trace_count += int(row == control)
            rows.append(row)
            trace.write(json.dumps(row) + "\n")
            if step % 25 == 0 or step == 1600:
                trace.flush()
                elapsed = time.perf_counter() - training_started
                status = dict(phase="training", step=step, steps=1600, seconds=elapsed,
                              backend="routed", particle_profile=SHARED_ROUTED_PROFILE,
                              guard_activity=activity(), exact_old_trace_rows=exact_trace_count,
                              rolling=rolling_losses(rows))
                write(args.output / "status.json", status)
                emit(event="training", step=step, seconds=elapsed, exact_old_trace_rows=exact_trace_count,
                     rolling=status["rolling"], guard_activity=status["guard_activity"])
            if step % 200 == 0:
                emit(event="game_progress", **monitor.evaluate(loop, rows))
            if step % 400 == 0:
                save(args.output / f"checkpoint-{step:05d}.pt")
    training_seconds = time.perf_counter() - training_started
    save(args.output / "final.pt")
    require("new_frozen_weights_unchanged", frozen_digest(loop) == frozen)
    native_digest = state_digest(checkpoint(loop))
    final_critic = deepcopy(loop.policy.D).eval().requires_grad_(False)
    write(args.output / "status.json", dict(phase="evaluating", step=1600, steps=1600, backend="routed",
                                           particle_profile=SHARED_ROUTED_PROFILE, guard_activity=activity()))
    new_metrics = evaluate_final(loop.policy.served_model(), data, fixed, final_critic, device)
    require("new_evaluation_read_only", state_digest(checkpoint(loop)) == native_digest)
    saved_old = torch.load(input_paths["old_1600"], map_location="cpu", weights_only=False, mmap=True)
    new_native = checkpoint(loop)["policy"]
    owner_equal = {key: state_digest(new_native[key]) == state_digest(saved_old["policy"][key])
                   for key in ("models", "averages", "table", "averaged_table", "optimizers", "controller", "routing", "lr_settle", "streams")}
    del new_native
    with torch.random.fork_rng(devices=[device.index or 0]):
        legacy = make_training_loop(base, data, device=device, architecture=LINEAR_MODULATED_V2, profile=LEGACY_ROUTED_PROFILE)
    restore(legacy, saved_old)
    old_digest = state_digest(checkpoint(legacy))
    old_metrics = evaluate_final(legacy.policy.served_model(), data, fixed, final_critic, device)
    require("old_evaluation_read_only", state_digest(checkpoint(legacy)) == old_digest)
    comparison = {pool: dict(count=old_metrics[pool]["count"], old_rmse=old_metrics[pool]["rmse"],
                 new_rmse=new_metrics[pool]["rmse"], rmse_change=new_metrics[pool]["rmse"] - old_metrics[pool]["rmse"],
                 **{judge: dict(old_g_game=old_metrics[pool][judge]["g_game"], new_g_game=new_metrics[pool][judge]["g_game"],
                     new_minus_old=new_metrics[pool][judge]["g_game"] - old_metrics[pool][judge]["g_game"])
                    for judge in ("fixed_start_D", "arm_final_D")}) for pool in old_metrics}
    write(args.output / "evaluation-new.json", new_metrics)
    write(args.output / "evaluation-old.json", old_metrics)
    require("application_sources_unchanged", all(sha(ROOT / name) == value for name, value in app_hashes.items()))
    require("native_sources_unchanged", all(sha(pg_root / name) == value for name, value in pg_hashes.items()))
    require("inputs_unchanged", all(sha(input_paths[key]) == value for key, value in input_hashes.items()))
    receipt = dict(qualified=True, plan=plan, checks=checks, comparison=comparison, model_and_owner_equal=owner_equal,
                   final_native_digest=native_digest, final_checkpoint_sha256=sha(args.output / "final.pt"),
                   old_final_native_digest=old_digest, training_seconds=training_seconds,
                   total_seconds=time.perf_counter() - started, exact_old_trace_rows=exact_trace_count,
                   controller=activity(), evaluation_state_unchanged=True,
                   critic_labels={"fixed_start_D": "common frozen V2 cabe step1856", "arm_final_D": "common PR223 final1600"})
    write(args.output / "receipt.json", receipt)
    write(args.output / "status.json", dict(phase="complete", step=1600, steps=1600, backend="routed",
                                           particle_profile=SHARED_ROUTED_PROFILE, guard_activity=activity(),
                                           comparison=comparison, exact_old_trace_rows=exact_trace_count))
    emit(event="complete", comparison=comparison, model_and_owner_equal=owner_equal, native_digest=native_digest)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run a fixed-budget native Supra ParticleGAN pilot; tail stdout for progress."""
import argparse
import gc
import hashlib
from importlib import metadata
import json
import subprocess
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import particlegan

from supra.runtime import SupraRuntime
from supra.particle_game import update
from supra.particle_pilot import checkpoint, evaluate, frozen_digest, initial_digest, make_loop, restore, state_digest


def emit(value):
    print(json.dumps(value, sort_keys=True, default=str), flush=True)


def checked_update(loop):
    with torch.autograd.set_multithreading_enabled(False):
        row = update(loop)
    for name in ("loss_d", "loss_g", "penalty", "bank_grad_norm", "output_sigma"):
        if not torch.isfinite(torch.tensor(row[name])):
            raise RuntimeError(f"nonfinite {name}: {row}")
    return row


def optimizer_provenance():
    """Record the imported checkout or installed wheel's VCS provenance."""
    root = Path(particlegan.__file__).resolve().parents[1]
    commit, dirty = None, None
    try:
        repository = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=root,
                                             text=True, stderr=subprocess.DEVNULL).strip()
        if Path(repository).resolve() == root:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
            dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        pass
    if commit is None:
        try:
            distribution = metadata.distribution("particlegan")
            if Path(distribution.locate_file("particlegan/__init__.py")).resolve() == Path(particlegan.__file__).resolve():
                direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
                commit = direct_url.get("vcs_info", {}).get("commit_id")
        except metadata.PackageNotFoundError:
            pass
    sources = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
               for path in sorted(Path(particlegan.__file__).parent.glob("*.py"))}
    return dict(particlegan_commit=commit, particlegan_root=str(root), particlegan_dirty=dirty,
                particlegan_source_digest=state_digest(sources))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("outputs/e22-pilot/data.pt"))
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/e22-pilot/verification"))
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--probe-interval", type=int, default=20)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    data = torch.load(args.data, map_location="cpu", weights_only=False)
    provenance = dict(data_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
                      teacher_sha256=hashlib.sha256(args.teacher.read_bytes()).hexdigest(),
                      **optimizer_provenance(),
                      dataset_metadata=data["metadata"], steps=args.steps,
                      checkpoint_selection="final fixed update horizon; no metric selection")
    if provenance["teacher_sha256"] != data["metadata"]["adapter_sha256"]:
        raise ValueError("teacher weights do not match the dataset's published teacher")
    runtime = SupraRuntime(device="cuda:0", rank=16, allow_hub=False)
    runtime.load(args.teacher)
    runtime.model.requires_grad_(False)
    emit(dict(event="start", provenance=provenance))
    loop = make_loop(runtime.model, data, mode="movable", probe_interval=args.probe_interval)
    frozen = frozen_digest(loop)
    first = checked_update(loop)
    second = checked_update(loop)
    if second["dense_gradient_rows"] != 128 or second["bank_grad_norm"] <= 0:
        raise RuntimeError(f"no dense shared bank gradient by second update: {second}")
    saved = checkpoint(loop)
    smoke_path = args.output / "smoke-resume.pt"
    torch.save(saved, smoke_path)
    del saved
    uninterrupted = [checked_update(loop), checked_update(loop)]
    expected_digest = state_digest(checkpoint(loop))
    restore(loop, torch.load(smoke_path, map_location="cuda:0", weights_only=False))
    resumed = [checked_update(loop), checked_update(loop)]
    resumed_digest = state_digest(checkpoint(loop))
    exact_resume = expected_digest == resumed_digest and uninterrupted == resumed
    smoke = dict(event="smoke", first=first, second=second, exact_resume=exact_resume,
                 frozen_unchanged=frozen == frozen_digest(loop), evaluations=evaluate(loop, data))
    served = loop.policy.served_model()
    probe_zero = data["guard"]["context"].repeat(6, 1)[:64].to(loop.policy.device)
    probe_zero[:, 4098] = 0
    zero_residual = served.routed_forward(probe_zero)
    smoke["matched_teacher_probe_batch"] = dict(contexts=64, exact_zero=bool(zero_residual.eq(0).all()),
                                                max_error=float(zero_residual.abs().max()))
    del served, zero_residual, probe_zero
    emit(smoke)
    (args.output / "smoke.json").write_text(json.dumps(smoke, indent=2) + "\n")
    if (not exact_resume or not smoke["frozen_unchanged"] or not smoke["evaluations"]["zero_strength"]["exact"]
            or not smoke["matched_teacher_probe_batch"]["exact_zero"]):
        raise RuntimeError("real-model smoke failed integrity/recovery/base-strength checks")
    del loop
    gc.collect()
    torch.cuda.empty_cache()
    if args.smoke_only:
        return
    results = {}
    initial_hashes, matched_inputs = {}, {}
    for mode in ("fixed", "movable", "full"):
        loop = make_loop(runtime.model, data, mode=mode, probe_interval=args.probe_interval)
        initial_hashes[mode] = initial_digest(loop)
        frozen = frozen_digest(loop)
        initial = evaluate(loop, data)
        emit(dict(event="initial", mode=mode, evaluation=initial, config=loop.config))
        trace = []
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(args.steps):
            row = checked_update(loop)
            trace.append(row)
            if row["step"] % 10 == 0 or (row["move"] and row["move"].get("moves", 0) > 0) or row["step"] == args.steps:
                emit(dict(event="train", mode=mode, **row))
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        final = evaluate(loop, data)
        if frozen != frozen_digest(loop) or not final["zero_strength"]["exact"]:
            raise RuntimeError(f"frozen/base invariant failed for {mode}")
        matched_inputs[mode] = state_digest([(row["batch_indices"], row["base_noise_sums"], row["paired_rng_digest"], row["dv12_rng_digest"]) for row in trace])
        result = dict(config=loop.config, initial=initial, final=final, seconds=seconds,
                      optimizer_surprise=(None if loop.policy.surprise is None else loop.policy.surprise.diagnostics()),
                      accepted_moves=[row for row in trace if row["move"] and row["move"].get("moves", 0) > 0],
                      frozen_unchanged=True, final_step=loop.policy.completed_steps)
        results[mode] = result
        (args.output / f"{mode}-trace.json").write_text(json.dumps(trace, indent=2) + "\n")
        torch.save(checkpoint(loop), args.output / f"{mode}-final.pt")
        emit(dict(event="complete", mode=mode, result=result))
        del loop
        gc.collect()
        torch.cuda.empty_cache()
    report = dict(provenance=provenance, smoke=smoke, results=results, initial_hashes=initial_hashes,
                  initial_weights_matched=len(set(initial_hashes.values())) == 1,
                  matched_inputs=matched_inputs, training_inputs_matched=len(set(matched_inputs.values())) == 1,
                  limitations=["Only two of 71 published LoRA branches are replaced; 69 remain frozen.",
                               "Current structural decisions use the clean learned critic-feature proxy.",
                               "Each arm learns its own critic; feature errors are not a common ranking metric.",
                               "This fixed-budget pilot does not establish persistent game repair."])
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    emit(dict(event="report", path=str(args.output / "results.json"),
              initial_weights_matched=report["initial_weights_matched"],
              training_inputs_matched=report["training_inputs_matched"]))
    if not report["initial_weights_matched"] or not report["training_inputs_matched"]:
        raise RuntimeError("matched-arm integrity failed")


if __name__ == "__main__":
    main()

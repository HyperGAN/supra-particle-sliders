#!/usr/bin/env python3
"""Fixed 1600-update particle-architecture experiment on the real Supra task.

The archived V1 comparator is admitted only after exact native step-two replay.
V2 is trained from identical initialized tensors, using the same native game,
data stream, preservation schedule and ParticleGAN source. Evaluation cannot
change the budget, structural decisions or checkpoint selection.
"""
import argparse
from copy import deepcopy
import gc
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PIN = "f459cb6d6aaaabeb1af076ec53ad7a963618de90"
UPDATES = 1600


def emit(**row):
    print(json.dumps(row, default=str), flush=True)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, default=str) + "\n")
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=ROOT / "outputs/e22-full-f459cb6d")
    parser.add_argument("--particlegan-root", type=Path,
                        default=ROOT / "outputs/e22-convergence-gap/particlegan-f459-source")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/e22-particle-v2-1600")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("choose a fresh output directory")
    sys.path.insert(0, str(args.particlegan_root.resolve()))
    sys.path.insert(1, str(ROOT))
    import torch
    import particlegan
    from supra.runtime import model_module, TARGETS
    from supra.particle_adapter import LINEAR_MODULATED_V2, NONLINEAR_V1
    from supra.particle_export import export_served_adapter, load_particle_adapter
    from supra.particle_game import ConditionalTokenCritic
    from supra.particle_pilot import checkpoint, restore, frozen_digest, state_digest
    from supra.particle_training import make_training_loop, training_update, raw_velocity
    from diagnose_e22_supra_training_controls import evaluate_final

    if not torch.cuda.is_available():
        raise RuntimeError("native Supra/BF16 experiment requires CUDA")
    torch.set_num_threads(8)
    torch.cuda.set_device(torch.device(args.device))
    imported = Path(particlegan.__file__).resolve()
    reference_receipt = json.loads((args.reference / "run.json").read_text())
    if imported.parent.parent != args.particlegan_root.resolve():
        raise RuntimeError("ParticleGAN import differs from the declared archived source")
    pg_hashes = {path.name: sha(path) for path in sorted(imported.parent.glob("*.py"))}
    if reference_receipt["particlegan_commit"] != PIN or state_digest(pg_hashes) != reference_receipt["particlegan_source_digest"]:
        raise RuntimeError("native source differs from the archived comparator")
    saved = torch.load(args.reference / "final.pt", map_location="cpu", weights_only=False, mmap=True)
    old_two = torch.load(args.reference / "smoke-resume.pt", map_location="cpu", weights_only=False, mmap=True)
    data = torch.load(args.reference / "data.pt", map_location="cpu", weights_only=False)
    if saved["policy"]["completed_steps"] != UPDATES or old_two["policy"]["completed_steps"] != 2:
        raise RuntimeError("unexpected archived comparison horizon")
    if state_digest(data) != saved["config"]["dataset_digest"]:
        raise RuntimeError("archived training data changed")
    if saved["config"].get("architecture", NONLINEAR_V1) != NONLINEAR_V1:
        raise RuntimeError("the archived comparator must use V1")
    if str(torch.device(args.device)) != saved["policy"]["device"]:
        raise RuntimeError("the exact native comparison requires the original logical device")

    args.output.mkdir(parents=True)
    shutil.copy2(args.reference / "data.pt", args.output / "data.pt")
    source_paths = [Path(__file__), ROOT / "scripts/diagnose_e22_supra_training_controls.py"]
    source_paths += [ROOT / "supra" / name for name in (
        "particle_adapter.py", "particle_export.py", "particle_training.py", "particle_game.py",
        "particle_pilot.py", "particle_training_data.py", "runtime.py")]
    source_hashes = {str(path.relative_to(ROOT)): sha(path) for path in source_paths}
    for source in source_paths:
        target = args.output / "source" / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    write_json(args.output / "source/sha256.json", source_hashes)
    plan = dict(fixed_updates=UPDATES, reference_run=str(args.reference.resolve()),
                reference_checkpoint_sha256=sha(args.reference / "final.pt"),
                reference_step_two_sha256=sha(args.reference / "smoke-resume.pt"),
                data_sha256=sha(args.output / "data.pt"), particlegan_commit=PIN,
                particlegan_source_digest=state_digest(pg_hashes), source_sha256=source_hashes,
                architectures=[NONLINEAR_V1, LINEAR_MODULATED_V2],
                change="up(tanh(bridge([hidden, particle_code]))) becomes up(hidden + tanh(bridge([hidden, particle_code])))",
                retained="shared 128x4 particle bank; all 71 routers; DV12; native game, optimizers and structural controllers",
                comparator="archived V1 at 1600, admitted only after exact fresh two-update replay",
                selection="final fixed horizon, no output-metric selection or stopping",
                evaluation="complete original fit/test/holds/preservation pools, clean BF16/CFG3, common archived D plus new final D",
                output_metrics="evaluation only; never optimization, guard or structural criterion",
                limitation="single task and training stream; architecture comparison, not a claim of general superiority")
    write_json(args.output / "plan.json", plan)
    emit(event="plan", **plan)
    started = time.perf_counter()

    # Load only the exact frozen native teacher. No T5/VAE reload or trained
    # ordinary LoRA is introduced into either particle branch initialization.
    module = model_module()
    with torch.random.fork_rng(devices=[]), torch.device("meta"):
        base = module.SupraDiT()
        module.attach_supra_lora(base, rank=16, alpha=16, targets=TARGETS)
    teacher = {key.removeprefix("teacher."): value for key, value in saved["policy"]["models"]["encoder"].items()
               if key.startswith("teacher.")}
    base.load_state_dict(teacher, strict=True, assign=True)
    base.to(args.device).eval().requires_grad_(False)
    del teacher

    def reset_global_streams():
        # These auxiliary global streams are part of native checkpoints.
        # Training draws use their separately owned native/data streams.
        torch.set_rng_state(old_two["policy"]["cpu_rng"].cpu())
        torch.cuda.set_rng_state(old_two["policy"]["cuda_rng"].cpu(), args.device)

    def new_loop(architecture):
        reset_global_streams()
        return make_training_loop(base, data, device=args.device,
                                  probe_interval=saved["config"]["probe_interval"],
                                  branch_lr=saved["config"]["branch_lr"], architecture=architecture)

    legacy = new_loop(NONLINEAR_V1)
    initial_native = state_digest(checkpoint(legacy)["policy"])
    initial_data = legacy.data_rng.get_state().clone()
    initial_paired = legacy.paired_noise_rng.get_state().clone()
    legacy_rows = [training_update(legacy), training_update(legacy)]
    actual_two = state_digest(checkpoint(legacy))
    archived_two = state_digest(old_two)
    old_trace = [json.loads(line) for line in (args.reference / "train.jsonl").read_text().splitlines()]
    if actual_two != archived_two or legacy_rows != old_trace[:2]:
        # Preserve the failed parity evidence; never substitute an unqualified
        # historical comparator or silently loosen its native-state contract.
        actual = checkpoint(legacy)
        differences = {key: state_digest(actual["policy"][key]) != state_digest(old_two["policy"][key])
                       for key in actual["policy"]}
        write_json(args.output / "parity-failed.json", dict(actual=actual_two, expected=archived_two,
                   policy_differences=differences, fresh_rows=legacy_rows, archived_rows=old_trace[:2]))
        raise RuntimeError("fresh V1 failed exact replay of the archived native step-two state")
    write_json(args.output / "legacy-parity.json", dict(native_step_two_exact=True,
               trace_rows_exact=True, state_digest=actual_two, initial_native_digest=initial_native))
    emit(event="legacy_replay_passed", step=2, full_native_state_exact=True, trace_exact=True)
    del legacy
    gc.collect()
    torch.cuda.empty_cache()

    loop = new_loop(LINEAR_MODULATED_V2)
    if state_digest(checkpoint(loop)["policy"]) != initial_native:
        raise RuntimeError("V2 initialized native tensors/state differ from V1")
    if not torch.equal(loop.data_rng.get_state(), initial_data) or not torch.equal(loop.paired_noise_rng.get_state(), initial_paired):
        raise RuntimeError("V2 did not start from the identical sampled streams")
    original_frozen = frozen_digest(loop)
    provenance = dict(reference_receipt, config=loop.config, fixed_updates=UPDATES,
                      particlegan_root=str(args.particlegan_root.resolve()),
                      comparison="fixed native-game V2 particle architecture versus exact-replayed archived V1",
                      architecture_experiment=plan)
    write_json(args.output / "run.json", provenance)

    def save(path):
        tmp = path.with_suffix(".tmp")
        torch.save(checkpoint(loop), tmp)
        tmp.replace(path)

    # Fixed preflight proves zero start, bank gradients and exact continuation.
    served = loop.policy.served_model()
    context = data["fit"]["context"][:4].to(args.device).clone()
    context[:, 4099] = 0
    # Strength zero removes branch deltas. Pair the frozen teacher with the
    # source caption rather than the training task's positive target caption.
    source = context.clone()
    source[:, 4098] = source[:, 4097]
    if not torch.equal(raw_velocity(served, context), served.encoder.teacher_velocity(source)):
        raise RuntimeError("V2 strength-zero forward differs from the frozen native source")
    del served, context, source
    rows = [training_update(loop), training_update(loop)]
    if rows[-1]["dense_gradient_rows"] != 128:
        raise RuntimeError("V2 shared particle bank lacks dense gradients after two updates")
    save(args.output / "checkpoint-00002.pt")
    reference = [training_update(loop), training_update(loop)]
    exact_after_four = state_digest(checkpoint(loop))
    restore(loop, torch.load(args.output / "checkpoint-00002.pt", map_location="cpu", weights_only=False, mmap=True))
    replay = [training_update(loop), training_update(loop)]
    if reference != replay or exact_after_four != state_digest(checkpoint(loop)):
        raise RuntimeError("V2 exact native replay failed")
    rows += reference
    for row, old_row in zip(rows, old_trace):
        if any(row[key] != old_row[key] for key in ("hold", "batch_indices", "base_noise_sums", "paired_rng_digest")):
            raise RuntimeError("V2 preflight data or paired stream changed")
    write_json(args.output / "smoke.json", dict(initial_native_state_identical=True, legacy_step_two_exact=True,
               strength_zero_exact=True, exact_resume=True, dense_gradient_rows=128))
    emit(event="smoke_passed", initial_native_state_identical=True, exact_resume=True, dense_gradient_rows=128)
    training_started = time.perf_counter()
    with (args.output / "train.jsonl").open("w") as trace:
        for row in rows:
            trace.write(json.dumps(row) + "\n")
        for step in range(5, UPDATES + 1):
            row = training_update(loop)
            if row["step"] != step:
                raise RuntimeError("native update clock changed")
            reference_sampling = old_trace[step - 1]
            for key in ("hold", "batch_indices", "base_noise_sums", "paired_rng_digest"):
                if row[key] != reference_sampling[key]:
                    raise RuntimeError(f"training stream changed at {step}: {key}")
            rows.append(row)
            trace.write(json.dumps(row) + "\n")
            if step % 25 == 0:
                trace.flush()
                write_json(args.output / "status.json", dict(phase="training", step=step, steps=UPDATES,
                           training_seconds=time.perf_counter() - training_started))
                emit(event="training", seconds=time.perf_counter() - training_started, **row)
            if step % 400 == 0:
                save(args.output / f"checkpoint-{step:05d}.pt")
    save(args.output / "final.pt")
    if original_frozen != frozen_digest(loop):
        raise RuntimeError("frozen host or teacher changed")
    final_digest = state_digest(checkpoint(loop))
    final_critic = deepcopy(loop.policy.D).eval().requires_grad_(False)
    fixed_state = saved["policy"]["models"]["critic"]
    with torch.random.fork_rng(devices=[]):
        fixed_critic = ConditionalTokenCritic(fixed_state["scale"]).to(args.device).eval().requires_grad_(False)
    fixed_critic.load_state_dict(fixed_state, strict=True)
    served = loop.policy.served_model()
    export = export_served_adapter(served, args.output / "final.safetensors", extra_metadata={
        "selection": "fixed 1600-update horizon; output metrics evaluation only"})
    with torch.random.fork_rng(devices=[torch.device(args.device).index or 0]):
        clean = load_particle_adapter(base, args.output / "final.safetensors", device=args.device)
    context = data["test"]["context"][:4].to(args.device)
    z, t, ctx, mask, uctx, umask, strength = served.encoder.unpack(context)
    if not torch.equal(raw_velocity(served, context), clean.velocity(z, t, ctx, mask, uctx, umask, strength=strength)):
        raise RuntimeError("V2 clean export reload differs from native served output")
    del clean, context
    v2_metrics = evaluate_final(served, data, fixed_critic, final_critic, args.device)
    if final_digest != state_digest(checkpoint(loop)):
        raise RuntimeError("export or evaluation mutated native training state")
    write_json(args.output / "evaluation-v2.json", v2_metrics)
    controls = dict(controller=loop.policy.controller.diagnostics(),
                    accepted_moves=[row["move"] for row in rows if isinstance(row["move"], dict) and row["move"].get("moves", 0)],
                    bank_rows=len(loop.policy.table), served_source=served.source,
                    generator_lrs=[group["lr"] for group in loop.policy.opt_g.param_groups])
    del served, loop
    gc.collect()
    torch.cuda.empty_cache()

    legacy = new_loop(NONLINEAR_V1)
    restore(legacy, saved)
    if state_digest(checkpoint(legacy)) != state_digest(saved):
        raise RuntimeError("archived final V1 did not restore exactly")
    before_eval = state_digest(checkpoint(legacy))
    served = legacy.policy.served_model()
    legacy_metrics = evaluate_final(served, data, fixed_critic, final_critic, args.device)
    if before_eval != state_digest(checkpoint(legacy)):
        raise RuntimeError("V1 evaluation mutated native training state")
    write_json(args.output / "evaluation-v1.json", legacy_metrics)
    if any(sha(ROOT / name) != value for name, value in source_hashes.items()):
        raise RuntimeError("experiment source changed during execution")
    summary = dict(plan=plan, full_initial_native_state_identical=True, legacy_step_two_replay_exact=True,
                   native_resume_exact=True, frozen_unchanged=True, export_reload_exact=True,
                   evaluation_state_unchanged=True, data_and_paired_streams_matched=True,
                   final_state_digest=final_digest, controls=controls, export=export,
                   seconds=time.perf_counter() - started,
                   metrics={"v1": {name: {k: v for k, v in values.items() if k != "records"} for name, values in legacy_metrics.items()},
                            "v2": {name: {k: v for k, v in values.items() if k != "records"} for name, values in v2_metrics.items()}})
    write_json(args.output / "receipt.json", summary)
    write_json(args.output / "status.json", dict(phase="complete", step=UPDATES, steps=UPDATES))
    emit(event="complete", **summary)


if __name__ == "__main__":
    main()

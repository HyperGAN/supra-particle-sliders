#!/usr/bin/env python3
"""Fixed-horizon native particle continuation, preserving the complete game.

The default extends the qualified latest-source V2 model from 1856 to 6400
updates. No evaluation controls training or selects a checkpoint. Frozen
native weights are loaded directly from the owned teacher checkpoint.
"""
import argparse
from copy import deepcopy
from pathlib import Path
import shutil
import sys
import time
import json

ROOT = Path(__file__).resolve().parents[1]
PIN = "cabe2084284db923d525918cbf3e18de6f20faac"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "outputs/e22-particle-noise-update-256/latest")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/e22-particle-v2-6400")
    parser.add_argument("--steps", type=int, default=6400)
    parser.add_argument("--checkpoint-every", type=int, default=400)
    parser.add_argument("--probe-every", type=int, default=200)
    parser.add_argument("--particlegan-root", type=Path,
                        default=ROOT / "outputs/e22-convergence-gap/particlegan-cabe2084-source")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("choose a fresh continuation output directory")
    if args.checkpoint_every < 1 or args.probe_every < 1:
        parser.error("checkpoint/progress intervals must be positive")
    args.particlegan_root = args.particlegan_root.resolve()
    sys.path.insert(0, str(args.particlegan_root))
    sys.path.insert(1, str(ROOT))
    import torch
    import particlegan
    from experiment_e22_supra_noise_update import sha, write_json, emit
    from supra.runtime import model_module, TARGETS
    from supra.particle_adapter import LINEAR_MODULATED_V2
    from supra.particle_pilot import checkpoint, restore, state_digest, frozen_digest
    from supra.particle_training import make_training_loop, training_update, raw_velocity
    from supra.particle_export import export_served_adapter, load_particle_adapter
    from supra.particle_game import ConditionalTokenCritic, _rng_digest
    from diagnose_e22_supra_training_controls import evaluate_final
    from monitor_e22_supra_particle_convergence import GameProgressMonitor, rolling_losses

    if not torch.cuda.is_available():
        raise RuntimeError("native Supra/BF16 continuation requires CUDA")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.set_num_threads(8)
    imported = Path(particlegan.__file__).resolve()
    if imported.parent.parent != args.particlegan_root:
        raise RuntimeError("native source import differs from the declared archive")
    pg_paths = sorted(imported.parent.rglob("*.py"))
    pg_hashes = {str(path.relative_to(imported.parent)): sha(path) for path in pg_paths}
    pg_digest = state_digest({path.name: pg_hashes[path.name] for path in imported.parent.glob("*.py")})
    previous = json.loads((args.run / "run.json").read_text())
    previous_receipt = json.loads((args.run / "receipt.json").read_text())
    if previous["particlegan_commit"] != PIN or previous["particlegan_source_digest"] != pg_digest:
        raise RuntimeError("continuation native source differs from its qualified checkpoint")
    for field in ("initial_native_state_exact", "two_update_replay_exact", "frozen_unchanged",
                  "export_reload_exact", "evaluation_state_unchanged", "source_and_inputs_unchanged"):
        if previous_receipt.get(field) is not True:
            raise RuntimeError(f"starting run lacks qualification: {field}")
    input_paths = {name: args.run / name for name in ("final.pt", "data.pt", "run.json", "receipt.json")}
    input_hashes = {name: sha(path) for name, path in input_paths.items()}
    saved = torch.load(input_paths["final.pt"], map_location="cpu", weights_only=False, mmap=True)
    data = torch.load(input_paths["data.pt"], map_location="cpu", weights_only=False)
    initial_digest = state_digest(saved)
    if initial_digest != previous_receipt["final_native_digest"]:
        raise RuntimeError("starting native checkpoint differs from its receipt")
    start = int(saved["policy"]["completed_steps"])
    if not start < args.steps or start != previous["fixed_updates"]:
        parser.error("the fixed horizon must extend the qualified starting step")
    if saved["config"].get("architecture") != LINEAR_MODULATED_V2:
        raise RuntimeError("this continuation preserves V2 exactly")
    if json.loads(json.dumps(saved["config"])) != previous["config"]:
        raise RuntimeError("starting native configuration differs from its run receipt")
    if state_digest(data) != saved["config"]["dataset_digest"] or str(device) != saved["policy"]["device"]:
        raise RuntimeError("native dataset/device differs from its owned checkpoint")
    if saved["config"]["output_error_guard"] or saved["config"]["max_feature_context_harm"] != 0:
        raise RuntimeError("unexpected structural criterion or output guard")
    args.output.mkdir(parents=True)
    shutil.copy2(input_paths["data.pt"], args.output / "data.pt")
    source_paths = [Path(__file__), ROOT / "scripts/experiment_e22_supra_noise_update.py",
                    ROOT / "scripts/diagnose_e22_supra_training_controls.py",
                    ROOT / "scripts/monitor_e22_supra_particle_convergence.py"]
    source_paths += [ROOT / "supra" / name for name in (
        "particle_adapter.py", "particle_export.py", "particle_game.py", "particle_pilot.py",
        "particle_training.py", "particle_training_data.py", "runtime.py")]
    source_hashes = {str(path.relative_to(ROOT)): sha(path) for path in source_paths}
    for source in source_paths:
        target = args.output / "source" / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    for source in pg_paths:
        target = args.output / "source/particlegan" / source.relative_to(imported.parent)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    write_json(args.output / "source/sha256.json", dict(application=source_hashes, particlegan=pg_hashes))
    original = Path(f"/ml2/hypergan/supra-concept-sliders/outputs/final-boss-supra-converged/checkpoint-{args.steps:05d}/final-boss-supra.safetensors")
    if not original.is_file():
        raise RuntimeError("the declared fixed-horizon original comparison checkpoint is missing")
    plan = dict(start_step=start, fixed_updates=args.steps, additional_updates=args.steps - start,
                initial_native_digest=initial_digest, resumed_from=str(args.run.resolve()),
                input_sha256=input_hashes, particlegan_commit=PIN, particlegan_root=str(args.particlegan_root),
                particlegan_source_digest=pg_digest, architecture=LINEAR_MODULATED_V2,
                retained="shared 128x4 bank, 71 routing sites, DV12, all native optimizers and structural controls",
                change="only additional fixed-budget updates; exact configuration and owned streams preserved",
                original_baseline_path=str(original), original_baseline_sha256=sha(original),
                selection="final fixed horizon; no metric-dependent stopping/checkpoint selection",
                progress="separate rolling edit/preservation game losses every25; frozen-critic training probes every200",
                probe_every=args.probe_every,
                output_metrics="evaluation only; not loss, optimizer, row criterion or guard",
                evaluation="complete fit/test/holds/preservation, batch4/BF16/CFG3, common start critic and final critic",
                critic_labels={"fixed_start_D": f"common V2 step{start} critic", "arm_final_D": f"V2 final step{args.steps} critic"},
                limitation="single task and stream; equal-update original comparison requires separate shared-critic evaluation")
    write_json(args.output / "plan.json", plan)
    run = dict(previous, fixed_updates=args.steps, additional_updates=args.steps - start,
               comparison=f"particle V2 versus original ordinary-LoRA recipe at {args.steps} updates",
               original_baseline_path=str(original), original_baseline_sha256=sha(original), continuation=plan)
    write_json(args.output / "run.json", run)
    emit(event="plan", **plan)
    started = time.perf_counter()
    with torch.random.fork_rng(devices=[device.index or 0]):
        module = model_module()
        with torch.device("meta"):
            base = module.SupraDiT()
            module.attach_supra_lora(base, rank=16, alpha=16, targets=TARGETS)
        base.load_state_dict({key.removeprefix("teacher."): value for key, value in
                            saved["policy"]["models"]["encoder"].items() if key.startswith("teacher.")},
                            strict=True, assign=True)
        base.to(device).eval().requires_grad_(False)
        loop = make_training_loop(base, data, device=device, architecture=LINEAR_MODULATED_V2,
                                  probe_interval=saved["config"]["probe_interval"], branch_lr=saved["config"]["branch_lr"])
    restore(loop, saved)
    if initial_digest != state_digest(checkpoint(loop)):
        raise RuntimeError("initial full native state did not restore exactly")
    frozen = frozen_digest(loop)
    with torch.random.fork_rng(devices=[device.index or 0]):
        fixed = ConditionalTokenCritic(saved["policy"]["models"]["critic"]["scale"]).to(device)
        fixed.load_state_dict(saved["policy"]["models"]["critic"], strict=True)
        fixed.eval().requires_grad_(False)
    monitor = GameProgressMonitor(loop, data, fixed, args.output)
    emit(event="game_progress", **monitor.evaluate(loop, []))
    first = [training_update(loop), training_update(loop)]
    first_digest = state_digest(checkpoint(loop))
    restore(loop, saved)
    expected_data = torch.Generator().set_state(saved["data_rng"].cpu())
    expected_paired = torch.Generator(device=device).set_state(saved["paired_noise_rng"].cpu())
    rows = []

    def save(path):
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(checkpoint(loop), tmp)
        tmp.replace(path)

    training_started = time.perf_counter()
    with (args.output / "train.jsonl").open("w") as trace:
        for step in range(start + 1, args.steps + 1):
            hold = step % 5 == 0
            indices = torch.randint(len(data["holds" if hold else "fit"]["context"]), (4,), generator=expected_data)
            noises = [torch.randn((4, 256, 16), device=device, generator=expected_paired) for _ in range(2)]
            expected = dict(step=step, hold=hold, batch_indices=indices.tolist(),
                            base_noise_sums=[float(noise.sum()) for noise in noises], paired_rng_digest=_rng_digest(expected_paired))
            row = training_update(loop)
            if any(row[key] != value for key, value in expected.items()):
                raise RuntimeError(f"owned application stream changed at {step}")
            rows.append(row)
            trace.write(json.dumps(row) + "\n")
            if step == start + 2:
                if rows != first or first_digest != state_digest(checkpoint(loop)):
                    raise RuntimeError("native exact replay failed")
                write_json(args.output / "restore-review.json", dict(initial_native_state_exact=True, two_update_replay_exact=True))
                emit(event="resume_passed", step=step, initial_native_state_exact=True, two_update_replay_exact=True)
            if step % 25 == 0 or step == args.steps:
                trace.flush()
                seconds = time.perf_counter() - training_started
                write_json(args.output / "status.json", dict(phase="training", step=step, steps=args.steps, seconds=seconds))
                emit(event="training", seconds=seconds, rolling=rolling_losses(rows), **row)
            if step % args.probe_every == 0 or step == args.steps:
                emit(event="game_progress", **monitor.evaluate(loop, rows))
            if step % args.checkpoint_every == 0:
                save(args.output / f"checkpoint-{step:05d}.pt")
    save(args.output / "final.pt")
    if not torch.equal(loop.data_rng.get_state(), expected_data.get_state()) or not torch.equal(loop.paired_noise_rng.get_state(), expected_paired.get_state()):
        raise RuntimeError("final owned RNG states differ from the fixed sampled program")
    if frozen != frozen_digest(loop):
        raise RuntimeError("frozen host/teacher changed")
    final_digest = state_digest(checkpoint(loop))
    final_critic = deepcopy(loop.policy.D).eval().requires_grad_(False)
    served = loop.policy.served_model()
    export = export_served_adapter(served, args.output / "final.safetensors", extra_metadata={
        "selection": "final fixed horizon; output metrics evaluation only", "fixed_updates": args.steps})
    with torch.random.fork_rng(devices=[device.index or 0]):
        clean = load_particle_adapter(base, args.output / "final.safetensors", device=device)
    context = data["test"]["context"][:4].to(device)
    z, t, ctx, mask, uctx, umask, strength = served.encoder.unpack(context)
    if not torch.equal(raw_velocity(served, context), clean.velocity(z, t, ctx, mask, uctx, umask, strength=strength)):
        raise RuntimeError("clean export reload differs from public serving")
    del clean
    metrics = evaluate_final(served, data, fixed, final_critic, device)
    if final_digest != state_digest(checkpoint(loop)):
        raise RuntimeError("export/evaluation changed full native state")
    if any(sha(ROOT / name) != value for name, value in source_hashes.items()):
        raise RuntimeError("application source changed during continuation")
    if any(sha(imported.parent / name) != value for name, value in pg_hashes.items()):
        raise RuntimeError("native source changed during continuation")
    if any(sha(path) != input_hashes[name] for name, path in input_paths.items()):
        raise RuntimeError("starting input changed during continuation")
    write_json(args.output / "evaluation.json", metrics)
    receipt = dict(plan=plan, initial_native_state_exact=True, two_update_replay_exact=True,
                   final_native_digest=final_digest, final_checkpoint_sha256=sha(args.output / "final.pt"),
                   frozen_unchanged=True, export_reload_exact=True, evaluation_state_unchanged=True,
                   data_and_paired_streams_matched=True, source_and_inputs_unchanged=True,
                   final_data_rng_digest=_rng_digest(loop.data_rng), final_paired_rng_digest=_rng_digest(loop.paired_noise_rng),
                   final_dv12_rng_digest=_rng_digest(loop.policy.noise_generator), served_source=served.source,
                   accepted_moves=[row["move"] for row in rows if isinstance(row["move"], dict) and row["move"].get("moves", 0)],
                   controller=loop.policy.controller.diagnostics(), export=export, seconds=time.perf_counter() - started,
                   metrics={name: {key: value for key, value in pool.items() if key != "records"} for name, pool in metrics.items()})
    write_json(args.output / "receipt.json", receipt)
    write_json(args.output / "status.json", dict(phase="complete", step=args.steps, steps=args.steps))
    emit(event="complete", **receipt)


if __name__ == "__main__":
    main()

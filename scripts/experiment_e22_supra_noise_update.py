#!/usr/bin/env python3
"""Fixed 256-update native-game continuation from the V2 Supra step 1600.

Run each ParticleGAN source in a separate process. Both arms restore the exact
same complete checkpoint, retain particles and use identical application data
and paired-noise streams. Output metrics are final evaluation only. A newer
source may include speed changes as well as the noise-floor repair; this is a
source-version comparison rather than an isolated noise-fix intervention.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
START, UPDATES = 1600, 256
ARCHIVED_PIN = "f459cb6d6aaaabeb1af076ec53ad7a963618de90"


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
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "outputs/e22-particle-v2-1600")
    parser.add_argument("--particlegan-root", type=Path, required=True)
    parser.add_argument("--particlegan-commit", required=True)
    parser.add_argument("--expected-source-digest")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if len(args.particlegan_commit) != 40 or any(character not in "0123456789abcdef" for character in args.particlegan_commit):
        parser.error("--particlegan-commit must be the full lowercase 40-character source pin")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("choose a fresh continuation output directory")
    args.particlegan_root = args.particlegan_root.resolve()
    sys.path.insert(0, str(args.particlegan_root))
    sys.path.insert(1, str(ROOT))

    import torch
    import particlegan
    from supra.runtime import MODEL_REV, T5_REV, VAE_REV, TARGETS, model_module
    from supra.particle_adapter import LINEAR_MODULATED_V2
    from supra.particle_export import export_served_adapter, load_particle_adapter
    from supra.particle_game import ConditionalTokenCritic, _rng_digest
    from supra.particle_pilot import checkpoint, restore, frozen_digest, state_digest
    from supra.particle_training import make_training_loop, training_update, raw_velocity
    from diagnose_e22_supra_training_controls import evaluate_final

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("native Supra/BF16 continuation requires CUDA")
    torch.set_num_threads(8)
    torch.cuda.set_device(device)
    started = time.perf_counter()
    imported = Path(particlegan.__file__).resolve()
    if imported.parent.parent != args.particlegan_root:
        raise RuntimeError("ParticleGAN import differs from the declared continuation source")
    if (args.particlegan_root / ".git").exists():
        git_pin = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.particlegan_root, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--", "particlegan"],
                                        cwd=args.particlegan_root, text=True).strip()
        if git_pin != args.particlegan_commit or dirty:
            raise RuntimeError("ParticleGAN checkout does not match the declared clean source pin")
    pg_paths = sorted(imported.parent.rglob("*.py"))
    pg_hashes = {str(path.relative_to(imported.parent)): sha(path) for path in pg_paths}
    top_hashes = {path.name: pg_hashes[path.name] for path in imported.parent.glob("*.py")}
    pg_digest = state_digest(top_hashes)
    if args.expected_source_digest is not None and pg_digest != args.expected_source_digest:
        raise RuntimeError("ParticleGAN source differs from --expected-source-digest")

    reference_run = json.loads((args.run / "run.json").read_text())
    qualified = json.loads((args.run / "receipt.json").read_text())
    if reference_run["fixed_updates"] != START or reference_run["config"].get("architecture") != LINEAR_MODULATED_V2:
        raise RuntimeError("continuation must start from the declared V2 step-1600 experiment")
    if (reference_run["model_revision"], reference_run["text_revision"], reference_run["vae_revision"]) != (MODEL_REV, T5_REV, VAE_REV):
        raise RuntimeError("reference native model/text/VAE pins differ from this runtime")
    for field in ("full_initial_native_state_identical", "legacy_step_two_replay_exact", "native_resume_exact",
                  "frozen_unchanged", "export_reload_exact", "evaluation_state_unchanged", "data_and_paired_streams_matched"):
        if qualified.get(field) is not True:
            raise RuntimeError(f"V2 reference lacks required qualification: {field}")
    if args.particlegan_commit == ARCHIVED_PIN and pg_digest != reference_run["particlegan_source_digest"]:
        raise RuntimeError("archived control differs from the qualified original ParticleGAN source")
    input_paths = dict(checkpoint=args.run / "final.pt", data=args.run / "data.pt",
                       run=args.run / "run.json", qualification=args.run / "receipt.json")
    input_hashes = {label: sha(path) for label, path in input_paths.items()}
    saved = torch.load(input_paths["checkpoint"], map_location="cpu", weights_only=False, mmap=True)
    data = torch.load(input_paths["data"], map_location="cpu", weights_only=False)
    start_digest = state_digest(saved)
    if start_digest != qualified["final_state_digest"]:
        raise RuntimeError("starting checkpoint differs from the qualified final V2 state")
    if (saved["policy"]["completed_steps"] != START
            or json.dumps(saved["config"], sort_keys=True) != json.dumps(reference_run["config"], sort_keys=True)):
        raise RuntimeError("starting native horizon/config differs from its run receipt")
    if state_digest(data) != saved["config"]["dataset_digest"]:
        raise RuntimeError("starting dataset differs from its native checkpoint")
    if str(device) != saved["policy"]["device"]:
        raise RuntimeError("the exact native continuation requires the original logical device")

    args.output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_paths["data"], args.output / "data.pt")
    source_paths = [Path(__file__), ROOT / "scripts/diagnose_e22_supra_training_controls.py"]
    source_paths += [ROOT / "supra" / name for name in (
        "particle_adapter.py", "particle_export.py", "particle_training.py", "particle_game.py",
        "particle_pilot.py", "particle_training_data.py", "runtime.py")]
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
    plan = dict(label=args.label, start_step=START, additional_updates=UPDATES, final_step=START + UPDATES,
                reference_run=str(args.run.resolve()), input_sha256=input_hashes, initial_native_digest=start_digest,
                particlegan_root=str(args.particlegan_root), particlegan_commit=args.particlegan_commit,
                particlegan_source_digest=pg_digest, particlegan_source_sha256=pg_hashes,
                application_source_sha256=source_hashes, architecture=LINEAR_MODULATED_V2,
                retained="shared 128x4 bank; all 71 particle sites; routers; DV12; native game/structural controls",
                change="ParticleGAN source version only; may include speed changes and noise-floor repair",
                training="unchanged training_update and strict saved configuration; preservation every fifth update",
                selection="final fixed 256-update continuation; no output-metric stopping or selection",
                output_metrics="final evaluation only; no output loss/structural criterion/guard",
                evaluation="complete fit/test/holds/preservation pools, clean native BF16/CFG3, same private GPU72 panels",
                critic_labels={"fixed_start_D": "common V2 step1600 critic", "arm_final_D": "this source's step1856 critic"},
                limitation="one task/stream; source-version comparison does not isolate noise-floor repair")
    write_json(args.output / "plan.json", plan)
    emit(event="plan", **plan)

    # Pure-base loading avoids text/VAE paging. Module construction is isolated
    # from global streams, and native restore replaces every owned RNG/state.
    with torch.random.fork_rng(devices=[device.index or 0]):
        module = model_module()
        with torch.device("meta"):
            base = module.SupraDiT()
            module.attach_supra_lora(base, rank=16, alpha=16, targets=TARGETS)
        teacher = {key.removeprefix("teacher."): value for key, value in saved["policy"]["models"]["encoder"].items()
                   if key.startswith("teacher.")}
        base.load_state_dict(teacher, strict=True, assign=True)
        base.to(device).eval().requires_grad_(False)
        loop = make_training_loop(base, data, device=device, architecture=LINEAR_MODULATED_V2,
                                  probe_interval=saved["config"]["probe_interval"], branch_lr=saved["config"]["branch_lr"])
    restore(loop, saved)
    if state_digest(checkpoint(loop)) != start_digest:
        raise RuntimeError("full native starting state did not restore exactly under the selected source")
    original_frozen = frozen_digest(loop)

    def controls():
        policy = loop.policy
        roles = []
        if policy.lr_settle is not None:
            for index, row in enumerate(policy.lr_settle.testers):
                roles.append([dict(role=role, s=tester.s, mode=getattr(tester, "mode", None))
                              for tester, role in zip(row, policy.roles[index])])
        return dict(step=policy.completed_steps, output_sigma=policy.output_sigma(),
                    learned_sigma=None if policy.log_output_sigma is None else float(policy.log_output_sigma.detach().exp()),
                    roles=roles, mobility=policy.controller.mobility, controller=policy.controller.diagnostics(),
                    generator_lrs=[group["lr"] for group in policy.opt_g.param_groups],
                    critic_lrs=[group["lr"] for group in policy.opt_d.param_groups], bank_rows=len(policy.table))

    initial_controls = controls()
    # Precompute the expected application sampling program with private cloned
    # streams. It neither updates training RNGs nor uses an evaluation score.
    expected_data = torch.Generator().set_state(saved["data_rng"].cpu())
    expected_paired = torch.Generator(device=device).set_state(saved["paired_noise_rng"].cpu())
    sampling = []
    for step in range(START + 1, START + UPDATES + 1):
        hold = step % 5 == 0
        indices = torch.randint(len(data["holds" if hold else "fit"]["context"]), (4,), generator=expected_data)
        noise = [torch.randn((4, 256, 16), device=device, dtype=torch.float32, generator=expected_paired) for _ in range(2)]
        sampling.append(dict(step=step, hold=hold, batch_indices=indices.tolist(),
                             base_noise_sums=[float(value.sum()) for value in noise],
                             paired_rng_digest=_rng_digest(expected_paired)))
    write_json(args.output / "sampling-program.json", sampling)

    # Two native updates are replayed from the same saved state as a mandatory
    # exact-continuation check. Only their replay contributes to the 256 budget.
    first = [training_update(loop), training_update(loop)]
    first_digest = state_digest(checkpoint(loop))
    restore(loop, saved)
    if state_digest(checkpoint(loop)) != start_digest:
        raise RuntimeError("native state failed a second exact restoration")
    native_seconds = 0.
    rows = []
    with (args.output / "train.jsonl").open("w") as trace:
        for offset in range(UPDATES):
            update_started = time.perf_counter()
            row = training_update(loop)
            native_seconds += time.perf_counter() - update_started
            if any(row[key] != value for key, value in sampling[offset].items()):
                raise RuntimeError(f"application data/paired sampling changed at step {row['step']}")
            rows.append(row)
            trace.write(json.dumps(row) + "\n")
            if offset == 1:
                if rows != first or state_digest(checkpoint(loop)) != first_digest:
                    raise RuntimeError("selected source failed exact native two-update replay")
                write_json(args.output / "restore-review.json", dict(initial_native_state_exact=True,
                           initial_native_digest=start_digest, two_update_replay_exact=True))
                emit(event="resume_passed", start_step=START, replay_step=START + 2, full_native_state_exact=True)
            if (offset + 1) % 25 == 0 or offset + 1 == UPDATES:
                trace.flush()
                write_json(args.output / "status.json", dict(phase="training", label=args.label,
                           step=row["step"], final_step=START + UPDATES, native_training_seconds=native_seconds))
                emit(event="training", label=args.label, native_training_seconds=native_seconds, **row)
    if not torch.equal(loop.data_rng.get_state(), expected_data.get_state()) or not torch.equal(loop.paired_noise_rng.get_state(), expected_paired.get_state()):
        raise RuntimeError("final application RNGs differ from the predeclared matched sampling program")
    if original_frozen != frozen_digest(loop):
        raise RuntimeError("continuation modified frozen host or teacher weights")
    final_controls = controls()
    final_state = checkpoint(loop)
    final_digest = state_digest(final_state)
    temporary = args.output / "final.pt.tmp"
    torch.save(final_state, temporary)
    temporary.replace(args.output / "final.pt")
    run = dict(reference_run, config=loop.config, fixed_updates=START + UPDATES,
               particlegan_root=str(args.particlegan_root), particlegan_commit=args.particlegan_commit,
               particlegan_source_digest=pg_digest, additional_updates=UPDATES, noise_update_experiment=plan)
    write_json(args.output / "run.json", run)

    with torch.random.fork_rng(devices=[device.index or 0]):
        fixed = ConditionalTokenCritic(saved["policy"]["models"]["critic"]["scale"]).to(device)
        fixed.load_state_dict(saved["policy"]["models"]["critic"], strict=True)
        fixed.eval().requires_grad_(False)
    final_critic = deepcopy(loop.policy.D).eval().requires_grad_(False)
    served = loop.policy.served_model()
    export = export_served_adapter(served, args.output / "final.safetensors", extra_metadata={
        "selection": "fixed 256-update continuation; output metrics evaluation only", "source_label": args.label})
    with torch.random.fork_rng(devices=[device.index or 0]):
        clean = load_particle_adapter(base, args.output / "final.safetensors", device=device)
    context = data["test"]["context"][:4].to(device)
    z, t, ctx, mask, uctx, umask, strength = served.encoder.unpack(context)
    if not torch.equal(raw_velocity(served, context), clean.velocity(z, t, ctx, mask, uctx, umask, strength=strength)):
        raise RuntimeError("clean adapter reload differs from the native served output")
    del clean
    critic_digest = state_digest([fixed.state_dict(), final_critic.state_dict()])
    data_digest = state_digest(data)
    evaluation_started = time.perf_counter()
    metrics = evaluate_final(served, data, fixed, final_critic, device)
    if final_digest != state_digest(checkpoint(loop)):
        raise RuntimeError("export/evaluation mutated complete native state or RNGs")
    if critic_digest != state_digest([fixed.state_dict(), final_critic.state_dict()]) or data_digest != state_digest(data):
        raise RuntimeError("evaluation mutated detached critics or data")
    if any(sha(path) != input_hashes[label] for label, path in input_paths.items()):
        raise RuntimeError("a continuation input file changed during execution")
    if any(sha(ROOT / name) != value for name, value in source_hashes.items()):
        raise RuntimeError("application source changed during continuation")
    if any(sha(imported.parent / name) != value for name, value in pg_hashes.items()):
        raise RuntimeError("ParticleGAN source changed during continuation")
    if state_digest(saved) != start_digest:
        raise RuntimeError("the mapped initial native checkpoint was mutated")
    write_json(args.output / "evaluation.json", metrics)
    receipt = dict(plan=plan, initial_native_state_exact=True, initial_native_digest=start_digest,
                   two_update_replay_exact=True, final_native_digest=final_digest, final_checkpoint_sha256=sha(args.output / "final.pt"),
                   data_and_paired_streams_matched=True, sampling_program_digest=state_digest(sampling),
                   final_data_rng_digest=_rng_digest(loop.data_rng), final_paired_rng_digest=_rng_digest(loop.paired_noise_rng),
                   final_dv12_rng_digest=_rng_digest(loop.policy.noise_generator), frozen_unchanged=True,
                   export_reload_exact=True, evaluation_state_unchanged=True, source_and_inputs_unchanged=True,
                   initial_controls=initial_controls, final_controls=final_controls,
                   accepted_moves=[row["move"] for row in rows if isinstance(row["move"], dict) and row["move"].get("moves", 0)],
                   served_source=served.source, export=export, native_training_seconds=native_seconds,
                   evaluation_seconds=time.perf_counter() - evaluation_started, seconds=time.perf_counter() - started,
                   metrics={name: {key: value for key, value in pool.items() if key != "records"} for name, pool in metrics.items()})
    write_json(args.output / "receipt.json", receipt)
    write_json(args.output / "status.json", dict(phase="complete", label=args.label, step=START + UPDATES, updates=UPDATES))
    emit(event="complete", **receipt)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""One fixed 1856-to-2112 native-game continuation with fixed-RMS particle keys.

Only routing keys change. Bank values, masses, DV12, optimizer state, game,
and feature guards stay native. Its checkpoint requires dedicated restore;
no ordinary V2 clean adapter is exported.
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
START, UPDATES = 1856, 256
PREFLIGHT_UPDATES = 4
PIN = "cabe2084284db923d525918cbf3e18de6f20faac"


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
    parser.add_argument("--run", type=Path, default=ROOT / "outputs/e22-particle-noise-update-256/latest")
    parser.add_argument("--particlegan-root", type=Path, default=ROOT / "outputs/e22-convergence-gap/particlegan-cabe2084-source")
    parser.add_argument("--particlegan-commit", default=PIN)
    parser.add_argument("--expected-source-digest")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.label = "fixed_rms_particle_keys_v1" if args.label is None else args.label
    args.mode = "native"
    if args.particlegan_commit != PIN:
        parser.error("the key trial requires the predeclared cabe2084 source")
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
    from supra.particle_game import ConditionalTokenCritic, _rng_digest
    from supra.particle_pilot import checkpoint, restore, frozen_digest, state_digest
    from supra.particle_training import make_training_loop
    from diagnose_e22_supra_training_controls import evaluate_final, cpu_copy
    from experiment_e22_supra_preservation_units import game_units_update, rolling_game_summary
    from monitor_e22_supra_particle_convergence import GameProgressMonitor
    from experimental_e22_fixed_rms_keys import (ROUTING, FixedRMSKeyProjection, install_experimental_routing,
                                                experimental_checkpoint, restore_experimental)

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
        raise RuntimeError("key trial must start from the declared V2 step1856 continuation")
    if (reference_run["model_revision"], reference_run["text_revision"], reference_run["vae_revision"]) != (MODEL_REV, T5_REV, VAE_REV):
        raise RuntimeError("reference native model/text/VAE pins differ from this runtime")
    for field in ("initial_native_state_exact", "two_update_replay_exact", "frozen_unchanged", "export_reload_exact",
                  "evaluation_state_unchanged", "data_and_paired_streams_matched", "source_and_inputs_unchanged"):
        if qualified.get(field) is not True:
            raise RuntimeError(f"V2 reference lacks required qualification: {field}")
    if reference_run["particlegan_commit"] != PIN or pg_digest != reference_run["particlegan_source_digest"]:
        raise RuntimeError("both arms must retain the exact latest source used by the starting checkpoint")
    input_paths = dict(checkpoint=args.run / "final.pt", data=args.run / "data.pt",
                       run=args.run / "run.json", qualification=args.run / "receipt.json",
                       native_control_trace=ROOT / "outputs/e22-particle-preservation-units-256/native/train.jsonl")
    input_hashes = {label: sha(path) for label, path in input_paths.items()}
    saved = torch.load(input_paths["checkpoint"], map_location="cpu", weights_only=False, mmap=True)
    data = torch.load(input_paths["data"], map_location="cpu", weights_only=False)
    start_digest = state_digest(saved)
    if start_digest != qualified["final_native_digest"]:
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
    source_paths = [Path(__file__), ROOT / "scripts/diagnose_e22_supra_training_controls.py",
                    ROOT / "scripts/experiment_e22_supra_preservation_units.py",
                    ROOT / "scripts/monitor_e22_supra_particle_convergence.py",
                    ROOT / "scripts/experimental_e22_fixed_rms_keys.py"]
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
                reference_run=str(args.run.resolve()), mode="fixed_rms_particle_keys_v1", input_sha256=input_hashes, initial_native_digest=start_digest,
                particlegan_root=str(args.particlegan_root), particlegan_commit=args.particlegan_commit,
                particlegan_source_digest=pg_digest, particlegan_source_sha256=pg_hashes,
                application_source_sha256=source_hashes, architecture=LINEAR_MODULATED_V2,
                retained="shared 128x4 bank; all 71 particle sites; routers; DV12; native game/structural controls",
                change="fixed-RMS keys only; physical particle values/masses and all native game units retained",
                experimental_routing=ROUTING, ordinary_clean_export_supported=False,
                native_control="outputs/e22-particle-preservation-units-256/native",
                task_gradient_weight=dict(fit=1., holds=.1),
                critic_payoff_weight=dict(fit=1., holds=.1),
                controller_payoff_weight=dict(fit=1., holds=.1),
                extra_backward_telemetry="D game/KA2 norms and cosine; per-role G gradient norms; does not alter optimizer or RNG",
                training="native lifecycle and strict saved configuration; preservation every fifth update",
                selection="final fixed 256-update continuation; no output-metric stopping or selection",
                output_metrics="final evaluation only; no output loss/structural criterion/guard",
                evaluation="complete fit/test/holds/preservation pools, clean native BF16/CFG3, same private GPU72 panels",
                critic_labels={"fixed_start_D": "common V2 step1856 critic", "arm_final_D": "this mode's step2112 critic"},
                switch_diagnostic="native and fixed-RMS-key zero-update FAST game on the common monitor's 12 edit/12 preservation contexts, same private Gaussian/DV12 streams and frozen1856 critic; same probes repeated at2112",
                limitation="one task/stream; routing keys alone change, with native critic/controller units retained; warm reinterpretation changes the initial host output, so this tests continuation efficacy rather than identical-output fresh convergence; no independent image-quality judge")
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

    # Verify the unmodified native continuation before changing the forward.
    native_first = [game_units_update(loop, mode="native") for _ in range(PREFLIGHT_UPDATES)]
    native_first_digest = state_digest(checkpoint(loop))
    restore(loop, saved)
    native_second = [game_units_update(loop, mode="native") for _ in range(PREFLIGHT_UPDATES)]
    if native_first != native_second or state_digest(checkpoint(loop)) != native_first_digest:
        raise RuntimeError("unmodified native four-update preflight did not replay exactly")
    restore(loop, saved)
    if state_digest(checkpoint(loop)) != start_digest:
        raise RuntimeError("native preflight did not restore the exact starting state")
    native_control_trace = ROOT / "outputs/e22-particle-preservation-units-256/native/train.jsonl"
    control_first = [json.loads(row) for row in native_control_trace.read_text().splitlines()[:PREFLIGHT_UPDATES]]
    if native_first != control_first:
        raise RuntimeError("native preflight differs from the qualified comparison arm")
    with torch.random.fork_rng(devices=[device.index or 0]):
        fixed = ConditionalTokenCritic(saved["policy"]["models"]["critic"]["scale"]).to(device)
        fixed.load_state_dict(saved["policy"]["models"]["critic"], strict=True)
        fixed.eval().requires_grad_(False)
    fixed_critic_digest = state_digest(fixed.state_dict())
    monitor = GameProgressMonitor(loop, data, fixed, args.output / "game-monitor")
    if any(len(context) != 12 for context, _ in monitor.pools.values()):
        raise RuntimeError("routing-switch monitor does not use the declared 12/12 common probes")
    native_zero = monitor.evaluate(loop, [])
    write_json(args.output / "routing-switch-native.json", native_zero)
    install_experimental_routing(loop)
    experiment_initial = cpu_copy(experimental_checkpoint(loop))
    experiment_initial_digest = state_digest(experiment_initial)
    initial_native = dict(experiment_initial["native"])
    initial_native["config"] = dict(initial_native["config"])
    del initial_native["config"]["experimental_routing"]
    if state_digest(initial_native) != start_digest:
        raise RuntimeError("installing experimental routing changed native tensors or RNG state")
    del initial_native
    experimental_zero = monitor.evaluate(loop, [])
    write_json(args.output / "routing-switch-experimental.json", experimental_zero)
    switch_delta = {pool: {mode: experimental_zero["probes"][pool][mode]["frozen_start_D"]
                         - native_zero["probes"][pool][mode]["frozen_start_D"]
                         for mode in ("clean", "dv12")} for pool in monitor.pools}
    emit(event="routing_switch_game", step=START, fixed_game_delta=switch_delta,
         native_state_and_streams_unchanged=True, output_metrics_used=False)
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

    # Include the first preservation intervention in the exact replay witness.
    # Only its replay contributes to the fixed 256-update budget.
    first = [game_units_update(loop, mode=args.mode) for _ in range(PREFLIGHT_UPDATES)]
    first_digest = state_digest(experimental_checkpoint(loop))
    restore_experimental(loop, experiment_initial)
    if state_digest(experimental_checkpoint(loop)) != experiment_initial_digest:
        raise RuntimeError("native state failed a second exact restoration")
    native_seconds = 0.
    rows = []
    with (args.output / "train.jsonl").open("w") as trace:
        for offset in range(UPDATES):
            update_started = time.perf_counter()
            row = game_units_update(loop, mode=args.mode)
            native_seconds += time.perf_counter() - update_started
            if any(row[key] != value for key, value in sampling[offset].items()):
                raise RuntimeError(f"application data/paired sampling changed at step {row['step']}")
            rows.append(row)
            trace.write(json.dumps(row) + "\n")
            if offset == PREFLIGHT_UPDATES - 1:
                if rows != first or state_digest(experimental_checkpoint(loop)) != first_digest:
                    raise RuntimeError("selected mode failed exact native replay across preservation")
                write_json(args.output / "restore-review.json", dict(initial_native_state_exact=True,
                           initial_native_digest=start_digest, experimental_initial_digest=experiment_initial_digest,
                           native_control_four_update_replay_exact=True, native_control_trace_exact=True,
                           four_update_replay_exact=True,
                           preflight_updates=PREFLIGHT_UPDATES, includes_preservation=True))
                emit(event="resume_passed", start_step=START, replay_step=START + PREFLIGHT_UPDATES,
                     full_native_state_exact=True, includes_preservation=True)
            if (offset + 1) % 25 == 0 or offset + 1 == UPDATES:
                trace.flush()
                rolling = rolling_game_summary(rows)
                with (args.output / "progress.jsonl").open("a") as progress:
                    progress.write(json.dumps(rolling) + "\n")
                write_json(args.output / "status.json", dict(phase="training", label=args.label,
                           step=row["step"], final_step=START + UPDATES, native_training_seconds=native_seconds,
                           rolling_game=rolling))
                emit(event="training", label=args.label, native_training_seconds=native_seconds, rolling_game=rolling, **row)
    if not torch.equal(loop.data_rng.get_state(), expected_data.get_state()) or not torch.equal(loop.paired_noise_rng.get_state(), expected_paired.get_state()):
        raise RuntimeError("final application RNGs differ from the predeclared matched sampling program")
    if original_frozen != frozen_digest(loop):
        raise RuntimeError("continuation modified frozen host or teacher weights")
    final_controls = controls()
    final_state = cpu_copy(experimental_checkpoint(loop))
    final_digest = state_digest(final_state)
    temporary = args.output / "final.pt.tmp"
    torch.save(final_state, temporary)
    temporary.replace(args.output / "final.pt")
    run = dict(reference_run, config=loop.config, fixed_updates=START + UPDATES,
               particlegan_root=str(args.particlegan_root), particlegan_commit=args.particlegan_commit,
               particlegan_source_digest=pg_digest, additional_updates=UPDATES, experimental_routing=ROUTING, fixed_rms_key_experiment=plan,
               ordinary_clean_export_supported=False,
               comparison="fixed 256-update V2 native game versus fixed-RMS particle keys")
    write_json(args.output / "run.json", run)

    final_probe = monitor.evaluate(loop, rows)
    write_json(args.output / "final-monitor.json", final_probe)
    if state_digest(fixed.state_dict()) != fixed_critic_digest:
        raise RuntimeError("routing-switch/final monitor changed the common frozen1856 critic")
    final_critic = deepcopy(loop.policy.D).eval().requires_grad_(False)
    served = loop.policy.served_model()
    if (served.generator.experimental_routing != ROUTING
            or any(type(branch) is not FixedRMSKeyProjection for branch in served.generator.particle_branches())):
        raise RuntimeError("native served-model deepcopy lost the experimental routing law")
    # Dedicated restore is mandatory; never emit an ordinary V2 clean adapter.
    probe_context = data["test"]["context"][:4].to(device)
    probe_before = served.routed_forward(probe_context)
    del served
    restore_experimental(loop, final_state)
    if state_digest(experimental_checkpoint(loop)) != final_digest:
        raise RuntimeError("tagged final checkpoint failed dedicated exact restoration")
    served = loop.policy.served_model()
    if not torch.equal(probe_before, served.routed_forward(probe_context)):
        raise RuntimeError("dedicated experimental restore changed clean native output")
    critic_digest = state_digest([fixed.state_dict(), final_critic.state_dict()])
    data_digest = state_digest(data)
    evaluation_started = time.perf_counter()
    metrics = evaluate_final(served, data, fixed, final_critic, device)
    if final_digest != state_digest(experimental_checkpoint(loop)):
        raise RuntimeError("dedicated restore/evaluation mutated complete native state or RNGs")
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
                   native_control_four_update_replay_exact=True, native_control_trace_exact=True,
                   experimental_initial_digest=experiment_initial_digest,
                   four_update_replay_exact=True, replay_includes_preservation=True,
                   preflight_updates=PREFLIGHT_UPDATES, final_experimental_digest=final_digest,
                   final_native_digest=state_digest(final_state["native"]), final_checkpoint_sha256=sha(args.output / "final.pt"),
                   data_and_paired_streams_matched=True, sampling_program_digest=state_digest(sampling),
                   final_data_rng_digest=_rng_digest(loop.data_rng), final_paired_rng_digest=_rng_digest(loop.paired_noise_rng),
                   final_dv12_rng_digest=_rng_digest(loop.policy.noise_generator), frozen_unchanged=True,
                   dedicated_experimental_restore_exact=True, ordinary_clean_export_supported=False,
                   evaluation_state_unchanged=True, source_and_inputs_unchanged=True,
                   initial_controls=initial_controls, final_controls=final_controls,
                   accepted_moves=[row["move"] for row in rows if isinstance(row["move"], dict) and row["move"].get("moves", 0)],
                   served_source=served.source, experimental_routing=ROUTING, native_training_seconds=native_seconds,
                   routing_switch_fixed_game_delta=switch_delta,
                   zero_update_native_game=native_zero, zero_update_experimental_game=experimental_zero,
                   final_fixed_game_probe=final_probe,
                   evaluation_seconds=time.perf_counter() - evaluation_started, seconds=time.perf_counter() - started,
                   full_budget_game_summary=rolling_game_summary(rows, window=UPDATES),
                   metrics={name: {key: value for key, value in pool.items() if key != "records"} for name, pool in metrics.items()})
    write_json(args.output / "receipt.json", receipt)
    write_json(args.output / "status.json", dict(phase="complete", label=args.label, step=START + UPDATES, updates=UPDATES))
    emit(event="complete", **receipt)


if __name__ == "__main__":
    main()

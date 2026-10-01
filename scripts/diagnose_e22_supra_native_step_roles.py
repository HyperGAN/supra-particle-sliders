#!/usr/bin/env python3
"""Replay-qualified late native steps, with reversible parameter-role ablations.

Every intervention reuses one actual optimizer displacement, never a new
objective or selected learning rate. Particle codes and routing remain active.
The unchanged native continuation is restored after every diagnostic panel.
"""
import argparse
from copy import deepcopy
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ROLES = ("generator", "router", "table")
ARMS = [tuple(role for role, enabled in zip(ROLES, mask) if enabled)
        for mask in itertools.product((False, True), repeat=3)]


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def emit(**row):
    print(json.dumps(row), flush=True)


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "outputs/e22-particle-v2-6400")
    parser.add_argument("--particlegan-root", type=Path,
                        default=ROOT / "outputs/e22-convergence-gap/particlegan-cabe2084-source")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--updates", type=int, default=5)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("choose a fresh output directory")
    if args.updates < 5:
        parser.error("at least five consecutive updates must include preservation")
    sys.path.insert(0, str(args.particlegan_root.resolve()))
    sys.path.insert(1, str(ROOT))
    import torch
    import torch.nn.functional as F
    import particlegan
    from supra.runtime import model_module, TARGETS
    from supra.particle_game import patchify
    from supra.particle_pilot import checkpoint, restore, state_digest, frozen_digest
    from supra.particle_training import make_training_loop, training_update
    from monitor_e22_supra_particle_convergence import evaluation_modes

    if Path(particlegan.__file__).resolve().parent.parent != args.particlegan_root.resolve():
        raise RuntimeError("unexpected ParticleGAN import")
    torch.set_num_threads(4)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    began = time.perf_counter()
    saved = torch.load(args.run / "final.pt", map_location="cpu", weights_only=False, mmap=True)
    data = torch.load(args.run / "data.pt", map_location="cpu", weights_only=False)
    qualification = json.loads((args.run / "qualification-review.json").read_text())
    if not qualification.get("qualified"):
        raise RuntimeError("starting long run is not qualified")
    if saved["policy"]["completed_steps"] != 6400 or state_digest(data) != saved["config"]["dataset_digest"]:
        raise RuntimeError("unexpected starting horizon or dataset")
    args.output.mkdir(parents=True)
    initial = state_digest(saved)
    source_files = [Path(__file__), ROOT / "scripts/monitor_e22_supra_particle_convergence.py"]
    source_files += sorted((ROOT / "supra").glob("particle*.py"))
    source_files += [ROOT / "supra/runtime.py"]
    sources = {str(path.relative_to(ROOT)): sha(path) for path in source_files}
    for path in source_files:
        target = args.output / "source" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    pg_sources = {str(path.relative_to(args.particlegan_root)): sha(path)
                  for path in sorted((args.particlegan_root / "particlegan").rglob("*.py"))}
    inputs = {name: sha(args.run / name) for name in ("final.pt", "data.pt", "qualification-review.json")}
    plan = dict(start_step=6400, updates=args.updates, arms=[list(x) for x in ARMS],
                initial_native_digest=initial, input_sha256=inputs, application_source_sha256=sources,
                particlegan_source_sha256=pg_sources, imported_particlegan=str(Path(particlegan.__file__).resolve()),
                objective="native paired RpGAN only; no output metric",
                intervention="subsets of the actual G/router/table displacement, at the actual post-D critic",
                controls="same native G-pass private DV12/Gaussian draws; clean fixed-game cross-task probes",
                selection="five predetermined steps, all eight subsets; no variant selection or production change",
                limitation="local displacement attribution on one stream, not proof of long-run superiority")
    write(args.output / "plan.json", plan)
    emit(event="plan", start_step=6400, updates=args.updates)
    with torch.random.fork_rng(devices=[device.index or 0]):
        mod = model_module()
        with torch.device("meta"):
            base = mod.SupraDiT()
            mod.attach_supra_lora(base, rank=16, alpha=16, targets=TARGETS)
        base.load_state_dict({k.removeprefix("teacher."): v for k, v in saved["policy"]["models"]["encoder"].items()
                              if k.startswith("teacher.")}, strict=True, assign=True)
        base.to(device).eval().requires_grad_(False)
        loop = make_training_loop(base, data, device=device, architecture=saved["config"]["architecture"],
                                  probe_interval=saved["config"]["probe_interval"], branch_lr=saved["config"]["branch_lr"])
    restore(loop, saved)
    if state_digest(checkpoint(loop)) != initial:
        raise RuntimeError("initial restore differs")
    frozen = frozen_digest(loop)
    reference_rows = [training_update(loop) for _ in range(args.updates)]
    reference_final = state_digest(checkpoint(loop))
    restore(loop, saved)
    policy = loop.policy
    fixed_critic = deepcopy(policy.D).eval().requires_grad_(False)
    # Two spread training contexts per edit subject and four per preservation
    # subject, plus two held-out contexts per edit subject. Private CPU panels.
    private = torch.Generator().manual_seed(72)
    probes = {}
    for label, pool_name, per_subject in (("fit", "fit", 2), ("holds", "holds", 4), ("test", "test", 2)):
        pool = data[pool_name]["context"]
        ids = []
        for subject in pool[:, 4097].unique(sorted=True):
            available = (pool[:, 4097] == subject).nonzero().flatten()
            ids.extend(available[torch.linspace(0, len(available) - 1, per_subject).round().long()].tolist())
        contexts = pool[ids].to(device)
        panels = torch.randn(4, len(ids), 256, 16, generator=private).to(device)
        probes[label] = (contexts, panels)
    evidence = dict(plan=plan, rows=[])
    records = []
    original_generate = policy.routed_generate
    original_step = policy.opt_g.step
    original_after = policy.after_generator_step
    capture = {}

    def in_step_digest():
        # Native checkpoint() deliberately rejects an unfinished update. This
        # fingerprint reads its active owners without faking a ready boundary;
        # the complete native boundary is checked against reference replay.
        modules = policy._training_modules()
        return state_digest(dict(
            models={k: m.state_dict() for k, m in modules.items()},
            averages={k: m.state_dict() for k, m in policy._average_modules().items()},
            gradients={k: {n: p.grad for n, p in m.named_parameters()} for k, m in modules.items()},
            training={k: [m.training for m in owner.modules()] for k, owner in modules.items()},
            requires_grad={k: [p.requires_grad for p in m.parameters()] for k, m in modules.items()},
            table=policy.table, table_grad=policy.table.grad, averaged_table=policy.averaged_table,
            optimizers=[o.state_dict() for o in policy.optimizers],
            controller=policy.controller.state_dict(), routing=policy.routed_control.state_dict(),
            lr_settle=policy.lr_settle.state_dict(),
            penalty=policy.penalty.regularizer.state_dict(), penalty_stats=policy.penalty.last_stats,
            phase=policy._phase, completed_steps=policy.completed_steps,
            data_rng=loop.data_rng.get_state(), paired_rng=loop.paired_noise_rng.get_state(),
            dv12_rng=policy.noise_generator.get_state(), cpu_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state(device)))

    def generated(context, **kwargs):
        capture["calls"] += 1
        if capture["calls"] == 2:
            capture["context"] = context.detach().clone()
            capture["dv12_state"] = policy.noise_generator.get_state().clone()
            capture["controller"] = deepcopy(policy.controller)
            capture["sigma"] = float(policy._noise.output_sigma.detach())
        result = original_generate(context, **kwargs)
        if capture["calls"] == 2:
            capture["prediction"] = result.detach().clone()
        return result

    def stepped(*a, **kw):
        groups = {}
        for group, role in zip(policy.opt_g.param_groups, policy.roles[0]):
            if role in ROLES:
                groups[role] = list(group["params"])
        if set(groups) != set(ROLES):
            raise RuntimeError("unexpected optimizer role layout")
        capture["groups"] = groups
        capture["before"] = {role: [p.detach().clone() for p in params] for role, params in groups.items()}
        capture["gradients"] = {role: [None if p.grad is None else p.grad.detach().clone() for p in params]
                                for role, params in groups.items()}
        return original_step(*a, **kw)

    @torch.no_grad()
    def after():
        original_after()
        native_before = in_step_digest()
        current = {role: [p.detach().clone() for p in params] for role, params in capture["groups"].items()}
        telemetry = {}
        for role, params in capture["groups"].items():
            deltas = [value - before for value, before in zip(current[role], capture["before"][role])]
            gradients = capture["gradients"][role]
            g2 = sum(float(g.double().square().sum()) for g in gradients if g is not None)
            d2 = sum(float(d.double().square().sum()) for d in deltas)
            dot = sum(float((d.double() * g.double()).sum()) for d, g in zip(deltas, gradients) if g is not None)
            telemetry[role] = dict(gradient_norm=g2 ** .5, displacement_norm=d2 ** .5,
                                   gradient_dot_displacement=dot,
                                   descent_cosine=-dot / max((g2 * d2) ** .5, 1e-30))
        result = dict(step=policy.completed_steps + 1, hold=(policy.completed_steps + 1) % 5 == 0,
                      role_geometry=telemetry, arms={})
        raw = dict(step=result["step"], arms={})
        try:
            with evaluation_modes(policy):
                for arm in ARMS:
                    name = "+".join(arm) if arm else "none"
                    for role, params in capture["groups"].items():
                        values = current[role] if role in arm else capture["before"][role]
                        for parameter, value in zip(params, values):
                            parameter.copy_(value)
                    candidate = policy.routed_control.candidate()
                    models = policy._training_modules()
                    spec = policy.routed_control.spec
                    controller = capture["controller"]
                    prior = controller.routed_prior(candidate.table, candidate.log_mass)
                    stream = torch.Generator(device=device).set_state(capture["dv12_state"].cpu())
                    residual = spec.forward(models, capture["context"], candidate, perturb_fn=lambda codes:
                        controller.perturb_latent(codes, stream, prior, record=False))
                    if not arm and not torch.equal(residual, capture["prediction"]):
                        raise RuntimeError("native pre-step G forward does not replay exactly")
                    condition = policy.encoder.condition(capture["context"])
                    real = capture["sigma"] * capture["generator_base"]
                    fake = real + patchify(residual).float() / policy.D.scale
                    gap = policy.D(real, condition) - policy.D(fake, condition)
                    scores = dict(training_native_dv12=dict(g_game=float(F.softplus(gap).mean())))
                    raw_arm = dict(training_native_dv12=dict(gap=gap.cpu()))
                    for label, (contexts, noise) in probes.items():
                        panel_gaps = []
                        for start in range(0, len(contexts), 4):
                            context = contexts[start:start + 4]
                            prediction = spec.forward(models, context, candidate)
                            real_panel = capture["sigma"] * noise[:, start:start + len(context)]
                            real_flat = real_panel.flatten(0, 1)
                            fake_flat = (real_panel + (patchify(prediction) / fixed_critic.scale).unsqueeze(0)).flatten(0, 1)
                            condition = policy.encoder.condition(context).repeat(4, 1)
                            panel_gaps.append((fixed_critic(real_flat, condition) - fixed_critic(fake_flat, condition)).reshape(4, len(context)).cpu())
                        values = torch.cat(panel_gaps, dim=1)
                        scores[label] = dict(g_game=float(F.softplus(values).double().mean()))
                        raw_arm[label] = dict(gap=values)
                    result["arms"][name] = scores
                    raw["arms"][name] = raw_arm
        finally:
            for role, params in capture["groups"].items():
                for parameter, value in zip(params, current[role]):
                    parameter.copy_(value)
        if in_step_digest() != native_before:
            raise RuntimeError("role panels altered native state")
        for name, pools in result["arms"].items():
            for label, score in pools.items():
                score["delta_from_no_step"] = score["g_game"] - result["arms"]["none"][label]["g_game"]
        result["native_state_unchanged_by_panels"] = True
        records.append(result)
        evidence["rows"].append(raw)
        write(args.output / "roles.json", dict(plan=plan, rows=records, completed=False))
        torch.save(evidence, args.output / "score-panels.pt")
        emit(event="role_step", step=result["step"], hold=result["hold"],
             training_deltas={k: v["training_native_dv12"]["delta_from_no_step"] for k, v in result["arms"].items()},
             test_deltas={k: v["test"]["delta_from_no_step"] for k, v in result["arms"].items()})

    policy.routed_generate = generated
    policy.opt_g.step = stepped
    policy.after_generator_step = after
    rows = []
    try:
        for index in range(args.updates):
            capture.clear()
            capture["calls"] = 0
            paired = torch.Generator(device=device).set_state(loop.paired_noise_rng.get_state().cpu())
            torch.randn((4, 256, 16), device=device, generator=paired)
            capture["generator_base"] = torch.randn((4, 256, 16), device=device, generator=paired)
            row = training_update(loop)
            if row != reference_rows[index]:
                raise RuntimeError("instrumented native training row differs from exact replay")
            rows.append(row)
    finally:
        policy.routed_generate = original_generate
        policy.opt_g.step = original_step
        policy.after_generator_step = original_after
    final = state_digest(checkpoint(loop))
    if final != reference_final or frozen_digest(loop) != frozen:
        raise RuntimeError("instrumented native full state differs from reference replay")
    summaries = {}
    for task, hold in (("edit", False), ("preservation", True)):
        chosen = [r for r in records if r["hold"] == hold]
        summaries[task] = {name: {pool: sum(r["arms"][name][pool]["delta_from_no_step"] for r in chosen) / len(chosen)
                                  for pool in ("training_native_dv12", "fit", "holds", "test")}
                           for name in records[0]["arms"]}
    complete = dict(plan=plan, rows=records, summary_mean_game_delta=summaries, completed=True,
                    exact_native_rows=True, exact_native_full_replay=True, frozen_unchanged=True,
                    final_native_digest=final, reference_final_native_digest=reference_final,
                    source_and_inputs_unchanged=(all(sha(args.run / name) == value for name, value in inputs.items())
                        and all(sha(ROOT / name) == value for name, value in sources.items())),
                    seconds=time.perf_counter() - began)
    if not complete["source_and_inputs_unchanged"]:
        raise RuntimeError("diagnostic source or inputs changed")
    write(args.output / "native-training-rows.json", rows)
    write(args.output / "roles.json", complete)
    emit(event="complete", seconds=complete["seconds"], summary=summaries)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Fresh fixed-400 native-game particle gating versus the qualified V2 control.

The sole training intervention is the explicitly versioned V3 generator law.
Particles, native DV12, critics, optimizer roles and zero-feature-harm guards
remain native. Output metrics are endpoint diagnostics, never selection rules.
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
UPDATES = 400


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def diagnostic(value):
    import torch
    if isinstance(value, torch.Tensor):
        return diagnostic(value.detach().cpu().tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: diagnostic(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [diagnostic(item) for item in value]
    return value


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(diagnostic(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def emit(**row):
    print(json.dumps(diagnostic(row), allow_nan=False), flush=True)


def audit_gated(loop, context, *, all_sites=True):
    """Read-only actual V3 features, clean/DV12 game gradients and site effects."""
    import torch
    import torch.nn.functional as F
    from particlegan import RoutedCandidate
    from particlegan.policy import output_noise_std
    from particlegan.routing import RoutedExecution
    from supra.particle_adapter import GATED_PARTICLE_V3
    from supra.particle_game import patchify
    from supra.particle_pilot import checkpoint, state_digest
    from monitor_e22_supra_particle_convergence import evaluation_modes

    policy = loop.policy
    if policy._phase != "ready" or policy.G.architecture != GATED_PARTICLE_V3:
        raise ValueError("V3 audit requires a completed native update")
    before = state_digest(checkpoint(loop))
    context = context.to(policy.device)
    spec = policy.routed_control.spec
    models = policy._training_modules()
    candidate = policy.routed_control.candidate()
    parameters = {"generator." + name: value for name, value in policy.G.named_parameters()
                  if value.requires_grad}
    parameters.update({"router." + name: value for name, value in policy.router.named_parameters()
                       if value.requires_grad})
    parameters["bank.table"] = policy.table
    gradients_before = {name: None if value.grad is None else value.grad.detach().clone()
                        for name, value in parameters.items()}
    pair = torch.Generator(device=policy.device)
    pair.set_state(loop.paired_noise_rng.get_state())
    gaussian = torch.randn(len(context), 256, 16, generator=pair, device=policy.device)
    noise = torch.Generator(device=policy.device)
    noise.set_state(policy.noise_generator.get_state())
    prior = policy.controller.routed_prior(candidate.table, candidate.log_mass)
    condition = policy.encoder.condition(context)
    sigma = policy._output_sigma(output_noise_std(policy.recipe, policy.completed_steps), detach=True)
    real = sigma * gaussian
    loss = policy.recipe.make_loss()
    captured_codes, captured_hidden, branch_records = {}, {}, {}

    class Capture(RoutedExecution):
        def mix(self, site, logits):
            codes = super().mix(site, logits)
            if capturing[0]:
                captured_codes[site] = (torch.cat((codes[:, 0], codes[:, 1]), 0)
                                        if policy.G.cfg > 1 else codes).detach()
            calls[0].append(site)
            return codes

    calls, capturing = [[]], [False]

    def forward(current=candidate, perturb=None):
        calls[0] = []
        execution = Capture(spec.sites, current, len(context), perturb)
        try:
            residual = spec.model_forward(models, context, current, execution)
            usage = execution.finish()
        finally:
            execution.close()
        if calls[0] != list(policy.G.sites):
            raise RuntimeError("V3 audit did not execute every native site in order")
        return residual, usage

    def game(residual):
        with torch.no_grad():
            real_logits = policy.D(real, condition)
        return loss.g_loss(policy.D(real + patchify(residual).float() / policy.D.scale, condition), real_logits)

    def game_gradient(residual):
        payoff = game(residual)
        gradients = torch.autograd.grad(payoff, tuple(parameters.values()), allow_unused=True)
        norms = {name: 0. if gradient is None else float(gradient.detach().double().norm())
                 for name, gradient in zip(parameters, gradients)}
        table_gradient = gradients[list(parameters).index("bank.table")]
        return dict(g_game=float(payoff.detach()), parameter_gradient_norms=norms,
                    bank_rows_nonzero=0 if table_gradient is None else int((table_gradient.norm(dim=1) > 0).sum()),
                    router_tensors_nonzero=sum(value > 0 for name, value in norms.items() if name.startswith("router.")),
                    generator_tensors_nonzero=sum(value > 0 for name, value in norms.items() if name.startswith("generator.")))

    def rms(value):
        return float(value.detach().float().square().mean().sqrt())

    handles = []
    for branch in policy.G.particle_branches():
        def down_hook(_, __, hidden, site=branch.site):
            captured_hidden[site] = hidden.detach()

        def up_hook(_, inputs, branch=branch):
            with torch.no_grad(), torch.autocast(policy.device.type, enabled=False):
                hidden, codes = captured_hidden[branch.site], captured_codes[branch.site]
                rank = hidden.shape[-1]
                hidden_term = F.linear(hidden, branch.bridge.weight[:, :rank], branch.bridge.bias).tanh()
                gate_preactivation = F.linear(codes.float(), branch.bridge.weight[:, rank:])
                gate = gate_preactivation.tanh()
                expected = hidden + hidden_term + hidden * gate
                if not torch.equal(inputs[0], expected):
                    raise RuntimeError("captured V3 up-input does not match its declared formula")
                effect = F.linear(hidden * gate, branch.up.weight)
                complete = F.linear(expected, branch.up.weight)
                centered = expected.reshape(-1, rank).double()
                centered = centered - centered.mean(0)
                covariance = centered.T @ centered / len(centered)
                branch_records[branch.site] = dict(hidden_rms=rms(hidden), code_rms=rms(codes),
                    particle_gate_rms=rms(gate), gate_saturation_fraction=float((gate_preactivation.abs() > 3).float().mean()),
                    code_effect_at_fixed_input_rms=rms(effect), total_adapter_delta_rms=rms(complete),
                    code_effect_over_total_delta=rms(effect) / max(rms(complete), 1e-30),
                    up_input_effective_rank=float(covariance.trace().square() / covariance.square().sum().clamp_min(1e-30)),
                    formula_exact=True)
        handles.append(branch.down.register_forward_hook(down_hook))
        handles.append(branch.up.register_forward_pre_hook(up_hook))
    try:
        with evaluation_modes(policy), torch.autograd.set_multithreading_enabled(False):
            capturing[0] = True
            clean, usage = forward()
            capturing[0] = False
            for handle in handles:
                handle.remove()
            handles.clear()
            clean_values = clean.detach()
            clean_gradients = game_gradient(clean)
            noisy, _ = forward(perturb=lambda codes: policy.controller.perturb_latent(codes, noise, prior, record=False))
            noisy_gradients = game_gradient(noisy)
            interventions = {}

            def measure(name, current=candidate, perturb=None):
                with torch.no_grad():
                    value, _ = forward(current, perturb)
                    payoff = float(game(value))
                interventions[name] = dict(g_game=payoff,
                    game_change=payoff - clean_gradients["g_game"],
                    output_change_rms_evaluation_only=rms(value - clean_values))
            measure("zero_particle_codes", perturb=lambda codes: torch.zeros_like(codes))
            mass = candidate.log_mass.detach().clone()
            deleted = int(usage.detach().mean(0).argmax())
            mass[deleted] = -torch.inf
            measure("delete_most_used_row", RoutedCandidate(candidate.table, mass,
                    {**candidate.row_state, "log_mass": mass}, averaged=candidate.averaged))
            interventions["delete_most_used_row"]["row"] = deleted
            site_effects = {}
            indexes = range(len(policy.G.sites)) if all_sites else (0, len(policy.G.sites) // 2, len(policy.G.sites) - 1)
            for index in indexes:
                count = [0]
                def one_site(codes, index=index):
                    site_index = count[0]
                    count[0] += 1
                    return codes + .1 if site_index == index else codes
                measure("site", perturb=one_site)
                if count[0] != len(policy.G.sites):
                    raise RuntimeError("V3 site intervention missed a native routing call")
                site_effects[policy.G.sites[index]] = interventions.pop("site")
    finally:
        for handle in handles:
            handle.remove()
    if state_digest(checkpoint(loop)) != before:
        raise RuntimeError("V3 diagnostic changed native state or training RNG")
    if any((original is None) != (parameters[name].grad is None) or
           (original is not None and not torch.equal(original, parameters[name].grad))
           for name, original in gradients_before.items()):
        raise RuntimeError("V3 diagnostic changed preexisting gradients")
    return dict(step=policy.completed_steps, architecture=policy.G.architecture,
                clean_gradient=clean_gradients, dv12_gradient=noisy_gradients,
                branches=branch_records, interventions=interventions, per_site_interventions=site_effects,
                native_state_rng_and_gradients_unchanged=True,
                limits="Read-only connectivity and dependence; full-pool game ablation measures usefulness separately.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, default=ROOT / "outputs/e22-pr223-shared-1600")
    parser.add_argument("--initial", type=Path, default=ROOT / "outputs/e22-pr223-shared-profile-smoke")
    parser.add_argument("--teacher", type=Path, default=ROOT / "outputs/e22-particle-v2-6400")
    parser.add_argument("--judge", type=Path, default=ROOT / "outputs/e22-particle-noise-update-256/latest")
    parser.add_argument("--particlegan-root", type=Path, default=ROOT.parent / "ParticleGAN-pr223-supra")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/e22-particle-gated-v3-400")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("choose a fresh output directory")
    pg_root = args.particlegan_root.resolve()
    if subprocess.check_output(["git", "-C", str(pg_root), "rev-parse", "HEAD"], text=True).strip() != PIN:
        raise RuntimeError("unexpected ParticleGAN pin")
    if subprocess.check_output(["git", "-C", str(pg_root), "status", "--porcelain", "--", "particlegan"], text=True).strip():
        raise RuntimeError("ParticleGAN package source is dirty")
    sys.path.insert(0, str(pg_root))
    sys.path.insert(1, str(ROOT))
    import torch
    import particlegan
    from supra.runtime import TARGETS, model_module
    from supra.particle_adapter import GATED_PARTICLE_V3, LINEAR_MODULATED_V2
    from supra.particle_export import export_served_adapter, load_particle_adapter
    from supra.particle_game import ConditionalTokenCritic
    from supra.particle_pilot import checkpoint, frozen_digest, restore, state_digest
    from supra.particle_training import SHARED_ROUTED_PROFILE, make_training_loop, raw_velocity, training_update
    from supra.particle_training_data import SOURCE_COLUMN, TARGET_COLUMN, STRENGTH_COLUMN
    from diagnose_e22_supra_training_controls import evaluate_final
    from evaluate_e22_supra_particle_contribution import evaluate_pool
    from monitor_e22_supra_particle_convergence import GameProgressMonitor, rolling_losses

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("real native BF16 Supra requires CUDA")
    torch.cuda.set_device(device)
    torch.set_num_threads(8)
    if Path(particlegan.__file__).resolve().parent.parent != pg_root:
        raise RuntimeError("ParticleGAN import differs from declared checkout")
    started = time.perf_counter()
    args.output.mkdir(parents=True)
    checks = {}

    def require(name, condition):
        checks[name] = bool(condition)
        if not checks[name]:
            write(args.output / "failed-checks.json", checks)
            raise RuntimeError(name)

    input_paths = dict(control_checkpoint=args.control / "checkpoint-00400.pt",
        control_trace=args.control / "train.jsonl", control_receipt=args.control / "receipt.json",
        control_review=args.control / "independent-review.json", initial=args.initial / "checkpoint-00002.pt",
        initial_result=args.initial / "result.json", initial_review=args.initial / "independent-review.json",
        teacher=args.teacher / "final.pt", teacher_review=args.teacher / "qualification-review.json",
        data=args.teacher / "data.pt", judge=args.judge / "final.pt")
    inputs_sha = {name: sha(path) for name, path in input_paths.items()}
    app_files = sorted((ROOT / "supra").glob("particle*.py")) + [ROOT / "supra/runtime.py", Path(__file__)]
    app_files += [ROOT / "scripts" / name for name in ("diagnose_e22_supra_training_controls.py",
        "evaluate_e22_supra_particle_contribution.py", "monitor_e22_supra_particle_convergence.py")]
    app_sha = {str(path.relative_to(ROOT)): sha(path) for path in app_files}
    pg_files = sorted((pg_root / "particlegan").rglob("*.py"))
    pg_sha = {str(path.relative_to(pg_root)): sha(path) for path in pg_files}
    for path in app_files + pg_files:
        destination = args.output / "source" / (path.relative_to(ROOT) if path in app_files
                                               else Path("native") / path.relative_to(pg_root))
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    plan = dict(schema="supra_particle_gated_v3_matched_400_v1", fixed_updates=UPDATES,
        architectures=[LINEAR_MODULATED_V2, GATED_PARTICLE_V3], profile=SHARED_ROUTED_PROFILE,
        particlegan_commit=PIN, device=str(device), gpu=torch.cuda.get_device_name(device),
        torch_version=torch.__version__, input_paths={name: str(path.resolve()) for name, path in input_paths.items()},
        input_sha256=inputs_sha, application_source_sha256=app_sha, particlegan_source_sha256=pg_sha,
        intervention="up(h+tanh(Hh+Cz+b)) -> up(h+tanh(Hh+b)+h*tanh(Cz)); same bridge tensors",
        retained="71sites rank16 shared128x4 particles; native keys/masses/DV12/RpGAN/KA2/optimizers/roles/row controllers",
        guard="native learned features, FAST+averaged and per-context zero feature harm; output guard disabled",
        initialization="fresh public deterministic_orthogonal_, identical native step0 tensors/owners; zero up outputs",
        comparator="qualified shared V2 checkpoint400, admitted only after exact fresh V2 step2 replay",
        judges="common frozen V2 D1856 and same qualified shared V2 D400 for both endpoints",
        progress="fixed 24 training contexts, private paired panels and common V2 step2 DV12 stream; no selection",
        selection="final fixed400 horizon; no output objective, guard, seed sweep or parameter/checkpoint selection",
        limitations="one task/stream; architecture causal test; no claim against ordinary LoRA at1600/6400")
    write(args.output / "plan.json", plan)
    write(args.output / "status.json", dict(phase="preparing_control", step=0, steps=UPDATES,
          architecture=GATED_PARTICLE_V3, particle_profile=SHARED_ROUTED_PROFILE))
    emit(event="plan", fixed_updates=UPDATES, architecture=GATED_PARTICLE_V3, pin=PIN)
    control_receipt = json.loads(input_paths["control_receipt"].read_text())
    control_review = json.loads(input_paths["control_review"].read_text())
    initial_result = json.loads(input_paths["initial_result"].read_text())
    initial_review = json.loads(input_paths["initial_review"].read_text())
    teacher_review = json.loads(input_paths["teacher_review"].read_text())
    require("qualified_control", control_receipt["qualified"] and control_review["qualified"])
    require("qualified_common_judge_input", inputs_sha["judge"] == control_receipt["plan"]["input_sha256"]["judge"])
    require("qualified_initial", initial_result["qualified"] and all(initial_result["checks"].values())
            and initial_review.get("qualified", initial_review.get("all_checks_passed", False)))
    initial = torch.load(input_paths["initial"], map_location="cpu", weights_only=False, mmap=True)
    teacher = torch.load(input_paths["teacher"], map_location="cpu", weights_only=False, mmap=True)
    data = torch.load(input_paths["data"], map_location="cpu", weights_only=False)
    require("qualified_teacher_digest", teacher_review["qualified"]
            and state_digest(teacher) == teacher_review["final_native_digest"])
    require("initial_digest", state_digest(initial) == initial_result["checkpoint_native_digest"])
    require("common_cached_dataset", state_digest(data) == initial["config"]["dataset_digest"]
            == teacher["config"]["dataset_digest"])
    old_rows = [json.loads(line) for line in input_paths["control_trace"].read_text().splitlines()][:UPDATES]
    require("complete_control_trace", len(old_rows) == UPDATES and old_rows[-1]["step"] == UPDATES)
    source_teacher = {key.removeprefix("teacher."): value for key, value in
                      teacher["policy"]["models"]["encoder"].items() if key.startswith("teacher.")}
    with torch.random.fork_rng(devices=[]), torch.device("meta"):
        module = model_module()
        base = module.SupraDiT()
        module.attach_supra_lora(base, rank=16, alpha=16, targets=TARGETS)
    base.load_state_dict(source_teacher, strict=True, assign=True)
    base.to(device).eval().requires_grad_(False)
    require("frozen_source_teacher_only", state_digest(base.state_dict()) == state_digest(source_teacher))
    del source_teacher, teacher
    gc.collect()

    def fresh(architecture):
        torch.set_rng_state(initial["policy"]["cpu_rng"].cpu())
        torch.cuda.set_rng_state(initial["policy"]["cuda_rng"].cpu(), device)
        with torch.random.fork_rng(devices=[device.index or 0]):
            return make_training_loop(base, data, device=device, architecture=architecture,
                branch_lr=initial["config"]["branch_lr"], probe_interval=initial["config"]["probe_interval"],
                profile=SHARED_ROUTED_PROFILE)

    control = fresh(LINEAR_MODULATED_V2)
    v2_step_zero_digest = state_digest(checkpoint(control)["policy"])
    v2_config = deepcopy(control.config)
    prefix = [training_update(control), training_update(control)]
    require("fresh_v2_step_two_full_native_exact", state_digest(checkpoint(control)) == state_digest(initial))
    require("fresh_v2_prefix_rows_exact", prefix == old_rows[:2])
    emit(event="control_prefix_exact", step=2, full_native_exact=True, rows_exact=True)
    saved_control = torch.load(input_paths["control_checkpoint"], map_location="cpu", weights_only=False, mmap=True)
    require("qualified_control_400_content_digest", state_digest(saved_control)
            == control_review["checkpoints"]["400"]["new_native_digest"])
    require("qualified_control_trace_hash", inputs_sha["control_trace"] == control_review["artifact_sha256"]["train.jsonl"])
    restore(control, saved_control)
    control_digest = state_digest(checkpoint(control))
    require("control400_strict_restore", control_digest == state_digest(saved_control)
            and control.policy.completed_steps == UPDATES)
    with torch.random.fork_rng(devices=[device.index or 0]):
        fixed = ConditionalTokenCritic(data["coordinate_scale"]).to(device)
    judge = torch.load(input_paths["judge"], map_location="cpu", weights_only=False, mmap=True)
    require("common_judge_1856", judge["policy"]["completed_steps"] == 1856)
    fixed.load_state_dict(judge["policy"]["models"]["critic"], strict=True)
    fixed.eval().requires_grad_(False)
    require("qualified_common_judge_state", state_digest(fixed.state_dict())
            == control_review["scoring_critic_digests"]["common_frozen1856"])
    final_judge = deepcopy(control.policy.D).eval().requires_grad_(False)
    plan["critic_digests"] = dict(D1856=state_digest(fixed.state_dict()), D400=state_digest(final_judge.state_dict()))
    del judge, saved_control
    write(args.output / "plan.json", plan)
    control_metrics = evaluate_final(control.policy.served_model(), data, fixed, final_judge, device)
    require("control_evaluation_immutable", state_digest(checkpoint(control)) == control_digest)
    write(args.output / "evaluation-v2.json", control_metrics)
    reference_monitor = GameProgressMonitor(control, data, fixed, args.output / "control-probes400")
    reference_monitor.dv12_state = initial["policy"]["streams"]["noise_generator"].clone()
    control_probe = reference_monitor.evaluate(control, old_rows)
    write(args.output / "control-progress400.json", control_probe)
    del reference_monitor, control
    gc.collect()
    torch.cuda.empty_cache()

    loop = fresh(GATED_PARTICLE_V3)
    policy = loop.policy
    require("fresh_v3_identical_native_step_zero", state_digest(checkpoint(loop)["policy"]) == v2_step_zero_digest)
    require("only_architecture_config_changed", {key: value for key, value in loop.config.items() if key != "architecture"}
            == {key: value for key, value in v2_config.items() if key != "architecture"})
    require("fresh_v3_no_trained_optimizer_state", all(not optimizer.state for optimizer in policy.optimizers))
    require("native_v3_particle_contract", len(policy.G.sites) == 71 and tuple(policy.table.shape) == (128, 4)
            and policy.G.rank == 16 and policy.ema_G.architecture == GATED_PARTICLE_V3
            and not policy.routed_control.spec.output_error_guard and policy.routed_control.spec.max_context_harm == 0.)
    frozen = frozen_digest(loop)
    source = data["fit"]["context"][:4].to(device).clone()
    source[:, TARGET_COLUMN] = source[:, SOURCE_COLUMN]
    served = policy.served_model()
    expected = served.encoder.teacher_velocity(source)
    require("v3_zero_output_base_exact", torch.equal(raw_velocity(served, source), expected))
    source[:, STRENGTH_COLUMN] = 0
    require("v3_zero_strength_base_exact", torch.equal(raw_velocity(served, source), expected))
    del served, source, expected
    write(args.output / "run.json", dict(plan, config=loop.config, fixed_updates=UPDATES,
          particlegan_source_digest=state_digest(pg_sha)))

    def save(path):
        temporary = path.with_suffix(".tmp")
        torch.save(checkpoint(loop), temporary)
        temporary.replace(path)

    rows = [training_update(loop), training_update(loop)]
    save(args.output / "checkpoint-00002.pt")
    audit_two = audit_gated(loop, data["fit"]["context"][:4])
    write(args.output / "audit-step-two.json", audit_two)

    def qualify_audit(name, audit):
        require(name + "_71_exact_basis_sites", len(audit["branches"]) == 71
                and all(row["formula_exact"] for row in audit["branches"].values()))
        for mode in ("clean_gradient", "dv12_gradient"):
            gradient = audit[mode]
            require(name + "_" + mode + "_owners", gradient["bank_rows_nonzero"] == 128
                    and gradient["router_tensors_nonzero"] == 142 and gradient["generator_tensors_nonzero"] == 284
                    and all(math.isfinite(value) for value in gradient["parameter_gradient_norms"].values()))
        require(name + "_all_site_interventions_finite", len(audit["per_site_interventions"]) == 71
                and all(math.isfinite(row["output_change_rms_evaluation_only"])
                        for row in audit["per_site_interventions"].values()))
        require(name + "_all_fp32_branch_code_effects", all(row["code_effect_at_fixed_input_rms"] > 0
                for row in audit["branches"].values()))
        audit["final_output_site_effects_nonzero_diagnostic"] = sum(
            row["output_change_rms_evaluation_only"] > 0 for row in audit["per_site_interventions"].values())
        if name == "final":
            require("final_whole_model_particle_dependence", audit["interventions"]["zero_particle_codes"]["output_change_rms_evaluation_only"] > 0)
    qualify_audit("step2", audit_two)
    write(args.output / "audit-step-two.json", audit_two)
    reference_rows = [training_update(loop) for _ in range(3)]
    step_five = state_digest(checkpoint(loop))
    restore(loop, torch.load(args.output / "checkpoint-00002.pt", map_location="cpu", weights_only=False, mmap=True))
    replay_rows = [training_update(loop) for _ in range(3)]
    require("v3_replay_three_rows_exact", replay_rows == reference_rows)
    require("v3_cpu_checkpoint_replay_full_native_exact", state_digest(checkpoint(loop)) == step_five)
    rows += reference_rows
    monitor = GameProgressMonitor(loop, data, fixed, args.output)
    monitor.dv12_state = initial["policy"]["streams"]["noise_generator"].clone()
    del initial
    gc.collect()
    emit(event="preflight_complete", step=5, exact_resume=True, dense_rows=128)

    def activity():
        return diagnostic(dict(accepted_moves=policy.routed_control.counters["moves"],
            optimizer_surprise_fires=policy.surprise.fires, anchor_release_events=policy.surprise.anchor_events,
            epoch_rebases=policy.reopen_guard.epoch_rebases, controller=policy.controller.diagnostics(),
            reopen_guard=policy.reopen_guard.state_dict(), lr_settle=policy.lr_settle.diagnostics()))

    def stream_check(row):
        reference = old_rows[row["step"] - 1]
        for name in ("hold", "game_weight", "batch_indices", "base_noise_sums", "paired_rng_digest"):
            require(f"stream_{row['step']}_{name}", row[name] == reference[name])
    for row in rows:
        stream_check(row)
    training_started = time.perf_counter()
    with (args.output / "train.jsonl").open("w") as trace:
        for row in rows:
            trace.write(json.dumps(row) + "\n")
        for step in range(6, UPDATES + 1):
            row = training_update(loop)
            require("clock_" + str(step), row["step"] == step)
            stream_check(row)
            rows.append(row)
            trace.write(json.dumps(row) + "\n")
            if step % 25 == 0:
                trace.flush()
                status = dict(phase="training", step=step, steps=UPDATES,
                    seconds=time.perf_counter() - training_started, architecture=GATED_PARTICLE_V3,
                    rolling=rolling_losses(rows), guard_activity=activity())
                write(args.output / "status.json", status)
                emit(event="training", **status)
            if step % 100 == 0:
                emit(event="game_progress", **monitor.evaluate(loop, rows))
    training_seconds = time.perf_counter() - training_started
    save(args.output / "final.pt")
    require("final_400_updates", policy.completed_steps == UPDATES)
    require("frozen_host_teacher_unchanged", frozen_digest(loop) == frozen)
    require("dense_bank_after_acquisition", all(row["dense_gradient_rows"] == 128 for row in rows[1:]))
    dv12_equal = sum(row["dv12_rng_digest"] == old_rows[index]["dv12_rng_digest"] for index, row in enumerate(rows))
    if policy.routed_control.counters["moves"] == 0:
        require("all_400_dv12_streams_exact", dv12_equal == UPDATES)
    native_digest = state_digest(checkpoint(loop))
    write(args.output / "status.json", dict(phase="evaluating", step=UPDATES, steps=UPDATES, guard_activity=activity()))
    audit_final = audit_gated(loop, data["fit"]["context"][:4])
    write(args.output / "audit-final.json", audit_final)
    qualify_audit("final", audit_final)
    write(args.output / "audit-final.json", audit_final)
    served = policy.served_model()
    gated_metrics = evaluate_final(served, data, fixed, final_judge, device)
    write(args.output / "evaluation-v3.json", gated_metrics)
    contributions = {pool: evaluate_pool(served, data[pool], {"common_D1856": fixed, "common_D400": final_judge},
                    pool_name=pool) for pool in ("fit", "test", "holds", "preservation")}
    write(args.output / "particle-contribution.json", contributions)
    export = export_served_adapter(served, args.output / "final.safetensors",
        extra_metadata=dict(selection="fixed400 horizon", output_metrics="evaluation only"))
    with torch.random.fork_rng(devices=[device.index or 0]):
        clean = load_particle_adapter(base, args.output / "final.safetensors", device=device)
    context = data["test"]["context"][:4].to(device)
    z, t, ctx, mask, uctx, umask, strength = served.encoder.unpack(context)
    require("v3_clean_export_reload_exact", torch.equal(raw_velocity(served, context),
            clean.velocity(z, t, ctx, mask, uctx, umask, strength=strength)))
    require("v3_export_tag_explicit", export["config"]["architecture"] == GATED_PARTICLE_V3
            and clean.generator.architecture == GATED_PARTICLE_V3)
    require("final_diagnostics_export_native_state_immutable", state_digest(checkpoint(loop)) == native_digest)
    comparison = {pool: dict(count=control_metrics[pool]["count"],
        control_rmse=control_metrics[pool]["rmse"], gated_rmse=gated_metrics[pool]["rmse"],
        rmse_change_evaluation_only=gated_metrics[pool]["rmse"] - control_metrics[pool]["rmse"],
        **{judge: dict(control_g_game=control_metrics[pool][judge]["g_game"],
            gated_g_game=gated_metrics[pool][judge]["g_game"],
            new_minus_control=gated_metrics[pool][judge]["g_game"] - control_metrics[pool][judge]["g_game"])
            for judge in ("fixed_start_D", "arm_final_D")}) for pool in control_metrics}
    require("application_sources_unchanged", all(sha(ROOT / name) == value for name, value in app_sha.items()))
    require("native_sources_unchanged", all(sha(pg_root / name) == value for name, value in pg_sha.items()))
    require("immutable_inputs_unchanged", all(sha(input_paths[name]) == value for name, value in inputs_sha.items()))
    receipt = dict(qualified=True, plan=plan, checks=checks, comparison=comparison,
        v2_step_zero_native_digest=v2_step_zero_digest, control_native_digest=control_digest,
        final_native_digest=native_digest, final_checkpoint_sha256=sha(args.output / "final.pt"),
        final_export_sha256=sha(args.output / "final.safetensors"), dv12_streams_equal=dv12_equal,
        training_seconds=training_seconds, total_seconds=time.perf_counter() - started,
        controller=activity(), export=export, source_immutable=True,
        critics={"fixed_start_D": "same frozen V2 D1856", "arm_final_D": "same qualified shared V2 D400"})
    write(args.output / "receipt.json", receipt)
    write(args.output / "status.json", dict(phase="complete", step=UPDATES, steps=UPDATES,
        architecture=GATED_PARTICLE_V3, comparison=comparison, guard_activity=activity()))
    emit(event="complete", comparison=comparison, final_native_digest=native_digest,
         training_seconds=training_seconds, total_seconds=receipt["total_seconds"])


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Versioned editing-only continuation after the qualified native V3 step1600.

Changes only application batch selection/task weight after strict restoration;
native particle/game/optimizer/structural law stays unchanged. Both fixed
editing budgets use common critics. Output never selects a horizon/checkpoint.
"""
import argparse
from copy import deepcopy
import gc
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_ROOT = ROOT.parent / "supra-concept-sliders/outputs/final-boss-supra-converged"
ORIGINAL_SHA = {1600: "0c97f2f2fd05bf17be99c901a87a05447f0448ad6edae06ecba4e5d623fed997",
                6400: "24a98ba055d0b5298136d67dc7491a339684da03416c510d1d443fe730eb7069",
                "published1600": "134f5a9d12206b74f39a2c3f9c90fdb38477a1c3204b5deea5152b1cbe49a47f"}
sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiment_e22_supra_particle_gated import PIN, audit_gated, diagnostic, emit, sha, write
from e22_supra_historical_control import historical_inputs, reconstruct
from e22_supra_editing_only import EDITING_ONLY, editing_config, make_from_saved, opt_in, training_update
from review_e22_supra_pr223 import committed_python_hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial", type=Path, default=ROOT / "outputs/e22-particle-gated-v3-1600")
    parser.add_argument("--steps", type=int, choices=(6400,), default=6400)
    parser.add_argument("--control", type=Path, default=ROOT / "outputs/e22-particle-v2-6400")
    parser.add_argument("--control-prefix", type=Path, default=ROOT / "outputs/e22-particle-v2-1600")
    parser.add_argument("--control-middle", type=Path, default=ROOT / "outputs/e22-particle-noise-update-256/latest")
    parser.add_argument("--data", type=Path, default=ROOT / "outputs/e22-particle-v2-6400/data.pt")
    parser.add_argument("--judge", type=Path, default=ROOT / "outputs/e22-particle-noise-update-256/latest/final.pt")
    parser.add_argument("--common-initial", type=Path, default=ROOT / "outputs/e22-pr223-shared-profile-smoke")
    parser.add_argument("--original-1600", type=Path, default=ORIGINAL_ROOT / "checkpoint-01600/final-boss-supra.safetensors")
    parser.add_argument("--original-6400", type=Path, default=ORIGINAL_ROOT / "checkpoint-06400/final-boss-supra.safetensors")
    parser.add_argument("--original-published-1600", type=Path,
                        default=ROOT.parent / "supra-concept-sliders/outputs/final-boss-supra-1600/final-boss-supra.safetensors")
    parser.add_argument("--particlegan-root", type=Path, default=ROOT.parent / "ParticleGAN-pr223-supra")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/e22-particle-gated-v3-editing-only-6400")
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
    from safetensors import safe_open
    from safetensors.torch import load_file
    from supra.runtime import TARGETS, MODEL_ID, MODEL_REV, T5_REV, VAE_REV, adapter_state, load_adapter_state, model_module
    from supra.particle_adapter import GATED_PARTICLE_V3, LINEAR_MODULATED_V2
    from supra.particle_export import export_served_adapter, load_particle_adapter
    from supra.particle_game import ConditionalTokenCritic
    from supra.particle_pilot import checkpoint, frozen_digest, restore, state_digest
    from supra.particle_training import LEGACY_ROUTED_PROFILE, SHARED_ROUTED_PROFILE, raw_velocity
    from diagnose_e22_supra_training_controls import evaluate_final
    from evaluate_e22_supra_particle_contribution import evaluate_pool
    from monitor_e22_supra_particle_convergence import GameProgressMonitor, rolling_losses

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("native BF16 Supra continuation requires CUDA")
    torch.cuda.set_device(device)
    torch.set_num_threads(8)
    if Path(particlegan.__file__).resolve().parent.parent != pg_root:
        raise RuntimeError("unexpected ParticleGAN import")
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    checks = {}

    def require(name, condition):
        checks[name] = bool(condition)
        if not checks[name]:
            write(args.output / "failed-checks.json", checks)
            raise RuntimeError(name)

    control_review_path = args.control / ("independent-review.json" if (args.control / "independent-review.json").is_file()
                                         else "qualification-review.json")
    input_paths = dict(initial=args.initial / "final.pt", initial_receipt=args.initial / "receipt.json",
        initial_review=args.initial / "independent-review.json", initial_trace=args.initial / "train.jsonl",
        control=args.control / "final.pt", control_review=control_review_path, control_trace=args.control / "train.jsonl",
        data=args.data, judge=args.judge, common_initial=args.common_initial / "checkpoint-00002.pt",
        common_initial_result=args.common_initial / "result.json", common_initial_review=args.common_initial / "independent-review.json",
        data_provenance=args.data.parent / "run.json", original1600=args.original_1600, original6400=args.original_6400,
        originalpublished1600=args.original_published_1600, control_provenance=args.control / "run.json")
    input_paths.update(historical_inputs(args.control_prefix, args.control_middle, args.control))
    inputs_sha = {name: sha(path) for name, path in input_paths.items()}
    app_files = sorted((ROOT / "supra").glob("particle*.py")) + [ROOT / "supra/runtime.py", Path(__file__)]
    app_files += [ROOT / "scripts" / name for name in ("experiment_e22_supra_particle_gated.py",
        "diagnose_e22_supra_training_controls.py", "evaluate_e22_supra_particle_contribution.py",
        "monitor_e22_supra_particle_convergence.py", "e22_supra_historical_control.py", "e22_supra_editing_only.py",
        "review_e22_supra_pr223.py")]
    app_sha = {str(path.relative_to(ROOT)): sha(path) for path in app_files}
    pg_files = sorted((pg_root / "particlegan").rglob("*.py"))
    pg_sha = {str(path.relative_to(pg_root)): sha(path) for path in pg_files}
    for path in app_files + pg_files:
        destination = args.output / "source" / (path.relative_to(ROOT) if path in app_files
                                               else Path("native") / path.relative_to(pg_root))
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    initial_receipt = json.loads(input_paths["initial_receipt"].read_text())
    initial_review = json.loads(input_paths["initial_review"].read_text())
    control_review = json.loads(input_paths["control_review"].read_text())
    common_result = json.loads(input_paths["common_initial_result"].read_text())
    common_review = json.loads(input_paths["common_initial_review"].read_text())
    require("qualified_initial_chain", initial_receipt["qualified"] and initial_review["qualified"]
            and all(initial_receipt["checks"].values()))
    require("qualified_control", control_review.get("qualified", control_review.get("all_checks_passed", False)))
    require("qualified_common_step2", common_result["qualified"] and common_review["qualified"])
    initial = torch.load(input_paths["initial"], map_location="cpu", weights_only=False, mmap=True)
    control_state = torch.load(input_paths["control"], map_location="cpu", weights_only=False, mmap=True)
    common_initial = torch.load(input_paths["common_initial"], map_location="cpu", weights_only=False, mmap=True)
    data = torch.load(input_paths["data"], map_location="cpu", weights_only=False)
    data_provenance = json.loads(input_paths["data_provenance"].read_text())
    control_provenance = json.loads(input_paths["control_provenance"].read_text())
    original_refs = dict(converged_1600=dict(step=1600, input="original1600", sha=ORIGINAL_SHA[1600]),
        published_1600=dict(step=1600, input="originalpublished1600", sha=ORIGINAL_SHA["published1600"]),
        converged_6400=dict(step=6400, input="original6400", sha=ORIGINAL_SHA[6400]))
    original_metadata = {}
    for label, reference in original_refs.items():
        step = reference["step"]
        require(f"declared_original_{label}_sha", inputs_sha[reference["input"]] == reference["sha"])
        with safe_open(str(input_paths[reference["input"]]), framework="pt", device="cpu") as handle:
            metadata = {key: json.loads(value) for key, value in (handle.metadata() or {}).items()}
        expected = dict(format="supra-native-lora-v1", rank=16, alpha=16, targets=list(TARGETS),
            model_id=MODEL_ID, model_revision=MODEL_REV, text_encoder_revision=T5_REV,
            vae_revision=VAE_REV, step=step, prompts_sha256=data_provenance["prompts_sha256"])
        require(f"original_{label}_metadata_pins_task", all(metadata.get(key) == value for key, value in expected.items()))
        original_metadata[label] = metadata
    start_step = initial["policy"]["completed_steps"]
    require("qualified_initial_content", state_digest(initial) == initial_receipt["final_native_digest"]
            == initial_review["final_native_digest"] and inputs_sha["initial"] == initial_receipt["final_checkpoint_sha256"])
    require("qualified_control_content", state_digest(control_state) == control_review["final_native_digest"])
    require("qualified_common_initial_content", state_digest(common_initial) == common_result["checkpoint_native_digest"]
            and inputs_sha["common_initial"] == common_review["checkpoint_sha256"])
    require("fixed_horizon_chain", start_step in (400, 1600) and start_step < args.steps
            and control_state["policy"]["completed_steps"] == args.steps)
    require("explicit_native_v3_shared_config", initial["config"]["architecture"] == GATED_PARTICLE_V3
            and initial["config"]["particle_profile"] == SHARED_ROUTED_PROFILE
            and not initial["config"]["output_error_guard"] and initial["config"]["max_feature_context_harm"] == 0.)
    require("same_cached_data", state_digest(data) == initial["config"]["dataset_digest"]
            == control_state["config"]["dataset_digest"] == common_initial["config"]["dataset_digest"])
    original_rows = [json.loads(line) for line in input_paths["initial_trace"].read_text().splitlines()]
    control_rows, control_chain = reconstruct(input_paths, data, torch=torch, state_digest=state_digest,
        sha=sha, committed_python_hashes=committed_python_hashes, repository=pg_root,
        require=lambda condition, message: require(message, condition))
    require("historical_control_suffix_aliases", input_paths["control_trace"].resolve() == input_paths["control_suffix_trace"].resolve()
            and input_paths["control"].resolve() == input_paths["control_suffix_checkpoint"].resolve()
            and input_paths["control_review"].resolve() == input_paths["control_suffix_review"].resolve())
    require("historical_control_native_endpoint", state_digest(control_state) == control_chain["native_boundary_digests"]["suffix"])
    require("long_initial_boundary1600", start_step == 1600)
    require("initial_pre_schedule_configuration", "training_schedule" not in initial["config"])
    require("complete_input_traces", len(original_rows) == start_step and len(control_rows) == args.steps)
    expected_data = torch.Generator().set_state(initial["data_rng"].cpu())
    editing_sampling = [dict(step=step, hold=False, game_weight=1.,
        batch_indices=torch.randint(len(data["fit"]["context"]), (4,), generator=expected_data).tolist())
        for step in range(1601, args.steps + 1)]
    write(args.output / "editing-sampling-program.json", editing_sampling)
    with (args.output / "control-trace-full.jsonl").open("w") as complete_trace:
        for row in control_rows:
            complete_trace.write(json.dumps(row) + "\n")
    control_chain["reconstructed_trace_sha256"] = sha(args.output / "control-trace-full.jsonl")
    write(args.output / "control-chain.json", control_chain)
    plan = dict(schema="supra_particle_gated_v3_editing_only_long_v1", fixed_updates=args.steps, start_step=start_step,
        architecture=GATED_PARTICLE_V3, profile=SHARED_ROUTED_PROFILE, particlegan_commit=PIN,
        particlegan_root=str(pg_root), device=str(device), gpu=torch.cuda.get_device_name(device), torch_version=torch.__version__,
        control_profile=control_state["config"].get("particle_profile", LEGACY_ROUTED_PROFILE),
        control_particlegan_commit=control_provenance["particlegan_commit"],
        control_trace_parts=control_chain["control_trace_parts"],
        control_program_digest=control_chain["complete_recorded_program_digest"],
        control_reconstructed_trace_sha256=control_chain["reconstructed_trace_sha256"],
        training_schedule=EDITING_ONLY, schedule_start_step=1600, inherited_edit_updates=1280,
        inherited_preservation_updates=320, endpoint_steps=[5440, 6400], endpoint_edit_updates={"5440": 5120, "6400": 6080},
        historical_control_edit_updates=5120, historical_control_preservation_updates=1280,
        editing_sampling_program_digest=state_digest(editing_sampling),
        editing_sampling_program_sha256=sha(args.output / "editing-sampling-program.json"),
        input_paths={name: str(path.resolve()) for name, path in input_paths.items()}, input_sha256=inputs_sha,
        application_source_sha256=app_sha, particlegan_source_sha256=pg_sha,
        intervention="editing_only_v1 after strict update1600 restore; future fit-only batches/task weight1; unchanged native optimizer/game/structural law",
        restored="all native models/EMA/particles/optimizer/controller/streams and original config exactly before explicit schedule opt-in",
        guard="native learned critic features, FAST+averaged and per-context zero feature harm; output guard disabled",
        judges=f"same frozen V2D1856 and same qualified V2D{args.steps} for both endpoints",
        progress="same24fixed training contexts/private paired panels/common qualified V2step2 DV12 stream",
        ordinary_references="converged ordinary LoRA trajectory at1600, previous published1600 (same adapter tensors as converged1600), fixedconverged6400 quality target; same570contexts/twojudges/panels",
        ordinary_reference_metadata=original_metadata,
        selection="predeclared endpoints5440 and6400; report both, never choose best; no output objective/guard/stopping/parameter or checkpoint selection; no seed sweep",
        limitations="one task/stream; application schedule changes after1600; historical V2 uses different legacy source/profile and preservation schedule;5440 matches5120 editing updates,6400 has6080 edits")
    write(args.output / "plan.json", plan)
    write(args.output / "status.json", dict(phase="preparing", step=start_step, steps=args.steps,
          architecture=GATED_PARTICLE_V3, particle_profile=SHARED_ROUTED_PROFILE, training_schedule=EDITING_ONLY,
          editing_updates=1280, preservation_updates=320))
    emit(event="plan", start_step=start_step, fixed_updates=args.steps, architecture=GATED_PARTICLE_V3)
    source_teacher = {key.removeprefix("teacher."): value for key, value in
                      initial["policy"]["models"]["encoder"].items() if key.startswith("teacher.")}
    with torch.random.fork_rng(devices=[]), torch.device("meta"):
        module = model_module()
        base = module.SupraDiT()
        module.attach_supra_lora(base, rank=16, alpha=16, targets=TARGETS)
    base.load_state_dict(source_teacher, strict=True, assign=True)
    base.to(device).eval().requires_grad_(False)
    require("pure_base_teacher_zero_ordinary_outputs", all(not bool(value.count_nonzero())
            for key, value in source_teacher.items() if key.endswith(".up.weight")))
    del source_teacher

    def make(saved, architecture):
        require("factory_architecture_" + architecture, saved["config"]["architecture"] == architecture)
        with torch.random.fork_rng(devices=[device.index or 0]):
            return make_from_saved(base, data, saved, device=device)

    loop = make(initial, GATED_PARTICLE_V3)
    restore(loop, initial)
    require("initial_strict_full_native_restore", state_digest(checkpoint(loop)) == state_digest(initial))
    frozen = frozen_digest(loop)
    initial_digest = state_digest(checkpoint(loop))
    with torch.random.fork_rng(devices=[device.index or 0]):
        fixed = ConditionalTokenCritic(data["coordinate_scale"]).to(device)
        final_judge = ConditionalTokenCritic(data["coordinate_scale"]).to(device)
    judge = torch.load(input_paths["judge"], map_location="cpu", weights_only=False, mmap=True)
    require("fixed_common_judge1856", judge["policy"]["completed_steps"] == 1856)
    fixed.load_state_dict(judge["policy"]["models"]["critic"], strict=True)
    final_judge.load_state_dict(control_state["policy"]["models"]["critic"], strict=True)
    fixed.eval().requires_grad_(False)
    final_judge.eval().requires_grad_(False)
    plan["critic_digests"] = dict(D1856=state_digest(fixed.state_dict()),
                                  control_final=state_digest(final_judge.state_dict()))
    native_digest = initial_digest
    # Compare the control with the same two judge tensors and panel reset,
    # never with either arm's own different evolving critic.
    control_state = torch.load(input_paths["control"], map_location="cpu", weights_only=False, mmap=True)
    # Native restore owns global RNGs as well. Contain the independent control
    # restore/evaluation so the continued V3 boundary keeps its own full state.
    with torch.random.fork_rng(devices=[device.index or 0]):
        control = make(control_state, LINEAR_MODULATED_V2)
        restore(control, control_state)
        control_digest = state_digest(checkpoint(control))
        require("control_strict_full_native_restore", control_digest == state_digest(control_state))
        old_metrics = evaluate_final(control.policy.served_model(), data, fixed, final_judge, device)
        require("control_evaluation_native_state_immutable", state_digest(checkpoint(control)) == control_digest)
    require("v3_full_native_state_immutable_after_control_eval", state_digest(checkpoint(loop)) == native_digest)
    write(args.output / "evaluation-v2.json", old_metrics)
    # The baseline owns a separate pure-base copy. Installing ordinary weights
    # cannot touch the continued particle host or its target-caption teacher.
    with torch.random.fork_rng(devices=[device.index or 0]):
        original_model = deepcopy(base).eval().requires_grad_(False)

    class OriginalReference:
        def __init__(self, model, encoder):
            self.model, self.encoder = model, encoder

        @torch.no_grad()
        def routed_forward(self, context):
            z, t, ctx, mask, uctx, umask, strength = self.encoder.unpack(context)
            batch = len(z)
            branches = [module for module in self.model.modules() if hasattr(module, "multiplier")]
            previous = [module.multiplier for module in branches]
            try:
                for branch in branches:
                    branch.multiplier = strength
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    output = self.model(torch.cat((z, z)), torch.cat((t, t)),
                        torch.cat((ctx, uctx.expand(batch, -1, -1))),
                        torch.cat((mask, umask.expand(batch, -1))))
                conditional, unconditional = output.float().chunk(2)
                velocity = unconditional + 3 * (conditional - unconditional)
                return velocity - self.encoder.teacher_velocity(context)
            finally:
                for branch, multiplier in zip(branches, previous):
                    branch.multiplier = multiplier

    original = OriginalReference(original_model, loop.policy.encoder)
    originals = {}
    for label, reference in original_refs.items():
        weights = load_file(str(input_paths[reference["input"]]), device="cpu")
        expected = adapter_state(original_model)
        require(f"original_{label}_all_142_tensors", set(weights) == set(expected) and len(weights) == 142)
        require(f"original_{label}_tensor_shapes_dtypes_finite", all(value.shape == expected[key].shape
            and value.dtype == expected[key].dtype and bool(torch.isfinite(value).all()) for key, value in weights.items()))
        load_adapter_state(original_model, weights)
        original_digest = state_digest(original_model.state_dict())
        originals[label] = evaluate_final(original, data, fixed, final_judge, device)
        require(f"original_{label}_reference_immutable", state_digest(original_model.state_dict()) == original_digest)
        require(f"original_{label}_v3_teacher_native_immutable", state_digest(checkpoint(loop)) == native_digest)
        write(args.output / f"evaluation-original-{label}.json", originals[label])
    del judge, control, control_state, original, original_model, weights, expected
    gc.collect()
    require("initial_native_unchanged_after_reference_preflight", state_digest(checkpoint(loop)) == initial_digest)
    before_schedule = checkpoint(loop)
    opt_in(loop)
    after_schedule = checkpoint(loop)
    require("schedule_opt_in_changes_config_only", state_digest({key: value for key, value in before_schedule.items() if key != "config"})
            == state_digest({key: value for key, value in after_schedule.items() if key != "config"}))
    require("schedule_opt_in_exact_config", loop.config == editing_config(initial["config"]))
    torch.save(after_schedule, args.output / "schedule-01600.pt")
    editing_boundary = torch.load(args.output / "schedule-01600.pt", map_location="cpu", weights_only=False, mmap=True)
    restore(loop, editing_boundary)
    require("tagged_schedule_strict_cpu_restore", state_digest(checkpoint(loop)) == state_digest(editing_boundary))
    continued = [training_update(loop), training_update(loop)]
    after_two = state_digest(checkpoint(loop))
    # Restore the actual CPU-mapped boundary without mutating any schema/owner.
    restore(loop, editing_boundary)
    replay = [training_update(loop), training_update(loop)]
    require("two_update_rows_exact", replay == continued)
    require("two_update_full_native_replay_exact", state_digest(checkpoint(loop)) == after_two)
    del before_schedule, after_schedule, editing_boundary
    restored_arch = loop.policy.G.architecture
    require("fast_ema_law_preserved", restored_arch == loop.policy.ema_G.architecture == GATED_PARTICLE_V3)
    write(args.output / "plan.json", plan)
    write(args.output / "run.json", dict(plan, config=loop.config, fixed_updates=args.steps,
          particlegan_source_digest=state_digest(pg_sha)))
    rows = original_rows + continued
    monitor = GameProgressMonitor(loop, data, fixed, args.output)
    monitor.dv12_state = common_initial["policy"]["streams"]["noise_generator"].clone()
    del common_initial
    gc.collect()
    emit(event="resume_exact", start_step=start_step, step=loop.policy.completed_steps, full_native_exact=True)

    def activity():
        policy = loop.policy
        return diagnostic(dict(accepted_moves=policy.routed_control.counters["moves"],
            optimizer_surprise_fires=policy.surprise.fires, anchor_release_events=policy.surprise.anchor_events,
            epoch_rebases=policy.reopen_guard.epoch_rebases, controller=policy.controller.diagnostics(),
            reopen_guard=policy.reopen_guard.state_dict(), lr_settle=policy.lr_settle.diagnostics()))

    def save(path):
        temporary = path.with_suffix(".tmp")
        torch.save(checkpoint(loop), temporary)
        temporary.replace(path)

    def paired(row):
        reference = control_rows[row["step"] - 1]
        fields = ("hold", "game_weight", "batch_indices", "base_noise_sums", "paired_rng_digest") if row["step"] <=1600 else (
            "base_noise_sums", "paired_rng_digest")
        for name in fields:
            require(f"stream_{row['step']}_{name}", row[name] == reference[name])
        if row["step"] > 1600:
            expected = editing_sampling[row["step"] - 1601]
            require("editing_task_" + str(row["step"]), all(row[name] == value for name, value in expected.items()))

    def budget_fields(step):
        return dict(training_schedule=EDITING_ONLY, editing_updates=1280 + step - 1600,
                    preservation_updates=320)

    def comparisons(metrics):
        comparison = {pool: dict(count=old_metrics[pool]["count"], control_rmse=old_metrics[pool]["rmse"],
            gated_rmse=metrics[pool]["rmse"], rmse_change_evaluation_only=metrics[pool]["rmse"] - old_metrics[pool]["rmse"],
            **{name: dict(control_g_game=old_metrics[pool][name]["g_game"], gated_g_game=metrics[pool][name]["g_game"],
                new_minus_control=metrics[pool][name]["g_game"] - old_metrics[pool][name]["g_game"])
               for name in ("fixed_start_D", "arm_final_D")}) for pool in old_metrics}
        ordinary = {label: {pool: dict(count=originals[label][pool]["count"], original_rmse=originals[label][pool]["rmse"],
            gated_rmse=metrics[pool]["rmse"], rmse_change_evaluation_only=metrics[pool]["rmse"] - originals[label][pool]["rmse"],
            **{name: dict(original_g_game=originals[label][pool][name]["g_game"], gated_g_game=metrics[pool][name]["g_game"],
                new_minus_original=metrics[pool][name]["g_game"] - originals[label][pool][name]["g_game"])
               for name in ("fixed_start_D", "arm_final_D")}) for pool in metrics} for label in original_refs}
        return comparison, ordinary

    endpoints = {}

    def evaluate_endpoint(step):
        before = state_digest(checkpoint(loop))
        metrics = evaluate_final(loop.policy.served_model(), data, fixed, final_judge, device)
        comparison, ordinary = comparisons(metrics)
        require("endpoint_native_state_immutable_" + str(step), state_digest(checkpoint(loop)) == before)
        require("endpoint_frozen_owners_" + str(step), frozen_digest(loop) == frozen)
        write(args.output / f"evaluation-v3-step-{step:05d}.json", metrics)
        result = dict(step=step, **budget_fields(step), native_digest=before,
            checkpoint_sha256=sha(args.output / f"checkpoint-{step:05d}.pt"),
            evaluation_sha256=sha(args.output / f"evaluation-v3-step-{step:05d}.json"),
            comparison=comparison, original_comparison=ordinary, native_state_unchanged=True,
            evaluation_only=True, output_metrics_used_for_selection=False,
            primary="editing learned-game convergence", preservation="secondary evaluation; no future preservation training")
        write(args.output / f"endpoint-{step:05d}.json", result)
        endpoints[str(step)] = result
        emit(event="fixed_endpoint", **result)
        return metrics
    for row in rows:
        paired(row)
    training_started = time.perf_counter()
    with (args.output / "train.jsonl").open("w") as trace:
        for row in rows:
            trace.write(json.dumps(row) + "\n")
        for step in range(start_step + 3, args.steps + 1):
            row = training_update(loop)
            require("clock_" + str(step), row["step"] == step)
            paired(row)
            rows.append(row)
            trace.write(json.dumps(row) + "\n")
            if step % 25 == 0 or step == args.steps:
                trace.flush()
                status = dict(phase="training", step=step, steps=args.steps, architecture=GATED_PARTICLE_V3,
                    particle_profile=SHARED_ROUTED_PROFILE, seconds=time.perf_counter() - training_started,
                    rolling=rolling_losses(rows), guard_activity=activity(), **budget_fields(step))
                write(args.output / "status.json", status)
                emit(event="training", **status)
            if step % 200 == 0 or step == 5440:
                emit(event="game_progress", **monitor.evaluate(loop, rows))
            if step % 400 == 0 or step == 5440:
                save(args.output / f"checkpoint-{step:05d}.pt")
            if step == 5440:
                write(args.output / "status.json", dict(phase="evaluating_fixed_endpoint", step=step, steps=args.steps,
                      architecture=GATED_PARTICLE_V3, particle_profile=SHARED_ROUTED_PROFILE, **budget_fields(step)))
                evaluate_endpoint(step)
    training_seconds = time.perf_counter() - training_started
    save(args.output / "final.pt")
    require("fixed_final_native_clock", loop.policy.completed_steps == args.steps)
    require("versioned_schedule_full_config_retained", loop.config == editing_config(initial["config"]))
    require("final_expected_fit_data_rng", torch.equal(loop.data_rng.get_state(), expected_data.get_state()))
    require("actual_edit_and_preserve_budget", sum(not row["hold"] for row in rows) == 6080
            and sum(row["hold"] for row in rows) == 320)
    require("frozen_host_teacher_unchanged", frozen_digest(loop) == frozen)
    require("dense_all_bank_after_acquisition", all(row["dense_gradient_rows"] == 128 for row in rows[1:]))
    dv12_equal = sum(row["dv12_rng_digest"] == control_rows[index]["dv12_rng_digest"] for index, row in enumerate(rows))
    if loop.policy.routed_control.counters["moves"] == 0:
        require("all_dv12_streams_exact_without_moves", dv12_equal == args.steps)
    native_digest = state_digest(checkpoint(loop))
    write(args.output / "status.json", dict(phase="evaluating", step=args.steps, steps=args.steps,
          architecture=GATED_PARTICLE_V3, particle_profile=SHARED_ROUTED_PROFILE, guard_activity=activity(), **budget_fields(args.steps)))
    audit = audit_gated(loop, data["fit"]["context"][:4])
    require("final_71_actual_gated_sites", len(audit["branches"]) == len(audit["per_site_interventions"]) == 71
            and all(value["formula_exact"] for value in audit["branches"].values()))
    for mode in ("clean_gradient", "dv12_gradient"):
        grad = audit[mode]
        require("final_" + mode + "_dense_owners", grad["bank_rows_nonzero"] == 128
                and grad["router_tensors_nonzero"] == 142 and grad["generator_tensors_nonzero"] == 284
                and math.isfinite(grad["g_game"]) and all(math.isfinite(value) for value in grad["parameter_gradient_norms"].values()))
    audit["final_output_site_effects_nonzero_diagnostic"] = sum(
        value["output_change_rms_evaluation_only"] > 0 for value in audit["per_site_interventions"].values())
    require("final_whole_model_particle_dependence", audit["interventions"]["zero_particle_codes"]["output_change_rms_evaluation_only"] > 0)
    write(args.output / "audit-final.json", audit)
    served = loop.policy.served_model()
    new_metrics = evaluate_endpoint(args.steps)
    write(args.output / "evaluation-v3.json", new_metrics)
    contributions = {pool: evaluate_pool(served, data[pool], {"common_D1856": fixed, "common_control_final_D": final_judge},
                    pool_name=pool) for pool in ("fit", "test", "holds", "preservation")}
    write(args.output / "particle-contribution.json", contributions)
    export = export_served_adapter(served, args.output / "final.safetensors",
        extra_metadata=dict(selection=f"fixed{args.steps} horizon", output_metrics="evaluation only"))
    with torch.random.fork_rng(devices=[device.index or 0]):
        clean = load_particle_adapter(base, args.output / "final.safetensors", device=device)
    context = data["test"]["context"][:4].to(device)
    z, t, ctx, mask, uctx, umask, strength = served.encoder.unpack(context)
    require("v3_clean_export_reload_exact", torch.equal(raw_velocity(served, context),
            clean.velocity(z, t, ctx, mask, uctx, umask, strength=strength)))
    require("v3_export_tag_explicit", export["config"]["architecture"] == clean.generator.architecture == GATED_PARTICLE_V3)
    require("all_v3_evaluation_native_state_immutable", state_digest(checkpoint(loop)) == native_digest)
    del clean, context, served, initial
    gc.collect()
    comparison, original_comparison = comparisons(new_metrics)
    require("application_sources_unchanged", all(sha(ROOT / name) == value for name, value in app_sha.items()))
    require("native_sources_unchanged", all(sha(pg_root / name) == value for name, value in pg_sha.items()))
    require("immutable_inputs_unchanged", all(sha(input_paths[name]) == value for name, value in inputs_sha.items()))
    receipt = dict(qualified=True, plan=plan, checks=checks, comparison=comparison, original_comparison=original_comparison,
        initial_native_digest=initial_digest, final_native_digest=native_digest,
        final_checkpoint_sha256=sha(args.output / "final.pt"), final_export_sha256=sha(args.output / "final.safetensors"),
        control_native_digest=control_digest, export=export, dv12_streams_equal=dv12_equal,
        endpoints=endpoints, schedule_checkpoint_sha256=sha(args.output / "schedule-01600.pt"),
        training_seconds=training_seconds, total_seconds=time.perf_counter() - started, controller=activity(),
        critics={"fixed_start_D": "same frozen V2D1856", "arm_final_D": f"same qualified V2D{args.steps}"})
    write(args.output / "receipt.json", receipt)
    write(args.output / "status.json", dict(phase="complete", step=args.steps, steps=args.steps,
        architecture=GATED_PARTICLE_V3, particle_profile=SHARED_ROUTED_PROFILE, comparison=comparison,
        original_comparison=original_comparison, guard_activity=activity(), endpoints=endpoints, **budget_fields(args.steps)))
    emit(event="complete", comparison=comparison, original_comparison=original_comparison, native_digest=native_digest,
         training_seconds=training_seconds, total_seconds=receipt["total_seconds"])


if __name__ == "__main__":
    main()

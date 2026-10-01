#!/usr/bin/env python3
"""Five real Supra updates and strict recovery for the shared PR223 profile.

The qualified historical checkpoint supplies only frozen teacher tensors to
the fresh shared run. Its trained particles, branches, optimizers and streams
are used solely in a separate legacy-restore compatibility check. No output
metric changes training, structural decisions or checkpoint selection.
"""
import argparse
from contextlib import contextmanager
from copy import deepcopy
import gc
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PG_PIN = "bdf05d1be0f68cfdb0c71e81e7e0d3cce477572f"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    def diagnostic_json(item):
        import torch
        if isinstance(item, torch.Tensor):
            return item.detach().cpu().tolist()
        raise TypeError(f"unsupported diagnostic JSON type: {type(item).__name__}")
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False, default=diagnostic_json) + "\n")


def emit(**row):
    print(json.dumps(row, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "outputs/e22-particle-v2-6400")
    parser.add_argument("--particlegan-root", type=Path,
                        default=ROOT.parent / "ParticleGAN-pr223-supra")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs/e22-pr223-shared-profile-smoke")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.run, args.particlegan_root, args.output = (
        path.resolve() for path in (args.run, args.particlegan_root, args.output))
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("choose a fresh output directory")
    actual_pin = subprocess.check_output(
        ["git", "-C", str(args.particlegan_root), "rev-parse", "HEAD"], text=True).strip()
    if actual_pin != PG_PIN:
        raise RuntimeError(f"ParticleGAN checkout must be {PG_PIN}; found {actual_pin}")
    if subprocess.check_output(["git", "-C", str(args.particlegan_root), "status", "--porcelain",
                                "--", "particlegan"], text=True).strip():
        raise RuntimeError("ParticleGAN package source differs from the pinned commit")
    sys.path.insert(0, str(args.particlegan_root))
    sys.path.insert(1, str(ROOT))
    import torch
    import particlegan
    from particlegan import RoutedRows
    from particlegan.routing import RoutedExecution
    from supra.runtime import TARGETS, model_module
    from supra.particle_game import features_for_rows
    from supra.particle_pilot import checkpoint, frozen_digest, restore, state_digest
    from supra.particle_training import (
        LEGACY_ROUTED_PROFILE, SHARED_ROUTED_PROFILE, make_training_loop,
        raw_model_forward, training_update,
    )
    from supra.particle_training_data import SOURCE_COLUMN, STRENGTH_COLUMN, TARGET_COLUMN

    if Path(particlegan.__file__).resolve().parent.parent != args.particlegan_root:
        raise RuntimeError("unexpected ParticleGAN import")
    torch.set_num_threads(4)
    device = torch.device(args.device)
    if device.type != "cuda":
        parser.error("this receipt qualifies the real BF16 CUDA host")
    torch.cuda.set_device(device)
    began = time.perf_counter()
    args.output.mkdir(parents=True)
    source_files = [Path(__file__)] + sorted((ROOT / "supra").glob("particle*.py"))
    source_files.append(ROOT / "supra/runtime.py")
    source_hashes = {str(path.relative_to(ROOT)): sha(path) for path in source_files}
    pg_files = sorted((args.particlegan_root / "particlegan").rglob("*.py"))
    pg_hashes = {str(path.relative_to(args.particlegan_root)): sha(path) for path in pg_files}
    for path in source_files:
        target = args.output / "source" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    for path in pg_files:
        target = args.output / "source/particlegan" / path.relative_to(args.particlegan_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    plan = dict(
        schema="supra_pr223_shared_profile_smoke_v1", particlegan_pin=PG_PIN,
        particlegan_source_sha256=pg_hashes, application_source_sha256=source_hashes,
        imported_particlegan=str(Path(particlegan.__file__).resolve()),
        profile=SHARED_ROUTED_PROFILE, updates=5, checkpoint_step=2, replay_updates=3,
        input_directory=str(args.run), input_sha256={name: sha(args.run / name) for name in
            ("final.pt", "data.pt", "qualification-review.json")},
        device=str(device), gpu=torch.cuda.get_device_name(device), torch_version=torch.__version__,
        initialization="public deterministic_orthogonal_ from fresh role streams; zero particle output branches",
        historical_input_use="frozen teacher only in fresh shared run; separate strict legacy restore check",
        objective="native paired RpGAN/KA2 only; no output metric or structural output guard",
        selection="fixed five-update horizon; no seed sweep, hyperparameter or checkpoint selection",
        limitations="integration/recovery smoke; interval100 structural proposals and contracted-ladder surprise firing are not exercised; no convergence claim",
    )
    write(args.output / "plan.json", plan)
    emit(event="plan", profile=SHARED_ROUTED_PROFILE, pin=PG_PIN, updates=5,
         checkpoint_step=2, replay_updates=3)
    checks = {}

    def require(name, condition):
        condition = bool(condition)
        checks[name] = condition
        if not condition:
            write(args.output / "failed-checks.json", checks)
            raise RuntimeError(f"qualification failed: {name}")

    saved = torch.load(args.run / "final.pt", map_location="cpu", weights_only=False, mmap=True)
    data = torch.load(args.run / "data.pt", map_location="cpu", weights_only=False)
    qualified = json.loads((args.run / "qualification-review.json").read_text())
    old_digest = state_digest(saved)
    require("qualified_historical_source", qualified.get("qualified") is True
            and saved["policy"]["completed_steps"] == 6400
            and old_digest == qualified["final_native_digest"])
    require("qualified_cached_dataset", state_digest(data) == saved["config"]["dataset_digest"])
    teacher_source = {key.removeprefix("teacher."): tensor for key, tensor in
                      saved["policy"]["models"]["encoder"].items() if key.startswith("teacher.")}
    teacher_digest = state_digest(teacher_source)
    with torch.random.fork_rng(devices=[device.index or 0]):
        mod = model_module()
        with torch.device("meta"):
            base = mod.SupraDiT()
            mod.attach_supra_lora(base, rank=16, alpha=16, targets=TARGETS)
        base.load_state_dict(teacher_source, strict=True, assign=True)
        base.to(device).eval().requires_grad_(False)
        legacy = make_training_loop(base, data, device=device,
            architecture=saved["config"]["architecture"], branch_lr=saved["config"]["branch_lr"],
            probe_interval=saved["config"]["probe_interval"], profile=LEGACY_ROUTED_PROFILE)
    restore(legacy, saved)
    require("legacy_checkpoint_restore_full_native_exact", state_digest(checkpoint(legacy)) == old_digest)
    require("legacy_defaults_omit_new_recipe_fields", not any(name in legacy.config["recipe"]
            for name in ("reopen_guard", "birth_death_backend", "birth_death_cells",
                         "birth_death_metric_rank", "birth_death_chunk", "birth_death_parent_policy")))
    emit(event="legacy_restore", full_native_exact=True, native_digest=old_digest)
    del legacy
    gc.collect()
    torch.cuda.empty_cache()
    with torch.random.fork_rng(devices=[device.index or 0]):
        loop = make_training_loop(base, data, device=device,
            architecture=saved["config"]["architecture"], branch_lr=saved["config"]["branch_lr"],
            probe_interval=saved["config"]["probe_interval"], profile=SHARED_ROUTED_PROFILE)
    policy = loop.policy
    fresh_digest = state_digest(checkpoint(loop))
    frozen_before = frozen_digest(loop)
    require("fresh_start_step_zero", policy.completed_steps == 0)
    require("only_frozen_teacher_loaded", state_digest(policy.encoder.teacher.state_dict()) == teacher_digest
            and state_digest(base.state_dict()) == teacher_digest)
    require("fresh_optimizers_no_trained_state", all(not optimizer.state for optimizer in policy.optimizers))
    require("fresh_zero_output_all_71_particle_branches", len(policy.G.sites) == 71
            and all(not bool(branch.up.weight.detach().count_nonzero()) for branch in policy.G.particle_branches()))
    require("shared_profile_explicit", loop.config["particle_profile"] == SHARED_ROUTED_PROFILE
            and policy.recipe.birth_death_backend == "auto" and policy.recipe.reopen_guard == "settled")
    require("native_particles_and_shared_control", tuple(policy.table.shape) == (128, 4)
            and policy.table.requires_grad and policy.birth_death is policy.routed_control
            and policy.row_evidence is policy.routed_control.evidence and policy.reopen_guard is not None)
    require("native_guard_is_zero_feature_harm_only", policy.routed_control.spec.max_context_harm == 0.
            and not policy.routed_control.spec.output_error_guard and loop.config["output_error_guard"] is False)
    rejections = {}
    for label, operation in (
        ("application", lambda: restore(loop, saved)),
        ("native_policy", lambda: policy.load_state_dict(saved["policy"])),
    ):
        try:
            operation()
        except ValueError as error:
            rejections[label] = str(error)
        require(f"old_checkpoint_rejected_by_shared_{label}", label in rejections)
        require(f"shared_{label}_rejection_transactional", state_digest(checkpoint(loop)) == fresh_digest)
    del saved, teacher_source
    gc.collect()

    @contextmanager
    def evaluation_modes():
        previous = [(module, module.training) for owner in policy._training_modules().values()
                    for module in owner.modules()]
        try:
            for module, _ in previous:
                module.training = False
            yield
        finally:
            for module, training in previous:
                module.training = training

    @torch.no_grad()
    def base_identity(*, zero_strength):
        context = loop.fit_context[:4].clone()
        context[:, TARGET_COLUMN] = context[:, SOURCE_COLUMN]
        if zero_strength:
            context[:, STRENGTH_COLUMN] = 0.
        spec = RoutedRows(model_forward=raw_model_forward, features=features_for_rows, sites=policy.G.sites)
        before = state_digest(checkpoint(loop))
        with evaluation_modes():
            candidate = policy.routed_control.candidate()
            actual = spec.forward(policy._training_modules(), context, candidate)
            expected = policy.encoder.teacher_velocity(context)
        require("zero_strength_state_preserved" if zero_strength else "fresh_base_state_preserved",
                state_digest(checkpoint(loop)) == before)
        return bool(torch.equal(actual, expected))

    require("fresh_particle_function_matches_frozen_source_base", base_identity(zero_strength=False))

    def cpu_clone(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {key: cpu_clone(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cpu_clone(item) for item in value]
        if isinstance(value, tuple):
            return tuple(cpu_clone(item) for item in value)
        return deepcopy(value)

    def cpu_tensors_only(value):
        if isinstance(value, torch.Tensor):
            return value.device.type == "cpu"
        if isinstance(value, dict):
            return all(cpu_tensors_only(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return all(cpu_tensors_only(item) for item in value)
        return True

    calls = []
    original_mix = RoutedExecution.mix

    def counted_mix(owner, site_name, logits):
        result = original_mix(owner, site_name, logits)
        calls.append(site_name)
        return result

    rows, replay_rows, site_calls, replay_site_calls = [], [], [], []
    snapshot_digest = None
    RoutedExecution.mix = counted_mix
    try:
        for index in range(5):
            calls.clear()
            row = training_update(loop)
            rows.append(row)
            site_calls.append(list(calls))
            require(f"native_step_{index + 1}_all_71_sites_in_both_passes", calls == list(policy.G.sites) * 2)
            emit(event="native_update", **row)
            if index == 1:
                snapshot = cpu_clone(checkpoint(loop))
                snapshot_digest = state_digest(snapshot)
                require("cpu_checkpoint_all_tensors_cpu", cpu_tensors_only(snapshot))
                require("cpu_checkpoint_conversion_preserves_full_native_state",
                        snapshot_digest == state_digest(checkpoint(loop)))
                torch.save(snapshot, args.output / "checkpoint-00002.pt")
                del snapshot
                emit(event="cpu_checkpoint", step=2, native_digest=snapshot_digest,
                     path=str(args.output / "checkpoint-00002.pt"))
        final_digest = state_digest(checkpoint(loop))
        require("frozen_parameters_unchanged_after_five_updates", frozen_digest(loop) == frozen_before)
        restored = torch.load(args.output / "checkpoint-00002.pt", map_location="cpu", weights_only=False)
        require("on_disk_cpu_checkpoint_full_native_exact", state_digest(restored) == snapshot_digest)
        restore(loop, restored)
        del restored
        require("strict_restore_step_two_full_native_exact", state_digest(checkpoint(loop)) == snapshot_digest)
        require("restored_active_model_and_bank_owners_on_gpu", policy.table.device == device
                and all(tensor.device == device for module in policy._training_modules().values()
                        for tensor in (*module.parameters(), *module.buffers())))
        for index in range(3):
            calls.clear()
            row = training_update(loop)
            replay_rows.append(row)
            replay_site_calls.append(list(calls))
            require(f"replay_step_{index + 3}_row_exact", row == rows[index + 2])
            require(f"replay_step_{index + 3}_site_sequence_exact", calls == site_calls[index + 2])
            emit(event="replay_update", **row)
        replay_digest = state_digest(checkpoint(loop))
        require("replayed_final_full_native_exact", replay_digest == final_digest)
    finally:
        RoutedExecution.mix = original_mix
    require("read_only_class_observer_restored", RoutedExecution.mix is original_mix)
    require("preservation_exercised_at_step_five", [row["hold"] for row in rows] == [False] * 4 + [True]
            and rows[-1]["game_weight"] == .1 and replay_rows[-1]["hold"] is True)
    require("dense_bank_gradient_after_zero_branch_acquisition", all(row["dense_gradient_rows"] == 128
            and row["bank_grad_norm"] > 0. for row in rows[1:]))
    require("zero_strength_exact_after_training", base_identity(zero_strength=True))
    require("frozen_parameters_unchanged_after_replay", frozen_digest(loop) == frozen_before)
    native = checkpoint(loop)
    selection = native["policy"]["backend_selection"]
    require("auto_defers_to_native_routed_owner", selection["actual_backend"] == "routed"
            and selection["sampling_backend"] == "routed"
            and selection["selection_reason"] == "routed_rows_owns_controls"
            and selection["generator_noise_factor"] == 1.)
    require("auto_keeps_all_role_base_rates", all(item["factor"] == 1. for group in selection["rate_mapping"]
            for item in group))
    require("guard_state_present_and_replayed", "reopen_guard" in native["policy"]
            and policy.reopen_guard is not None)
    require("probe_interval_remains_100", policy.routed_control.spec.probe_interval == 100)
    require("application_sources_held_during_run", all(sha(ROOT / name) == value for name, value in source_hashes.items()))
    require("native_sources_held_during_run", all(sha(args.particlegan_root / name) == value
            for name, value in pg_hashes.items()))
    result = dict(plan=plan, qualified=True, checks=checks, config=loop.config, rows=rows, replay_rows=replay_rows,
        historical_native_digest=old_digest, historical_shared_rejections=rejections,
        fresh_native_digest=fresh_digest, checkpoint_native_digest=snapshot_digest,
        final_native_digest=final_digest, replay_final_native_digest=replay_digest,
        frozen_parameter_digest=frozen_before, teacher_state_digest=teacher_digest,
        routing_site_calls=site_calls, replay_routing_site_calls=replay_site_calls,
        backend_selection=selection, reopen_guard=native["policy"]["reopen_guard"],
        routed_diagnostics=policy.routed_control.diagnostics(),
        elapsed_seconds=time.perf_counter() - began,
        max_gpu_memory_bytes=torch.cuda.max_memory_allocated(device))
    write(args.output / "result.json", result)
    emit(event="qualified", checks=len(checks), native_digest=final_digest,
         final_step=policy.completed_steps, elapsed_seconds=result["elapsed_seconds"])


if __name__ == "__main__":
    main()

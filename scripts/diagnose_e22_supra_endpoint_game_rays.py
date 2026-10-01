#!/usr/bin/env python3
"""Post-run signed native-game rays on fixed, qualified Supra FAST exports.

This separate diagnostic has zero updates and no selection or quality gate.
It must wait for the full two-arm run's independently qualified completion.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

EXECUTION_STARTED = time.monotonic()
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment_e22_supra_particle_gated import emit, sha, write
from scripts.review_e22_supra_neutral_initialization import Reviewer, canonical, check_evaluation, source_digest
from scripts.review_e22_supra_pr223 import state_digest

ALPHAS = (-1., -.25, -.05, -.01, 0., .01, .05, .25, .5, 1.)
ARMS = ("sampled_control", "sampled_hb_neutral")
ENDPOINTS = (5120, 6400)
POOLS = ("fit", "test", "holds", "preservation")
COUNTS = dict(zip(POOLS, (240, 240, 30, 60)))
JUDGES = ("D1856", "D6400")
SUBJECTS = (4, 5, 15, 16, 17, 18)
SIGMA, DRAWS, BATCH, TOKENS = .125, 4, 4, 256
PIN = "6ec7e5788e14ea15ddc3e16ac71110458108b6a6"
CARD = ROOT / "docs/e22_supra_endpoint_game_rays_v1.json"


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, label):
    if not condition:
        raise ValueError(label)


def physical_gpu0(environment=None):
    environment = os.environ if environment is None else environment
    require(environment.get("CUDA_VISIBLE_DEVICES") == "0", "physical GPU0 requires CUDA_VISIBLE_DEVICES=0")


def check_budget(started, limit=1200, clock=time.monotonic):
    elapsed = clock() - started
    if elapsed > limit:
        raise TimeoutError("separate posthoc1200-second budget exhausted")
    return elapsed


def final_report(output, report, started, limit=1200, clock=time.monotonic, writer=write):
    """Completion requires a post-serialization deadline check and bound receipt.

    A report without a complete completion.json is not qualified evidence.
    On a detected overrun, preserve results but mark both receipts incomplete.
    """
    report = {**report, "completion_receipt_required": True,
              "seconds_before_report_serialization": clock() - started}
    try:
        writer(output / "report.json", report)
        elapsed = check_budget(started, limit, clock)
        digest = sha(output / "report.json")
        check_budget(started, limit, clock)
        writer(output / "completion.json", dict(complete=True, qualified=True,
            report_sha256=digest, wall_seconds=clock() - started,
            clock_scope="After main report serialization/write/SHA; deadline also checked after this receipt write."))
        return check_budget(started, limit, clock)
    except TimeoutError:
        report.update(qualified=False, complete=False, budget_overrun=True,
                      seconds_at_overrun=clock() - started)
        writer(output / "report.json", report)
        writer(output / "completion.json", dict(complete=False, qualified=False, budget_overrun=True,
            report_sha256=sha(output / "report.json"), wall_seconds=clock() - started))
        raise


def validate_card(card):
    """The diagnostic accepts no amplitude, noise, pool, head or budget overrides."""
    require(card["schema"] == "supra_endpoint_game_rays_card_v1", "card schema")
    require(tuple(card["alphas"]) == ALPHAS and card["radial_derivative_alphas"] == [0, 1], "fixed signed rays")
    require(tuple(card["pools"]) == POOLS and card["counts"] == COUNTS, "all570 fixed contexts")
    require(tuple(card["judges"]) == JUDGES and tuple(card["editing_subjects"]) == SUBJECTS, "both judges/all six editing subjects")
    require(card["panel_law"] == dict(seed=72, draws=DRAWS, batch_size=BATCH, tokens=TOKENS,
        coordinates=16, sigma=SIGMA, pool_order=list(POOLS), device="cpu"), "fixed private panel law")
    require(card["execution_budget_seconds"] == 1200 and card["device"] == "cuda:0"
        and card["training_updates"] == 0 and card["selection"] is False, "posthoc budget/no selection")
    require(card["physical_gpu_binding"] == "CUDA_VISIBLE_DEVICES=0", "physical GPU0 binding")
    require(card["comparison_tolerance"] == 1e-6 and card["alpha1_match_tolerance"] == dict(atol=2e-6, rtol=1e-5), "fixed numerical tolerances")
    require(card["particlegan_commit"] == PIN, "fixed native revision")
    expected = [arm + "@" + str(step) for arm in ARMS for step in ENDPOINTS] + ["ordinary_lora_6400"]
    require(list(card["artifacts"]) == expected, "all five fixed artifact identities")


def qualification_gate(card, plan, status, receipt, review, review_sha256):
    """Fail before loading models/checkpoints/CUDA if completion is unqualified."""
    require(status["phase"] == receipt["status"] == "complete"
        and status["step"] == status["steps"] == status["editing_updates"] == 6400
        and status["preservation_updates"] == 0, "full run must be complete before diagnostic")
    require(review.get("qualified") is True and review.get("partial") is False
        and review.get("schema") == "supra_neutral_initialization_independent_review_v1", "independent full-run qualification required")
    require(review_sha256 == card["runtime_review_sha256"], "explicit frozen full-review SHA required")
    require(receipt["plan"] == plan and receipt["checks"] and all(v is True for v in receipt["checks"].values()), "qualified held runtime witnesses")
    require(review["protocol_sha256"] == card["parent_protocol_sha256"]
        and review["plan_sha256"] == card["parent_plan_sha256"], "exact qualified protocol/plan")
    require(review["reviewer_sha256"] == card["reviewer_sources_sha256"]["scripts/review_e22_supra_neutral_initialization.py"], "reviewer source identity")
    require(review["application_source_sha256"] == plan["application_source_sha256"]
        and review["native_source_digest"] == card["native_source_digest"], "qualified application/native sources")
    require(review["input_sha256"] == plan["input_sha256"], "qualified immutable input identities")
    expected_dependencies = {str(ROOT / name): digest for name, digest in card["reviewer_sources_sha256"].items()
                             if name != "scripts/review_e22_supra_neutral_initialization.py"}
    require(review["reviewer_dependency_sha256"] == expected_dependencies, "reviewer dependency identity")


def module_witness(module):
    """Include content, modes, gradient ownership and temporary routing attributes."""
    return state_digest(dict(state=module.state_dict(),
        parameters={name: (p.requires_grad, p.grad) for name, p in module.named_parameters()},
        modes={name: child.training for name, child in module.named_modules()},
        transient={name: {key: getattr(child, key) for key in ("multiplier", "_in_forward", "frame", "cfg")
                         if hasattr(child, key)} for name, child in module.named_modules()},
        metadata=getattr(module, "metadata", None)))


def rng_witness(device=None):
    value = dict(cpu=torch.get_rng_state().clone())
    if device is not None:
        value["cuda0"] = torch.cuda.get_rng_state(device).clone()
    return state_digest(value)


def panel_batches(data, expected_hashes):
    """Match the held evaluator's B4 draws, continuously across all four pools."""
    stream, batches, hashes = torch.Generator().manual_seed(72), {}, {}
    for pool in POOLS:
        batches[pool], digest = [], hashlib.sha256()
        for start in range(0, len(data[pool]["context"]), BATCH):
            count = min(BATCH, len(data[pool]["context"]) - start)
            panel = torch.randn(DRAWS, count, TOKENS, 16, generator=stream, dtype=torch.float32)
            digest.update(panel.contiguous().numpy().tobytes(order="C"))
            batches[pool].append(panel)
        hashes[pool] = digest.hexdigest()
    require(hashes == expected_hashes, "exact held full-pool unscaled Gaussian hashes")
    return batches, hashes, state_digest(stream.get_state())


def score_ray(judge, normalized_residual, condition, panel):
    """Per-context native game and dL/dalpha; never average the context derivative."""
    draws, batch = panel.shape[:2]
    condition = condition.repeat(draws, 1)
    with torch.no_grad():
        real = judge(panel.flatten(0, 1), condition).detach()
    games, derivatives = {}, {}
    for fraction in ALPHAS:
        alpha = torch.full((batch,), fraction, device=panel.device, dtype=panel.dtype,
                           requires_grad=fraction in (0., 1.))
        with torch.set_grad_enabled(alpha.requires_grad):
            fake = panel + alpha.reshape(1, batch, 1, 1) * normalized_residual.unsqueeze(0)
            values = F.softplus(real - judge(fake.flatten(0, 1), condition)).reshape(draws, batch).mean(0)
            key = f"{fraction:g}"
            games[key] = values.detach().cpu()
            if alpha.requires_grad:
                derivatives[key] = torch.autograd.grad(values.sum(), alpha)[0].detach().cpu()
        require(bool(torch.isfinite(games[key]).all()), "finite native ray game")
    for value in derivatives.values():
        require(bool(torch.isfinite(value).all()), "finite radial derivative")
    torch.testing.assert_close(games["0"], torch.full_like(games["0"], math.log(2)), rtol=0, atol=2e-7)
    return games, derivatives


def summarize_ray(games, derivatives, subjects, tolerance):
    """Retain both signs; a restorative endpoint does not rank different rays."""
    positive = [f"{alpha:g}" for alpha in ALPHAS if alpha >= 0]
    def reduce(indices):
        means = {key: float(value[indices].double().mean()) for key, value in games.items()}
        radial = {key: dict(mean=float(value[indices].double().mean()),
            negative_contexts=int((value[indices] < -tolerance).sum()),
            positive_contexts=int((value[indices] > tolerance).sum()),
            within_tolerance_contexts=int((value[indices].abs() <= tolerance).sum()))
            for key, value in derivatives.items()}
        asymmetry = {f"{alpha:g}": float((games[f"{alpha:g}"][indices]
            - games[f"{-alpha:g}"][indices]).double().mean()) for alpha in (.01, .05, .25, 1.)}
        reversals = {left + "->" + right: int((games[right][indices] + tolerance < games[left][indices]).sum())
                     for left, right in zip(positive[:-1], positive[1:])}
        return dict(contexts=len(indices), games=means, radial_derivatives=radial,
            positive_minus_negative_game=asymmetry, positive_ray_reversals=reversals,
            below_log2_by_alpha={key: int((value[indices] + tolerance < math.log(2)).sum()) for key, value in games.items()})
    result = reduce(torch.arange(len(subjects)))
    result["subjects"] = {str(int(subject)): reduce((subjects == subject).nonzero().flatten())
                          for subject in subjects.unique(sorted=True)}
    return result


def qualified_evaluations(run, review, data):
    """Bind original per-context metadata/reductions to the completed CPU review."""
    original = {"ordinary_lora_6400": read(run / "historical-references.json")["ordinary_lora_6400"]}
    for arm in ARMS:
        for step in ENDPOINTS:
            original[arm + "@" + str(step)] = read(run / arm / f"evaluation-{step:05d}.json")
    for label, result in original.items():
        reduced = check_evaluation(Reviewer(), result, data, label)
        require(canonical(reduced) == review["evaluation_summaries"][label], "qualified all-pool record reductions: " + label)
    return original


def preflight(run, pg, card, review_sha256):
    validate_card(card)
    require(run.resolve() == Path(card["parent_run"]).resolve(), "fixed parent run path")
    require(pg.resolve() == Path(card["particlegan_root"]).resolve(), "fixed native root")
    require(review_sha256 and len(review_sha256) == 64, "--qualified-review-sha256 must freeze the completed review")
    # This in-memory binding is recorded in the immutable diagnostic plan; no card edit.
    card = {**card, "runtime_review_sha256": review_sha256}
    plan, status, receipt, review = (read(run / name) for name in
        ("plan.json", "status.json", "receipt.json", "independent-review.json"))
    require(sha(run / "independent-review.json") == review_sha256, "actual completed review SHA")
    qualification_gate(card, plan, status, receipt, review, review_sha256)
    require(review["gpu_runtime_witnesses"]["receipt_sha256"] == sha(run / "receipt.json"), "exact qualified held GPU receipt")
    require(review["full_evaluation_panel_sha256"] == card["unscaled_panel_sha256_by_pool"], "qualified private panel identities")
    require(sha(run / "plan.json") == card["parent_plan_sha256"]
        and sha(ROOT / "docs/e22_supra_neutral_initialization_protocol.json") == card["parent_protocol_sha256"], "held plan/card unchanged")
    require(plan["particlegan_commit"] == PIN and plan["protocol"]["particlegan"]["python_source_digest"] == card["native_source_digest"], "parent native law")
    require(subprocess.check_output(["git", "-C", str(pg), "rev-parse", "HEAD"], text=True).strip() == PIN, "actual native revision")
    require(not subprocess.check_output(["git", "-C", str(pg), "status", "--porcelain", "--", "particlegan"], text=True).strip(), "native package must be clean")
    files = {}
    for name, digest in {**plan["application_source_sha256"], **card["diagnostic_sources_sha256"], **card["reviewer_sources_sha256"]}.items():
        path = ROOT / name
        require(sha(path) == digest, "application/diagnostic/reviewer source changed: " + name)
        files[str(path.resolve())] = digest
    native = {str(path.relative_to(pg)): sha(path) for path in sorted((pg / "particlegan").rglob("*.py"))}
    require(native == plan["particlegan_source_sha256"] and source_digest(native) == card["native_source_digest"], "exact native Python source")
    files.update({str((pg / name).resolve()): digest for name, digest in native.items()})
    for name in ("data", "data_provenance", "frozen_teacher", "frozen_teacher_review", "common_D1856",
                 "common_V2_D6400", "historical_ordinary_6400", "backend_lock", "backend_model_source"):
        declared = plan["protocol"]["inputs"][name]
        path = Path(declared["path"])
        require(sha(path) == declared["sha256"], "original input changed: " + name)
        files[str(path.resolve())] = declared["sha256"]
    files.update({str((run / name).resolve()): sha(run / name) for name in
        ("plan.json", "status.json", "receipt.json", "independent-review.json", "historical-references.json")})
    artifacts = {}
    for label, declared in card["artifacts"].items():
        path = Path(declared["path"])
        if label == "ordinary_lora_6400":
            require(path.resolve() == Path(plan["protocol"]["inputs"]["historical_ordinary_6400"]["path"]).resolve(), "fixed historical ordinary path")
            expected = plan["protocol"]["inputs"]["historical_ordinary_6400"]["sha256"]
        else:
            arm, step = label.split("@")
            require(path.resolve() == (run / arm / f"adapter-{int(step):05d}.safetensors").resolve(), "fixed fresh export path")
            expected = review["export_artifacts"][label]["sha256"]
            require(review["export_artifacts"][label]["config"]["served_source"] == "fast", "qualified FAST export")
        require(sha(path) == expected, "exact qualified artifact: " + label)
        artifacts[label] = dict(path=str(path.resolve()), sha256=expected)
        files[str(path.resolve())] = expected
        if label != "ordinary_lora_6400":
            arm, step = label.split("@")
            path = run / arm / f"evaluation-{int(step):05d}.json"
            require(read(path)["test"] == receipt["endpoint_test_scores"][label], "exact qualified endpoint test records")
            files[str(path.resolve())] = sha(path)
    require(review["judges"] == card["critic_tensor_digests"], "actual qualified judge tensor identities")
    # Loading frozen data is distinct from loading a full native/model checkpoint.
    # Validate reference metadata before CUDA/model/checkpoint initialization.
    data = torch.load(Path(plan["input_paths"]["data"]), map_location="cpu", weights_only=False)
    require(state_digest(data) == plan["protocol"]["data"]["digest"], "qualified frozen dataset identity")
    original = qualified_evaluations(run, review, data)
    return card, plan, review, artifacts, files, data, original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "outputs/e22-supra-neutral-initialization-6400")
    parser.add_argument("--particlegan-root", type=Path, default=ROOT.parent / "ParticleGAN-supra-neutral-develop")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/e22-supra-endpoint-game-rays-v1")
    parser.add_argument("--qualified-review-sha256", help="Explicit completed independent-review identity, frozen before capture.")
    parser.add_argument("--preflight-only", action="store_true", help="Validate qualified artifact/source bindings without loading models/checkpoints or CUDA.")
    args = parser.parse_args()
    started = EXECUTION_STARTED
    initial_cpu_rng = torch.get_rng_state().clone()
    require(not args.output.exists(), "preserve existing diagnostic artifacts")
    card, parent_plan, review, artifacts, files, data, original_evaluations = preflight(
        args.run, args.particlegan_root, read(CARD), args.qualified_review_sha256)
    require(torch.equal(torch.get_rng_state(), initial_cpu_rng), "global CPU RNG unchanged during preflight/data validation")
    files[str(CARD)] = sha(CARD)
    try:
        preflight_seconds = check_budget(started, card["execution_budget_seconds"])
        if args.preflight_only:
            emit(preflight_passed=True, model_forward_calls=0, full_checkpoint_loads=0, frozen_data_loads=1,
                 CUDA_initialized=torch.cuda.is_initialized(), seconds=preflight_seconds)
            check_budget(started, card["execution_budget_seconds"])
            return
    except TimeoutError as error:
        args.output.mkdir(parents=True)
        write(args.output / "failure.json", dict(phase="preflight", qualified=False, budget_overrun=True,
            error=str(error), seconds=time.monotonic() - started, training_updates=0,
            model_forward_calls=0, full_checkpoint_loads=0, CUDA_initialized=torch.cuda.is_initialized()))
        raise
    physical_gpu0()
    sys.path.insert(0, str(args.particlegan_root.resolve()))
    import particlegan
    from safetensors import safe_open
    from safetensors.torch import load_file
    from supra.runtime import TARGETS, MODEL_SOURCE, model_module, load_adapter_state
    from supra.particle_export import load_particle_adapter, native_pins
    from supra.particle_game import ConditionalTokenCritic, patchify
    from supra.particle_training_data import FrozenSliderContexts

    check_budget(started, card["execution_budget_seconds"])
    require(Path(particlegan.__file__).resolve().parent.parent == args.particlegan_root.resolve(), "actual native package import")
    require(sha(MODEL_SOURCE) == parent_plan["protocol"]["inputs"]["backend_model_source"]["sha256"], "actual pinned backend model source")
    require(sha(ROOT / "backend.lock.json") == parent_plan["protocol"]["inputs"]["backend_lock"]["sha256"], "actual backend lock")
    files[str(MODEL_SOURCE.resolve())] = sha(MODEL_SOURCE)
    files[str(ROOT / "backend.lock.json")] = sha(ROOT / "backend.lock.json")
    device = torch.device("cuda:0")
    require(torch.cuda.is_available(), "post-run full BF16 capture requires GPU0")
    torch.cuda.set_device(device)
    torch.set_num_threads(8)
    original_rng = rng_witness(device)
    args.output.mkdir(parents=True)
    checks, calls, raw = {}, dict(student=0, live_teacher=0, critic=0), {}
    def check(label, value):
        checks[label] = bool(value)
        require(value, label)
    def budget():
        return check_budget(started, card["execution_budget_seconds"])
    def unchanged_files():
        for path, digest in files.items():
            budget()
            check("immutable_file:" + path, sha(path) == digest)
    try:
        budget()
        data_before = state_digest(data)
        check("qualified_dataset_identity", data_before == parent_plan["protocol"]["data"]["digest"])
        for pool in POOLS:
            check("all_contexts:" + pool, len(data[pool]["context"]) == COUNTS[pool]
                and data[pool]["context"].shape[1] == 4100)
        for pool in ("fit", "test"):
            check("all_six_sources:" + pool, tuple(data[pool]["context"][:, 4097].long().unique(sorted=True).tolist()) == SUBJECTS)
        check("qualified_all_pool_record_reductions", True)  # Already checked before CUDA/model/checkpoint loading.
        panels, panel_hashes, panel_stream_digest = panel_batches(data,
            parent_plan["protocol"]["evaluation"]["gaussian_panels"]["unscaled_gaussian_sha256_by_pool"])
        panels_before = state_digest(panels)
        saved = {name: torch.load(Path(parent_plan["input_paths"][name]), map_location="cpu", weights_only=False, mmap=True)
                 for name in JUDGES}
        saved_before = {name: state_digest(dict(critic=s["policy"]["models"]["critic"],
            native=s["policy"]["streams"], data=s["data_rng"], paired=s["paired_noise_rng"])) for name, s in saved.items()}
        for name, s in saved.items():
            check("saved_judge_law:" + name, s["policy"]["completed_steps"] == int(name[1:])
                and s["config"]["dataset_digest"] == data_before
                and s["config"]["architecture"] == "linear_modulated_v2"
                and float(s["policy"]["last_output_sigma"]) == SIGMA)
            check("saved_judge_tensors:" + name, state_digest(s["policy"]["models"]["critic"]) == card["critic_tensor_digests"][name])
            check("saved_text_ownership:" + name, torch.equal(s["policy"]["models"]["encoder"]["contexts"], data["text_contexts"])
                and torch.equal(s["policy"]["models"]["encoder"]["masks"], data["text_masks"]))
        teacher_state = {key.removeprefix("teacher."): value for key, value in saved["D6400"]["policy"]["models"]["encoder"].items() if key.startswith("teacher.")}
        teacher_before = state_digest(teacher_state)
        with torch.random.fork_rng(devices=[0]), torch.device("meta"):
            backend = model_module()
            base = backend.SupraDiT()
            backend.attach_supra_lora(base, rank=16, alpha=16, targets=TARGETS)
        base.load_state_dict(teacher_state, strict=True, assign=True)
        base.to(device).eval().requires_grad_(False)
        check("pure_base_zero_ordinary_up", all(not value.count_nonzero() for key, value in teacher_state.items() if key.endswith(".up.weight")))
        encoder = FrozenSliderContexts(data["text_contexts"], data["text_masks"], base, cfg=3.).to(device).eval().requires_grad_(False)
        base_before, encoder_before = module_witness(base), module_witness(encoder)
        plan = dict(schema="supra_endpoint_game_rays_v1", card=card, card_sha256=sha(CARD),
            artifacts=artifacts, input_and_source_sha256=files, critic_tensor_digests=card["critic_tensor_digests"],
            unscaled_panel_sha256_by_pool=panel_hashes, panel_stream_digest=panel_stream_digest,
            torch_version=torch.__version__, device=str(device), gpu=torch.cuda.get_device_name(device),
            global_rng_digest=original_rng, preflight_cpu_rng_digest=state_digest(initial_cpu_rng),
            alpha1_match_tolerance=card["alpha1_match_tolerance"],
            target_pairing="Live FrozenSliderContexts target-caption BF16 CFG3 on identical B4 batch; no cached velocities.")
        write(args.output / "plan.json", plan)
        for label, artifact in artifacts.items():
            budget()
            with torch.random.fork_rng(devices=[0]):
                if label == "ordinary_lora_6400":
                    model = deepcopy(base).eval().requires_grad_(False)
                    load_adapter_state(model, load_file(artifact["path"], device="cpu"))
                else:
                    arm, step = label.split("@")
                    with safe_open(artifact["path"], framework="pt", device="cpu") as handle:
                        metadata = handle.metadata()
                    config = json.loads(metadata["config"])
                    check("exact_export_metadata:" + label, config == review["export_artifacts"][label]["config"]
                        and config["served_source"] == "fast" and config["completed_steps"] == int(step)
                        and config["cfg"] == 3 and config["rank"] == 16 and config["z_dim"] == 4 and config["num_particles"] == 128
                        and config["architecture"] == "gated_particle_v3" and len(config["sites"]) == 71
                        and json.loads(metadata["native_pins"]) == native_pins())
                    model = load_particle_adapter(base, artifact["path"], device=device)
            before = module_witness(model)
            captured = {}
            for pool in POOLS:
                residuals, conditions = [], []
                for start in range(0, COUNTS[pool], BATCH):
                    budget()
                    context = data[pool]["context"][start:start + BATCH].to(device)
                    with torch.no_grad():
                        z, t, ctx, mask, uctx, umask, strength = encoder.unpack(context)
                        if label == "ordinary_lora_6400":
                            ordinary = [m for m in model.modules() if hasattr(m, "multiplier")]
                            previous = [m.multiplier for m in ordinary]
                            try:
                                for m in ordinary:
                                    m.multiplier = strength
                                with torch.autocast("cuda", dtype=torch.bfloat16):
                                    velocity = model(torch.cat((z, z)), torch.cat((t, t)),
                                        torch.cat((ctx, uctx.expand(len(z), -1, -1))), torch.cat((mask, umask.expand(len(z), -1)))).float()
                                positive, negative = velocity.chunk(2)
                                velocity = negative + 3 * (positive - negative)
                            finally:
                                for m, previous_value in zip(ordinary, previous):
                                    m.multiplier = previous_value
                        else:
                            velocity = model.velocity(z, t, ctx, mask, uctx, umask, strength=strength)
                        residual = velocity - encoder.teacher_velocity(context)
                        calls["student"] += 1
                        calls["live_teacher"] += 1
                        check("finite_residual:" + label + ":" + pool + ":" + str(start), bool(torch.isfinite(residual).all()))
                        residuals.append(residual.cpu())
                        conditions.append(encoder.condition(context).cpu())
                    if (start // BATCH + 1) % 20 == 0:
                        emit(phase="capture_progress", artifact=label, pool=pool,
                             contexts=min(start + BATCH, COUNTS[pool]), seconds=time.monotonic() - started)
                captured[pool] = dict(physical_residual=torch.cat(residuals), condition=torch.cat(conditions))
                emit(phase="capture", artifact=label, pool=pool, contexts=COUNTS[pool], seconds=time.monotonic() - started)
            check("model_immutable:" + label, module_witness(model) == before)
            check("base_encoder_immutable:" + label, module_witness(base) == base_before and module_witness(encoder) == encoder_before)
            check("global_rng_immutable:" + label, rng_witness(device) == original_rng)
            raw[label] = captured
            del model
            gc.collect()
        check("teacher_input_tensors_immutable", state_digest(teacher_state) == teacher_before)
        for pool in POOLS:
            first = raw[next(iter(raw))][pool]["condition"]
            check("same_condition_all_artifacts:" + pool, all(torch.equal(item[pool]["condition"], first) for item in raw.values()))
        # Physical residuals are retained before any scoring, with immutable exact panel law.
        raw_path = args.output / "residuals.pt"
        torch.save(dict(plan=plan, captures=raw, unscaled_panel_batches=panels,
            context_metadata={pool: data[pool]["context"][:, 4096:].clone() for pool in POOLS}), raw_path)
        raw_digest, raw_memory_digest = sha(raw_path), state_digest(raw)
        del base, encoder, teacher_state
        gc.collect()
        results, per_context = {}, {}
        for name in JUDGES:
            with torch.random.fork_rng(devices=[0]):
                judge = ConditionalTokenCritic(data["coordinate_scale"]).float().to(device).eval().requires_grad_(False)
            judge.load_state_dict(saved[name]["policy"]["models"]["critic"], strict=True)
            before = module_witness(judge)
            check("loaded_judge_identity:" + name, state_digest(judge.state_dict()) == card["critic_tensor_digests"][name])
            results[name], per_context[name] = {}, {}
            for label, pools in raw.items():
                results[name][label], per_context[name][label] = {}, {}
                original = original_evaluations[label]
                for pool, captured in pools.items():
                    games, derivatives = {f"{x:g}": [] for x in ALPHAS}, {"0": [], "1": []}
                    for batch_index, start in enumerate(range(0, COUNTS[pool], BATCH)):
                        budget()
                        error = patchify(captured["physical_residual"][start:start+BATCH].to(device)).float() / judge.scale
                        condition = captured["condition"][start:start+BATCH].to(device)
                        panel = SIGMA * panels[pool][batch_index].to(device)
                        batch_games, batch_derivatives = score_ray(judge, error, condition, panel)
                        calls["critic"] += 1 + len(ALPHAS)
                        for key, value in batch_games.items():
                            games[key].append(value)
                        for key, value in batch_derivatives.items():
                            derivatives[key].append(value)
                    games = {key: torch.cat(values) for key, values in games.items()}
                    derivatives = {key: torch.cat(values) for key, values in derivatives.items()}
                    expected = torch.tensor([record[name] for record in original[pool]["records"]])
                    tolerance = card["alpha1_match_tolerance"]
                    torch.testing.assert_close(games["1"], expected, rtol=tolerance["rtol"], atol=tolerance["atol"])
                    check("alpha1_original_records:" + name + ":" + label + ":" + pool, True)
                    subjects = data[pool]["context"][:, 4097].long()
                    results[name][label][pool] = summarize_ray(games, derivatives, subjects, card["comparison_tolerance"])
                    results[name][label][pool]["alpha1_original_max_absolute_difference"] = float((games["1"] - expected).abs().max())
                    per_context[name][label][pool] = dict(games={key: value.tolist() for key, value in games.items()},
                        radial_derivatives={key: value.tolist() for key, value in derivatives.items()})
                    emit(phase="score", judge=name, artifact=label, pool=pool, seconds=time.monotonic() - started)
            check("judge_immutable:" + name, module_witness(judge) == before)
            del judge
            gc.collect()
        check("global_cpu_cuda_rng_immutable", rng_witness(device) == original_rng)
        check("data_panels_residuals_immutable", state_digest(data) == data_before and state_digest(panels) == panels_before and state_digest(raw) == raw_memory_digest)
        for name, s in saved.items():
            check("saved_judge_streams_immutable:" + name, state_digest(dict(critic=s["policy"]["models"]["critic"],
                native=s["policy"]["streams"], data=s["data_rng"], paired=s["paired_noise_rng"])) == saved_before[name])
        check("raw_residual_file_immutable", sha(raw_path) == raw_digest)
        unchanged_files()
        write(args.output / "rays.json", per_context)
        budget()
        final_seconds = final_report(args.output, dict(schema="supra_endpoint_game_rays_report_v1", qualified=True,
            plan=plan, checks=checks, results=results, raw_residual_sha256=raw_digest,
            per_context_rays_sha256=sha(args.output / "rays.json"),
            forward_calls=calls, training_updates=0, output_metrics_used_for_optimizer_or_selection=False,
            existing_output_diagnostics_validated_only_for_provenance=True, selection=False,
            qualification_credit="observational endpoint judge-shape only", limits=card["limits"]),
            started, card["execution_budget_seconds"])
        emit(phase="complete", seconds=final_seconds, updates=0)
    except Exception as error:
        write(args.output / "failure.json", dict(error=type(error).__name__ + ": " + str(error), checks=checks,
            seconds=time.monotonic() - started, forward_calls=calls, training_updates=0, qualified=False))
        raise


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Full-state precision continuation qualification; root executes once on GPU0.

Six native updates only: each arm's 12800->12801->12802 reference, followed by
its independent saved12801->12802 recovery. Offline TEST anchors and clean
exports qualify plumbing; no accuracy value enters the optimizer or controls.
--preflight-only checks source/input bytes using stdlib without importing Torch.
"""
import time
STARTED = time.monotonic()
import argparse
from copy import deepcopy
import gc
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
TASK = "supra_final_precision_12800_recovery_v1"
LIMIT = 300
START = 12800
MODES = ("BF16", "FP32_final")
ANCHORS = {"BF16": "neutral12800_BF16", "FP32_final": "neutral12800_FP32_final"}
JUDGES = ("D1856", "D6400")


def budget():
    if time.monotonic() - STARTED > LIMIT:
        raise TimeoutError("qualification GPU300 startup-through-rollback/final-writes cap")


def sha(path, *, charged=True):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 << 20):
            value.update(block)
            if charged: budget()
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def bindings(plan):
    result = {}
    for item in (*plan["sources"].values(), *plan["inputs"].values(), *plan["dependencies"].values()):
        name = str(Path(item["path"]).resolve())
        if name not in result: result[name] = sha(name)
        if result[name] != item["sha256"]:
            raise ValueError("bound source/input bytes changed: " + name)
    return result


def preflight(plan):
    if (plan["id"] != TASK or plan["seconds"] != LIMIT or plan["start_step"] != START
            or plan["end_step"] != START + 2 or plan["native_updates"] != 6
            or plan["optimizer_steps"] != 12 or plan["arms"] != list(MODES)
            or plan["training"] != dict(batch_size=4, pool="FIT", explicit_CPU7_indices=True,
                targets="zero physical residual", game_weight=1., preservation_updates=0,
                output_objective=False, output_error_guard=False, optimizer_intervention=False)):
        raise ValueError("fixed recovery task/update law differs")
    bound = bindings(plan)
    inp = plan["inputs"]
    producer = read(inp["scorer_report"]["path"]); companion = read(inp["scorer_completion"]["path"])
    review = read(inp["scorer_review"]["path"]); review_completion = read(inp["scorer_review_completion"]["path"])
    if not (producer["complete"] is True and companion["complete"] is True
            and companion["report_sha256"] == inp["scorer_report"]["sha256"]
            and producer["observations_sha256"] == companion["observations_sha256"] == inp["scorer_raw"]["sha256"]
            and review["complete"] is True and review["qualification"] == "PASS"
            and review_completion["complete"] is True and review_completion["qualification"] == "PASS"
            and review_completion["report_sha256"] == inp["scorer_review"]["sha256"]
            and inp["scorer_raw"]["sha256"] in review["input_sha256"].values()):
        raise ValueError("initial scorer anchors are not independently qualified/authenticated")
    software = read(inp["software_external"]["path"])
    if not (plan["software_qualified"] is True and software["complete"] is True
            and software["qualification"] == "PASS" and software["exit_code"] == 0
            and software["timed_out"] is False and software["seconds"] <= software["limit_seconds"] == 60
            and software["cases"] == dict(tests=10, failures=0, errors=0, skipped=0)
            and software["native_updates"] == 3 and software["optimizer_steps"] == 6
            and software["actual_asset_loads"] == 0 and software["CUDA"] is False
            and len(software["checks_before"]) == len(software["checks_after"]) == 43
            and all(v is True for v in software["checks_before"].values())
            and all(v is True for v in software["checks_after"].values())
            and software["stdout_sha256"] == inp["software_stdout"]["sha256"]
            and software["junit_sha256"] == inp["software_junit"]["sha256"]
            and software["plan_sha256"] == inp["software_plan"]["sha256"]
            and software["launcher_sha256"] == inp["software_launcher"]["sha256"]):
        raise ValueError("ten-case NEW software prerequisite is not qualified or bound")
    return bound


def exact(left, right, torch, path="state", counts=None):
    """Compare every tensor byte/device and scalar, including intentional NaNs."""
    if counts is None: counts = dict(tensors=0, scalars=0)
    if isinstance(left, torch.Tensor):
        if (not isinstance(right, torch.Tensor) or left.dtype != right.dtype
                or left.shape != right.shape or left.device != right.device
                or not torch.equal(left.detach().contiguous().reshape(-1).view(torch.uint8),
                                   right.detach().contiguous().reshape(-1).view(torch.uint8))):
            raise AssertionError("tensor byte/device mismatch: " + path)
        counts["tensors"] += 1
    elif isinstance(left, dict):
        if not isinstance(right, dict) or left.keys() != right.keys():
            raise AssertionError("mapping mismatch: " + path)
        for key in left: exact(left[key], right[key], torch, path + "." + str(key), counts)
    elif isinstance(left, (tuple, list)):
        if type(left) is not type(right) or len(left) != len(right):
            raise AssertionError("sequence mismatch: " + path)
        for index, (a, b) in enumerate(zip(left, right)):
            exact(a, b, torch, path + "[" + str(index) + "]", counts)
    else:
        same = type(left) is type(right)
        if same:
            same = (struct.pack("!d", left) == struct.pack("!d", right)
                    if isinstance(left, float) else left == right)
        if not same: raise AssertionError("scalar mismatch: " + path)
        counts["scalars"] += 1
    budget()
    return counts


def backend(torch):
    return dict(float32_matmul_precision=torch.get_float32_matmul_precision(),
                cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                allow_bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)


def stage(name, **values):
    print(json.dumps(dict(stage=name, seconds=time.monotonic()-STARTED, **values), allow_nan=False), flush=True)


def run(plan, out, torch):
    sys.path.insert(0, str(Path(plan["particlegan_root"]).resolve()))
    sys.path.insert(1, str(ROOT))
    import particlegan
    from supra import particle_final_precision as precision
    from supra import particle_final_precision_export as export
    from supra.particle_final_precision_evaluation import evaluate_test
    from supra.particle_pilot import state_digest
    from supra.particle_final_precision_task import load_task, fresh_restored, editing_update
    if Path(particlegan.__file__).resolve().parent.parent != Path(plan["particlegan_root"]).resolve():
        raise ValueError("wrong imported ParticleGAN package")
    modes = {"BF16": precision.BF16_FINAL, "FP32_final": precision.FP32_FINAL}
    inp = plan["inputs"]; device = torch.device("cuda:0")
    data, legacy, base, judges = load_task(plan)
    base_digest = state_digest(base.state_dict()); data_digest = state_digest(data)
    judge_digests = {name: state_digest(judge.state_dict()) for name, judge in judges.items()}
    anchor = torch.load(inp["scorer_raw"]["path"], map_location="cpu", weights_only=False, mmap=True)
    raw = dict(task=TASK, initial_captures={}, recovery={}, export={}, draw_traces={},
               measurement_finished=False, native_updates=0, optimizer_steps=0)
    loops = {}; initial = {}; receipts = {}; reference = {}; reference_rows = {}

    def classes(loop):
        p = loop.policy
        owners = dict(generator=p.G, critic=p.D, encoder=p.encoder, router=p.router,
                      ema_generator=p.ema_G, ema_critic=p.opt_d.ema_critic,
                      ema_encoder=p.ema_encoder, ema_router=p.ema_router)
        return {role: {name: type(m).__module__ + "." + type(m).__qualname__
                       for name, m in owner.named_modules()} for role, owner in owners.items()}

    def finite_learned(loop):
        p = loop.policy
        values = [*p.G.parameters(), *p.D.parameters(), *p.encoder.parameters(),
                  *p.router.parameters(), p.table, p.log_output_sigma]
        for value in values:
            if not bool(torch.isfinite(value).all()) or (value.grad is not None and not bool(torch.isfinite(value.grad).all())):
                raise FloatingPointError("nonfinite learned/frozen Parameter or actual gradient")
        for optimizer in (p.opt_g, p.opt_d):
            for state in optimizer.state.values():
                for key, value in state.items():
                    if isinstance(value, torch.Tensor) and not bool(torch.isfinite(value).all()):
                        raise FloatingPointError("nonfinite native optimizer tensor: " + str(key))

    def fresh(arm):
        loop, proof = fresh_restored(base, data, legacy, modes[arm], plan["neutral_native_digest"])
        precision.assert_policy_precision(loop.policy, modes[arm])
        if loop.policy.completed_steps != START: raise AssertionError("bootstrap clock differs")
        return loop, proof

    def saved_state(loop):
        value = precision.checkpoint_precision(loop)
        precision.assert_policy_precision(loop.policy, value["precision"])
        return value

    def native_step(loop, arm):
        trace = dict(data_before=state_digest(loop.data_rng.get_state()),
                     paired_before=state_digest(loop.paired_noise_rng.get_state()),
                     DV12_before=state_digest(loop.policy.noise_generator.get_state()))
        row = editing_update(loop)
        finite_learned(loop)
        trace.update(batch_indices=row["batch_indices"], data_after=state_digest(loop.data_rng.get_state()),
                     paired_after=state_digest(loop.paired_noise_rng.get_state()),
                     DV12_after=state_digest(loop.policy.noise_generator.get_state()),
                     base_noise_sums=row["base_noise_sums"])
        raw["native_updates"] += 1; raw["optimizer_steps"] += 2
        stage("native_recovery", arm=arm, step=row["step"], loss_g=row["loss_g"], loss_d_game=row["loss_d_game"])
        budget()
        return row, trace

    def raw_exports(loop, arm):
        origin = saved_state(loop); before = state_digest(origin); result = {}
        with precision.precision_rng_scope(loop):
            p = loop.policy; context = data["fit"]["context"][:4].to(device)
            for source in ("fast", "averaged"):
                served = export.precision_snapshot(loop, source=source)
                models, table = served.models, served.table
                routing = served.routing
                path = out / (arm + "-" + source + ".safetensors")
                receipt = export.export_served_adapter(served, path, precision=modes[arm])
                with torch.random.fork_rng(devices=[0]):
                    adapter = export.load_particle_adapter(base, path, expected_precision=modes[arm], source=source, device=device)
                other = modes["FP32_final" if arm == "BF16" else "BF16"]
                try:
                    export.load_particle_adapter(base, path, expected_precision=other, source=source, device=device)
                except ValueError as exc:
                    if not any(s in str(exc).lower() for s in ("precision", "arithmetic", "mode")): raise
                else: raise AssertionError("wrong-precision export was accepted")
                for strength in (0., 1.):
                    current = context.clone(); current[:, 4099] = strength
                    candidate = routing.candidate_for(models, table, averaged=source == "averaged")
                    with torch.no_grad(): expected = routing.forward(models, current, candidate)
                    z, t, text, mask, empty, emask, _ = served.encoder.unpack(current)
                    actual = adapter.velocity(z, t, text, mask, empty, emask, strength=strength)
                    exact(expected, actual, torch, arm + "." + source + ".raw_strength" + str(strength))
                    raw["export"][arm + "/" + source + "/" + str(strength)] = actual.cpu()
                result[source] = dict(raw_strength0_exact=True, raw_strength1_exact=True,
                    wrong_precision_rejected=True, file_sha256=sha(path), format_receipt=receipt)
                del adapter; gc.collect(); budget()
        if state_digest(saved_state(loop)) != before:
            raise AssertionError("exports changed native/caller modes, flags, gradients or streams")
        return result

    try:
        for arm in MODES:
            loop, receipts[arm] = fresh(arm)
            initial[arm] = saved_state(loop); loops[arm] = loop; reference_rows[arm] = []
            with precision.precision_rng_scope(loops[arm]):
                before = saved_state(loops[arm])
                value = evaluate_test(loops[arm], data, judges, precision=modes[arm],
                                      panel_sha=plan["TEST_panel_sha256"], budget=budget)
                old = anchor["captures"][ANCHORS[arm]]
                for name in ("physical_residual", "mse_by_context", "paired_game_by_panel",
                             "paired_game_by_context", "records", "summary", "gaussian_panels", "TEST_panel_sha256"):
                    exact(value[name], old[name], torch, "initial_scorer." + arm + "." + name)
                exact(before, saved_state(loops[arm]), torch, "initial_scorer.native_caller_immutable." + arm)
            raw["initial_captures"][arm] = value
            stage("initial_scorer_exact", arm=arm, contexts=240)
        # Every operation uses the arm's owned global state. Adaptive controller
        # values may differ across arithmetic arms; only actual draw addresses
        # and sampled indices are compared across arms.
        for step in (START + 1, START + 2):
            for arm in MODES:
                row, draws = native_step(loops[arm], arm)
                reference_rows[arm].append(row); raw["draw_traces"].setdefault(arm, []).append(draws)
                if row["step"] != step: raise AssertionError("wrong native recovery clock")
                if step == START + 1:
                    torch.save(saved_state(loops[arm]), out / (arm + "-12801.pt"))
            exact(raw["draw_traces"][MODES[0]][-1], raw["draw_traces"][MODES[1]][-1], torch,
                  "cross_arm_sample_and_private_draw_trace.step" + str(step))
        for arm in MODES:
            reference[arm] = saved_state(loops[arm])
            raw["recovery"][arm] = dict(bootstrap=receipts[arm], reference_rows=reference_rows[arm],
                                       reference_sha256=state_digest(reference[arm]), classes=classes(loops[arm]))
        # Free reference owner before constructing the independent resumed owner.
        for arm in MODES:
            del loops[arm]; gc.collect(); torch.cuda.empty_cache()
            resumed, proof = fresh(arm); loops[arm] = resumed
            midpoint = torch.load(out / (arm + "-12801.pt"), map_location="cpu", weights_only=False)
            precision.restore_precision(resumed, midpoint)
            row, draws = native_step(resumed, arm)
            counts = exact(reference[arm], saved_state(resumed), torch, "recovered_full_state." + arm)
            exact(reference_rows[arm][-1], row, torch, "recovered_native_row." + arm)
            exact(raw["draw_traces"][arm][-1], draws, torch, "recovered_private_draw_trace." + arm)
            exact(raw["recovery"][arm]["classes"], classes(resumed), torch, "recovered_FAST_EMA_teacher_classes." + arm)
            raw["recovery"][arm].update(recovered_sha256=state_digest(saved_state(resumed)),
                exact_leaf_counts=counts, recovered_row=row, independent_fresh_owner=True,
                midpoint_file_sha256=sha(out / (arm + "-12801.pt")))
            raw["recovery"][arm]["exports"] = raw_exports(resumed, arm)
        # Strict precision rejection must not alter the rejecting live owner.
        rejecting = loops["BF16"]; before = saved_state(rejecting)
        try: precision.restore_precision(rejecting, reference["FP32_final"])
        except ValueError as exc:
            if not any(s in str(exc).lower() for s in ("precision", "arithmetic", "mode")): raise
        else: raise AssertionError("wrong-precision recovery was accepted")
        exact(before, saved_state(rejecting), torch, "wrong_precision_rejection_immutable")
        if raw["native_updates"] != 6 or raw["optimizer_steps"] != 12:
            raise AssertionError("qualification exceeded fixed six-update cost")
        raw["measurement_finished"] = True
    finally:
        restored = {}
        for arm, loop in loops.items():
            precision.restore_precision(loop, initial[arm])
            restored[arm] = state_digest(saved_state(loop)) == state_digest(initial[arm])
        raw["original12800_owners_restored"] = restored
        raw["base_teacher_and_data_unchanged"] = state_digest(base.state_dict()) == base_digest and state_digest(data) == data_digest
        raw["fixed_judges_unchanged"] = all(state_digest(v.state_dict()) == judge_digests[k] for k, v in judges.items())
        torch.save(raw, out / "recovery-observations.pt")
        if not all(restored.values()) or not raw["base_teacher_and_data_unchanged"] or not raw["fixed_judges_unchanged"]:
            raise AssertionError("qualification rollback or frozen teacher/data/judges differ")
    return dict(task=TASK, complete=True, qualification="PASS", scientific_status="PASS",
        bootstrap_proofs=receipts, recovery=raw["recovery"],
        initial_scorer_anchors_exact={arm: ANCHORS[arm] for arm in MODES},
        wrong_precision_recovery_rejected=True, caller_native_original12800_rollback_exact=True,
        counts=dict(native_updates=6, optimizer_steps=12, mandatory_native_student_forwards=12,
            mandatory_native_teacher_forwards=12, initial_TEST_student_forwards=120,
            initial_TEST_teacher_forwards=120, export_raw_student_forwards=16,
            extra_native_guard_or_KA2_forwards="native controls only; not replaced or bypassed"),
        observed_sha256=sha(out / "recovery-observations.pt"), scope=plan["scope"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(); out = None; torch = None; entry = None; report = None; error = None; code = 2
    try:
        plan = read(args.card); card_sha = sha(args.card); bound = preflight(plan)
        if args.preflight_only:
            stage("stdlib_source_input_preflight_PASS", task=TASK, native_updates=0, model_forwards=0,
                  Torch_imports=0, API_calls=0, card_sha256=card_sha, bindings=len(bound))
            return 0
        if args.out is None or args.out.resolve() != Path(plan["output"]).resolve() or args.out.exists():
            raise ValueError("one declared fresh output required")
        args.out.mkdir(parents=True, exist_ok=False); out = args.out
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "0": raise ValueError("physical GPU0 required")
        import torch as torch_module
        torch = torch_module
        if not torch.cuda.is_available(): raise ValueError("CUDA required for full BF16 qualification")
        torch.cuda.set_device(0)
        entry = dict(cpu=torch.get_rng_state().clone(), cuda=torch.cuda.get_rng_state(0).clone(),
                     threads=torch.get_num_threads(), backend=backend(torch))
        if entry["backend"] != plan["precision_backend"]: raise ValueError("held precision backend differs")
        torch.set_num_threads(8)
        report = run(plan, out, torch)
        if bindings(plan) != bound or sha(args.card) != card_sha:
            raise AssertionError("source/input/card bindings changed")
        report.update(protocol_sha256=card_sha, source_identity=plan["sources"], actual_bindings=bound)
        code = 0
    except BaseException as exc:
        error = dict(type=type(exc).__name__, message=str(exc)); traceback.print_exc()
    finally:
        if entry is not None:
            torch.set_num_threads(entry["threads"]); torch.set_rng_state(entry["cpu"]); torch.cuda.set_rng_state(entry["cuda"], 0)
            if (not torch.equal(torch.get_rng_state(), entry["cpu"])
                    or not torch.equal(torch.cuda.get_rng_state(0), entry["cuda"])
                    or backend(torch) != entry["backend"]):
                error = dict(type="AssertionError", message="entry caller RNG/backend not restored")
        if time.monotonic()-STARTED > LIMIT: error = dict(type="TimeoutError", message="cleanup300s overrun")
        if error is not None: report = dict(task=TASK, complete=False, qualification="INCOMPLETE", scientific_status=None, error=error); code = 2
        if out is not None:
            report.update(seconds=time.monotonic()-STARTED, limit_seconds=LIMIT)
            write(out / "report.json", report)
            companion = dict(task=TASK, complete=report["complete"], qualification=report["qualification"],
                scientific_status=report.get("scientific_status"), report_sha256=sha(out / "report.json", charged=False),
                protocol_sha256=card_sha, seconds=time.monotonic()-STARTED, limit_seconds=LIMIT, exit_code=code)
            write(out / "completion.json", companion)
            if time.monotonic()-STARTED > LIMIT:
                companion.update(complete=False, qualification="INCOMPLETE", scientific_status=None,
                                 error="post-write300s overrun", exit_code=2)
                (out / "completion.json").write_text(json.dumps(companion, indent=2, allow_nan=False)+"\n"); code = 2
    return code


if __name__ == "__main__": raise SystemExit(main())

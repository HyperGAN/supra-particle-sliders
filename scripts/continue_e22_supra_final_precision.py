#!/usr/bin/env python3
"""Matched 12800->19200 native editing-only particle continuations.

Fixed diagnostics and terminal horizon. Both trained checkpoints are scored
under both arithmetic modes; metrics never feed native updates or selection.
"""
import time
STARTED = time.monotonic()
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
TASK = "supra_final_precision_matched_19200_v1"
START = 12800
END = 19200
ENDPOINTS = (16000, END)
LIMIT = 14400
ARM_LIMIT = 7200
ARMS = ("BF16", "FP32_final")
METRICS = ("rmse_diagnostic", "D1856", "D6400")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def write(path, value, *, replace=False):
    path = Path(path)
    if not replace:
        with path.open("x") as f:
            f.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
    else:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        tmp.replace(path)


def bindings(card):
    files = {}
    for group in ("sources", "dependencies", "inputs"):
        for item in card[group].values():
            p = item["path"]
            if sha(p) != item["sha256"]:
                raise ValueError("fixed source/input changed: " + p)
            files[p] = item["sha256"]
    return files


def preflight(card):
    if (card["id"] != TASK or card["seconds"] != LIMIT or card["per_arm_seconds"] != ARM_LIMIT
            or card["start_step"] != START or card["end_step"] != END
            or card["endpoints"] != list(ENDPOINTS) or card["arms"] != list(ARMS)
            or card["updates_per_arm"] != 6400 or card["native_updates"] != 12800
            or card["optimizer_steps"] != 25600
            or card["training"] != dict(batch_size=4, pool="FIT", explicit_CPU7_indices=True,
                targets="zero physical residual", game_weight=1., preservation_updates=0,
                output_objective=False, output_error_guard=False, optimizer_intervention=False)):
        raise ValueError("fixed matched continuation protocol differs")
    bound = bindings(card)
    inp = card["inputs"]
    prerequisite = json.loads(Path(inp["recovery_report"]["path"]).read_text())
    companion = json.loads(Path(inp["recovery_completion"]["path"]).read_text())
    external = json.loads(Path(inp["recovery_external"]["path"]).read_text())
    recovery_card = json.loads(Path(inp["recovery_protocol"]["path"]).read_text())
    if not (prerequisite["complete"] is True and prerequisite["qualification"] == "PASS"
            and prerequisite["scientific_status"] == "PASS"
            and companion["complete"] is True and companion["qualification"] == "PASS"
            and companion["exit_code"] == 0 and companion["scientific_status"] == "PASS"
            and companion["report_sha256"] == inp["recovery_report"]["sha256"]
            and companion["protocol_sha256"] == prerequisite["protocol_sha256"] == inp["recovery_protocol"]["sha256"]
            and prerequisite["counts"]["native_updates"] == 6
            and prerequisite["counts"]["optimizer_steps"] == 12
            and prerequisite["caller_native_original12800_rollback_exact"] is True
            and prerequisite["wrong_precision_recovery_rejected"] is True
            and prerequisite["initial_scorer_anchors_exact"] == {
                "BF16": "neutral12800_BF16", "FP32_final": "neutral12800_FP32_final"}
            and external["complete"] is True and external["exit_code"] == 0
            and external["timed_out"] is False and external["seconds"] <= external["limit_seconds"] == 300
            and external["child_completion_sha256"] == inp["recovery_completion"]["sha256"]
            and external["manifest_sha256"] == inp["recovery_manifest"]["sha256"]
            and external["launcher_sha256"] == inp["gpu_launcher"]["sha256"]
            and len(external["checks_before"]) == len(external["checks_after"]) == 65
            and all(v is True for v in external["checks_before"].values())
            and all(v is True for v in external["checks_after"].values())):
        raise ValueError("full-model recovery/scorer/export prerequisite is not qualified")
    for arm in ARMS:
        recovered = prerequisite["recovery"][arm]
        proof = prerequisite["bootstrap_proofs"][arm]
        if not (proof["original_native_digest"] == card["neutral_native_digest"]
                and proof["legacy_roundtrip_exact"] is True and proof["initialized_after_restore"] is False
                and recovered["independent_fresh_owner"] is True
                and recovered["reference_sha256"] == recovered["recovered_sha256"]):
            raise ValueError("qualified own-arm trained restoration differs")
    for name, item in prerequisite["source_identity"].items():
        if name in card["sources"] and item != card["sources"][name]:
            raise ValueError("qualified source differs: " + name)
    for name, item in recovery_card["dependencies"].items():
        if card["dependencies"].get(name) != item:
            raise ValueError("qualified native/backend dependency differs: " + name)
    for key in ("backend_sha256", "data_digest", "neutral_native_digest", "critic_tensor_digests", "TEST_panel_sha256", "precision_backend"):
        if card[key] != recovery_card[key]:
            raise ValueError("qualified full-task numerical law differs: " + key)
    scorer = json.loads(Path(inp["scorer_report"]["path"]).read_text())
    for key, label in (("frozen_original", "original28000_BF16"),
                       ("frozen_original_FP32", "original28000_FP32_final")):
        if card[key] != {metric: scorer["summaries"][label][metric] for metric in METRICS}:
            raise ValueError("qualified frozen original metrics differ")
    return bound


def run(card, out):
    sys.path.insert(0, card["particlegan_root"])
    sys.path.insert(1, str(ROOT))
    import torch
    import particlegan
    from supra.particle_pilot import state_digest
    from supra.particle_final_precision import BF16_FINAL, FP32_FINAL, checkpoint_precision
    from supra.particle_final_precision_task import load_task, fresh_restored, editing_update
    from supra.particle_final_precision_evaluation import evaluate_test
    from supra.particle_final_precision_export import export_particle_adapter
    if Path(particlegan.__file__).resolve().parent.parent != Path(card["particlegan_root"]).resolve():
        raise ValueError("wrong imported ParticleGAN")
    if shutil.disk_usage(out).free < 128 * 1024 ** 3:
        raise ValueError("declared 128GiB checkpoint disk allowance unavailable")
    torch.set_num_threads(1)
    actual_backend = dict(float32_matmul_precision=torch.get_float32_matmul_precision(),
                         cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                         cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                         allow_bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
    if actual_backend != card["precision_backend"]:
        raise ValueError("qualified Torch arithmetic backend differs")
    data, parent, base, judges = load_task(card)
    modes = {"BF16": BF16_FINAL, "FP32_final": FP32_FINAL}
    loops, proofs, streams = {}, {}, {}
    charged = {arm: 0. for arm in ARMS}
    histories = {arm: [] for arm in ARMS}
    coverage = {arm: dict(bank_live_updates=0, dense_bank_updates=0, moved_rows=0) for arm in ARMS}
    draw_comparison = dict(DV12_equal_steps=0, DV12_different_steps=0)
    results = {}
    initial_bank, initial_router, frozen = {}, {}, {}
    for arm in ARMS:
        loops[arm], proofs[arm] = fresh_restored(base, data, parent, modes[arm], card["neutral_native_digest"])
        (out / arm).mkdir()
        streams[arm] = (out / arm / "train.jsonl").open("x", buffering=1)
        p = loops[arm].policy
        initial_bank[arm] = state_digest(p.table)
        initial_router[arm] = state_digest(p.router.state_dict())
        frozen[arm] = state_digest({n: v for n, v in p.G.named_parameters() if not v.requires_grad})
    preparation = time.monotonic() - STARTED
    for arm in ARMS:
        charged[arm] = preparation / 2
    native_updates = 0
    active = dict(arm=None, tick=None)

    def budget():
        now = time.monotonic()
        effective = dict(charged)
        if active["arm"] is not None:
            effective[active["arm"]] += now - active["tick"]
        if now - STARTED > LIMIT or any(v > ARM_LIMIT for v in effective.values()):
            raise TimeoutError("fixed total/per-arm continuation budget exhausted")

    def status(step, phase):
        rolling = {}
        for arm in ARMS:
            tail = histories[arm][-100:]
            rolling[arm] = dict(charged_seconds=charged[arm], coverage=coverage[arm],
                rolling_count=len(tail), **{
                    k: None if not tail else sum(r[k] for r in tail) / len(tail)
                    for k in ("loss_g", "loss_d_game", "penalty", "output_sigma", "bank_grad_norm")})
        value = dict(task=TASK, phase=phase, step=step, start_step=START, end_step=END,
                     native_updates=native_updates, preservation_updates=0,
                     seconds=time.monotonic() - STARTED, arms=rolling,
                     fixed_original=card["frozen_original"], endpoints=results, draw_comparison=draw_comparison,
                     meaning="Native game losses; offline endpoint scores never select updates/checkpoints.")
        write(out / "status.json", value, replace=True)
        print(json.dumps(value, allow_nan=False), flush=True)

    def save(arm, step):
        path = out / arm / f"checkpoint-{step:05d}.pt"
        if path.exists():
            raise ValueError("exclusive checkpoint exists")
        temporary = path.with_suffix(".tmp")
        torch.save(checkpoint_precision(loops[arm]), temporary)
        temporary.replace(path)

    def finite_learned(arm):
        p = loops[arm].policy
        count = dict(parameters=0, gradients=0, moment_tensors=0)
        def moments(value):
            if isinstance(value, torch.Tensor):
                if not bool(torch.isfinite(value).all()):
                    raise FloatingPointError("nonfinite optimizer moment")
                count["moment_tensors"] += 1
            elif isinstance(value, dict):
                for child in value.values():
                    moments(child)
            elif isinstance(value, (tuple, list)):
                for child in value:
                    moments(child)
        for optimizer in (p.opt_g, p.opt_d):
            for group in optimizer.param_groups:
                for parameter in group["params"]:
                    if not bool(torch.isfinite(parameter).all()):
                        raise FloatingPointError("nonfinite learned parameter")
                    count["parameters"] += 1
                    if parameter.grad is not None:
                        if not bool(torch.isfinite(parameter.grad).all()):
                            raise FloatingPointError("nonfinite learned gradient")
                        count["gradients"] += 1
            for state in optimizer.state.values():
                moments(state)
        return count

    def endpoint(arm, step):
        loop = loops[arm]
        finite = finite_learned(arm)
        before = state_digest(checkpoint_precision(loop))
        for eval_arm in ARMS:
            observation = evaluate_test(loop, data, judges, precision=modes[eval_arm],
                                        panel_sha=card["TEST_panel_sha256"], budget=budget)
            path = out / arm / f"evaluation-{step:05d}-{eval_arm}.pt"
            torch.save(observation, path)
            compact = dict(summary=observation["summary"], records=observation["records"],
                           raw_sha256=sha(path), raw_path=str(path),
                           training_precision=modes[arm], evaluation_precision=modes[eval_arm])
            write(path.with_suffix(".json"), compact)
            results[f"{arm}@{step}:{eval_arm}"] = compact["summary"]
        observation = evaluate_test(loop, data, judges, precision=modes[arm],
                                    panel_sha=card["TEST_panel_sha256"], budget=budget, ablate=True)
        path = out / arm / f"code-ablation-{step:05d}.pt"
        torch.save(observation, path)
        write(path.with_suffix(".json"), dict(summary=observation["summary"],
              raw_sha256=sha(path), raw_path=str(path), meaning="Offline zero mixed codes; no training intervention."))
        results[f"{arm}@{step}:zero_code"] = observation["summary"]
        export_particle_adapter(loop, out / arm / f"adapter-{step:05d}.safetensors", source="fast")
        if state_digest(checkpoint_precision(loop)) != before:
            raise AssertionError("offline endpoint evaluation/export mutated trained state")
        if state_digest({n: v for n, v in loop.policy.G.named_parameters() if not v.requires_grad}) != frozen[arm]:
            raise AssertionError("frozen student parameters changed")
        write(out / arm / f"finite-learned-{step:05d}.json", finite)

    try:
        status(START, "training")
        for step in range(START + 1, END + 1):
            paired = {}
            for arm in ARMS:
                tick = time.monotonic()
                active.update(arm=arm, tick=tick)
                row = editing_update(loops[arm])
                if row["step"] != step:
                    raise AssertionError("native continuation clock differs")
                native_updates += 1
                histories[arm].append(row)
                streams[arm].write(json.dumps(row, allow_nan=False) + "\n")
                paired[arm] = row
                coverage[arm]["bank_live_updates"] += int(row["dense_gradient_rows"] > 0)
                coverage[arm]["dense_bank_updates"] += int(row["dense_gradient_rows"] == 128)
                coverage[arm]["moved_rows"] += int((row.get("move") or {}).get("moves", 0))
                if step % 400 == 0:
                    save(arm, step)
                if step in ENDPOINTS:
                    endpoint(arm, step)
                charged[arm] += time.monotonic() - tick
                active.update(arm=None, tick=None)
                budget()
            for key in ("batch_indices", "base_noise_sums", "paired_rng_digest"):
                if paired[ARMS[0]][key] != paired[ARMS[1]][key]:
                    raise AssertionError("matched FIT/paired draw law differs: " + key)
            equal = paired[ARMS[0]]["dv12_rng_digest"] == paired[ARMS[1]]["dv12_rng_digest"]
            draw_comparison["DV12_equal_steps" if equal else "DV12_different_steps"] += 1
            if step % 25 == 0:
                status(step, "training")
            budget()
        candidate = results[f"FP32_final@{END}:FP32_final"]
        control = results[f"BF16@{END}:FP32_final"]
        code_off = results[f"FP32_final@{END}:zero_code"]
        wins = {k: candidate[k] < card["frozen_original"][k] for k in METRICS}
        best_known_wins = {k: candidate[k] < card["frozen_original_FP32"][k] for k in METRICS}
        training_benefit = {k: candidate[k] < control[k] for k in METRICS}
        particles = {}
        for arm in ARMS:
            p = loops[arm].policy
            particles[arm] = dict(bank_changed=state_digest(p.table) != initial_bank[arm],
                router_changed=state_digest(p.router.state_dict()) != initial_router[arm],
                code_norms=[float(b.bridge.weight[:, 16:].detach().norm()) for b in p.G.particle_branches()],
                H_b_trainable=all(b.bridge.weight.requires_grad and b.bridge.bias.requires_grad
                                  for b in p.G.particle_branches()))
        retained = (particles["FP32_final"]["bank_changed"] and particles["FP32_final"]["router_changed"]
                    and particles["FP32_final"]["H_b_trainable"]
                    and all(math.isfinite(v) and v > 0 for v in particles["FP32_final"]["code_norms"])
                    and code_off["rmse_diagnostic"] >= 1.001 * candidate["rmse_diagnostic"])
        report = dict(task=TASK, complete=True, scientific_status="PASS" if all(wins.values()) and all(best_known_wins.values()) and retained else "FAIL",
                      original_strict_lower=wins, known_original_common_FP32_strict_lower=best_known_wins,
                      common_FP32_training_strict_lower=training_benefit,
                      training_benefit_demonstrated=all(training_benefit.values()), particles_retained=retained,
                      particles=particles, bootstrap_proofs=proofs, endpoint_results=results,
                      coverage=coverage, draw_comparison=draw_comparison,
                      native_updates=native_updates, optimizer_steps=2*native_updates,
                      preservation_updates=0, charged_seconds=charged, seconds=time.monotonic()-STARTED,
                      limit_seconds=LIMIT, fixed_original=card["frozen_original"],
                      fixed_original_FP32=card["frozen_original_FP32"],
                      comparison_limit="Original28000 was validation-selected MSE/AdamW with preservation; target comparison is not matched optimizer superiority.")
        status(END, "complete")
        return report
    finally:
        for stream in streams.values():
            stream.close()
        loops.clear()
        gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    card_bytes = args.card.read_bytes()
    card_sha = hashlib.sha256(card_bytes).hexdigest()
    card = json.loads(card_bytes)
    before = preflight(card)
    before[str(args.card)] = card_sha
    if args.preflight_only:
        print(json.dumps(dict(preflight="PASS", bindings=len(before), models=0, API_calls=0)), flush=True)
        return 0
    args.out.mkdir()
    error = None
    report = None
    try:
        report = run(card, args.out)
        if any(sha(p) != v for p, v in before.items()):
            raise AssertionError("frozen sources/inputs changed")
        shared_final = max(0., time.monotonic() - STARTED - sum(report["charged_seconds"].values()))
        for arm in ARMS:
            report["charged_seconds"][arm] += shared_final / 2
        report["shared_final_seconds"] = shared_final
        if any(v > ARM_LIMIT for v in report["charged_seconds"].values()):
            raise TimeoutError("shared cleanup/qualification exceeded per-arm cap")
        if time.monotonic() - STARTED > LIMIT:
            raise TimeoutError("cleanup exceeded fixed continuation cap")
        code = 0 if report["scientific_status"] == "PASS" else 1
    except BaseException as exc:
        traceback.print_exc()
        error = dict(type=type(exc).__name__, message=str(exc))
        report = dict(report or {}, task=TASK, complete=False, scientific_status="INCOMPLETE", error=error)
        code = 2
    report.update(card_sha256=card_sha, input_source_bindings=before,
                  seconds=time.monotonic()-STARTED, limit_seconds=LIMIT)
    write(args.out / "report.json", report)
    companion = dict(task=TASK, complete=report["complete"],
          scientific_status=report["scientific_status"], error=error, exit_code=code,
          report_sha256=sha(args.out / "report.json"), card_sha256=card_sha,
          seconds=time.monotonic()-STARTED, limit_seconds=LIMIT)
    write(args.out / "completion.json", companion)
    elapsed = time.monotonic() - STARTED
    last_shared = max(0., elapsed - sum(report.get("charged_seconds", {}).values()))
    final_charges = {arm: report["charged_seconds"][arm] + last_shared / 2 for arm in ARMS} if "charged_seconds" in report else {}
    companion.update(final_charged_seconds=final_charges,
                     seconds=time.monotonic()-STARTED)
    write(args.out / "completion.json", companion, replace=True)
    card_unchanged = sha(args.card) == card_sha
    elapsed = time.monotonic() - STARTED
    last_shared = max(0., elapsed - sum(report.get("charged_seconds", {}).values()))
    final_charges = {arm: report["charged_seconds"][arm] + last_shared / 2 for arm in ARMS} if "charged_seconds" in report else {}
    if (elapsed > LIMIT or any(v > ARM_LIMIT for v in final_charges.values())
            or not card_unchanged):
        error = dict(type="FinalQualificationError", message="card mutation or final-write total/per-arm overrun")
        report.update(complete=False, scientific_status="INCOMPLETE", error=error,
                      seconds=elapsed, final_charged_seconds=final_charges)
        write(args.out / "report.json", report, replace=True)
        companion.update(complete=False, scientific_status="INCOMPLETE", error=error, exit_code=2,
                         report_sha256=sha(args.out / "report.json"), seconds=time.monotonic()-STARTED,
                         final_charged_seconds=final_charges)
        write(args.out / "completion.json", companion, replace=True)
        code = 2
    return code


if __name__ == "__main__":
    raise SystemExit(main())

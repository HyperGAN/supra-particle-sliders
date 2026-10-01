#!/usr/bin/env python3
"""Independent CPU reconstruction of native-step role game-score receipts."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/e22-native-step-roles-6400")
    parser.add_argument("--run", type=Path, default=ROOT / "outputs/e22-particle-v2-6400")
    args = parser.parse_args()
    torch.set_num_threads(4)
    report = json.loads((args.output / "roles.json").read_text())
    payload = torch.load(args.output / "score-panels.pt", map_location="cpu", weights_only=False)
    plan = report["plan"]
    checks = {}
    checks["completed_native_replay"] = all(report.get(key) is True for key in
        ("completed", "exact_native_rows", "exact_native_full_replay", "frozen_unchanged", "source_and_inputs_unchanged"))
    checks["reference_full_digest_equal"] = report["final_native_digest"] == report["reference_final_native_digest"]
    checks["input_hashes"] = all(sha(args.run / name) == expected for name, expected in plan["input_sha256"].items())
    checks["archived_application_hashes"] = all(sha(args.output / "source" / name) == expected
        for name, expected in plan["application_source_sha256"].items())
    checks["current_application_hashes"] = all(sha(ROOT / name) == expected
        for name, expected in plan["application_source_sha256"].items())
    pg = Path(plan["imported_particlegan"]).parent.parent
    checks["particlegan_hashes"] = all(sha(pg / name) == expected for name, expected in plan["particlegan_source_sha256"].items())
    checks["raw_plan_equal"] = payload["plan"] == plan
    checks["predetermined_horizon"] = (len(report["rows"]) == len(payload["rows"]) == plan["updates"]
        and [r["step"] for r in report["rows"]] == list(range(6401, 6401 + plan["updates"])))
    checks["all_panels_native_state_unchanged"] = all(r["native_state_unchanged_by_panels"] for r in report["rows"])
    checks["both_tasks_present"] = {r["hold"] for r in report["rows"]} == {False, True}
    largest_error, count = 0., 0
    reconstructed = []
    expected_arms = {"none", "table", "router", "router+table", "generator", "generator+table",
                     "generator+router", "generator+router+table"}
    for record, raw in zip(report["rows"], payload["rows"]):
        if (record["step"] != raw["step"] or set(record["arms"]) != set(raw["arms"])
                or set(raw["arms"]) != expected_arms):
            raise RuntimeError("panel coordinates changed")
        rebuilt = {}
        for arm, pools in raw["arms"].items():
            rebuilt[arm] = {}
            for pool, values in pools.items():
                expected_shape = (4, 1) if pool == "training_native_dv12" else (4, 12)
                if tuple(values["gap"].shape) != expected_shape or not bool(torch.isfinite(values["gap"]).all()):
                    raise RuntimeError("invalid raw game panel shape/values")
                # Native training means were reduced in CUDA FP32; allow only
                # bounded cross-device FP32 arithmetic discrepancy here. Clean
                # probes used CPU FP64 aggregation and are reconstructed tight.
                score = float(F.softplus(values["gap"]).double().mean())
                error = abs(score - record["arms"][arm][pool]["g_game"])
                tolerance = 3e-7 if pool == "training_native_dv12" else 1e-12
                if error > tolerance:
                    raise RuntimeError(f"score mismatch {record['step']}/{arm}/{pool}: {error}")
                largest_error = max(largest_error, error)
                count += 1
                rebuilt[arm][pool] = score
                expected_delta = record["arms"][arm][pool]["g_game"] - record["arms"]["none"][pool]["g_game"]
                if abs(expected_delta - record["arms"][arm][pool]["delta_from_no_step"]) > 1e-12:
                    raise RuntimeError("per-step paired delta changed")
        reconstructed.append(dict(step=record["step"], hold=record["hold"], arms=rebuilt))
    checks["paired_score_reconstruction"] = True
    checks["finite_raw_shapes_and_per_step_deltas"] = True
    checks["all_eight_role_subsets"] = all(set(r["arms"]) == expected_arms for r in report["rows"])
    for task, hold in (("edit", False), ("preservation", True)):
        chosen = [r for r in report["rows"] if r["hold"] == hold]
        for arm, pools in report["summary_mean_game_delta"][task].items():
            for pool, expected in pools.items():
                actual = sum(r["arms"][arm][pool]["g_game"] - r["arms"]["none"][pool]["g_game"] for r in chosen) / len(chosen)
                if abs(actual - expected) > 1e-12:
                    raise RuntimeError("summary aggregation changed")
    checks["summary_reconstruction"] = True
    rows = json.loads((args.output / "native-training-rows.json").read_text())
    saved = torch.load(args.run / "final.pt", map_location="cpu", weights_only=False, mmap=True)
    data = torch.load(args.run / "data.pt", map_location="cpu", weights_only=False)
    stream = torch.Generator().set_state(saved["data_rng"].cpu())
    sampling_ok = len(rows) == plan["updates"]
    for row, record in zip(rows, report["rows"]):
        hold = row["step"] % 5 == 0
        ids = torch.randint(len(data["holds" if hold else "fit"]["context"]), (4,), generator=stream)
        sampling_ok &= row["hold"] == hold and row["batch_indices"] == ids.tolist()
        if (row["step"] != record["step"] or row["hold"] != record["hold"]
                or abs(row["loss_g"] / row["game_weight"]
                    - record["arms"]["none"]["training_native_dv12"]["g_game"]) > 3e-7):
            raise RuntimeError("pre-step game differs from the native training row")
    checks["cpu_data_sampling_reconstruction"] = bool(sampling_ok)
    checks["native_training_row_game_links"] = True
    review = dict(qualified=all(checks.values()), checks=checks, scores_reconstructed=count,
                  largest_cross_device_score_error=largest_error,
                  limitations="CPU validates sources, sampling and raw game panels. Complete native CUDA replay and in-step state immutability are qualified runtime witnesses; GPU model training is not rerun by this reviewer.")
    (args.output / "independent-review.json").write_text(json.dumps(review, indent=2) + "\n")
    print(json.dumps(review, indent=2))
    if not review["qualified"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

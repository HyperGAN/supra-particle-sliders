#!/usr/bin/env python3
"""Independent CPU qualification of native/game-trust Supra continuations."""
from argparse import ArgumentParser
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.review_e22_supra_pr223 import (
    require, read, sha, state_digest, load, canonical, finite, committed_python_hashes,
    trace_rows, frozen_parameters, audit_text_contexts, audit_evaluation,
    evaluation_comparison, owner_comparison, rolling_losses, COUNTS, JUDGES,
)

PIN = "cabe2084284db923d525918cbf3e18de6f20faac"
ARMS = ("native", "bank_game_trust_v1")
FRACTIONS = (1., .5, .25, .125, 0.)
ORIGINAL_SHA = "24a98ba055d0b5298136d67dc7491a339684da03416c510d1d443fe730eb7069"


def rng_digest(stream_state):
    return hashlib.sha256(bytes(stream_state.cpu().tolist())).hexdigest()


def audit_trust_record(record, row):
    """Recompute the largest passing candidate without importing selector code."""
    finite(record, "trust record")
    require(record["schema"] == "e22_bank_samebatch_game_trust_v1" and record["fractions"] == list(FRACTIONS), "trust law/fractions")
    require(record["step"] == row["step"] and record["task_weight"] == row["game_weight"], "own-batch trust update/task identity")
    for flag in ("original_native_Gpass_replayed_exact", "private_evaluations_state_unchanged", "native_optimizer_moments_unchanged"):
        require(record[flag] is True, f"trust runtime witness {flag}")
    require(record["output_metrics_used"] is False, "trust selector output boundary")
    require(record["moment_law"] == "native Adam moments retained; applied table step projected", "explicit projected-Adam moment law")
    require(record["criterion"] == "same training batch only; zero clean and native noisy paired-game harm relative post-G/router fraction0",
            "own same-batch game criterion")
    baseline = record["baseline"]
    require(set(baseline) == {"clean", "noisy"}, "paired clean/noisy game baseline")
    candidates = record["candidates"]
    require(1 <= len(candidates) <= 4 and [point["fraction"] for point in candidates] == list(FRACTIONS[:len(candidates)]),
            "largest-first predeclared candidate sequence")
    selected = 0.
    for index, point in enumerate(candidates):
        scores = point["game"]
        require(set(scores) == set(baseline), "two game-only candidate criteria")
        require(point["new_minus_baseline"] == {mode: scores[mode] - baseline[mode] for mode in baseline}, "candidate game delta arithmetic")
        passing = all(scores[mode] <= baseline[mode] for mode in baseline)
        if passing:
            require(index == len(candidates) - 1, "stop at first largest passing training-game candidate")
            selected = point["fraction"]
            break
    if selected == 0.:
        require(len(candidates) == 4, "all positive fractions must fail before bank zero projection")
    require(record["selected_fraction"] == selected, "independently selected largest valid bank fraction")
    scores = baseline if selected == 0. else candidates[-1]["game"]
    require(record["accepted_game"] == scores and record["accepted_new_minus_baseline"]
            == {mode: scores[mode] - baseline[mode] for mode in baseline}, "accepted game delta arithmetic")
    require(all(value <= 0 for value in record["accepted_new_minus_baseline"].values()), "zero own-batch clean/noisy game harm")
    require(record["dense_native_gradient_rows"] == row["dense_gradient_rows"]
            and record["proposed_displacement_norm"] >= 0 and record["applied_displacement_norm"] >= 0
            and record["pre_step_output_sigma"] > 0, "native bank/sigma telemetry")
    if selected == 0.:
        require(record["applied_displacement_norm"] == 0., "rejected table proposal must have zero applied displacement")
    if selected == 1.:
        require(record["applied_displacement_norm"] == record["proposed_displacement_norm"], "full native bank proposal norm")


def audit_progress(directory, plan, data, rows):
    stream = torch.Generator().manual_seed(72)
    panels = {}
    for name, pool_name, per_subject in (("edit", "fit", 2), ("preservation", "holds", 4)):
        pool = data[pool_name]["context"]
        indices = []
        for subject in pool[:, 4097].unique(sorted=True):
            choices = (pool[:, 4097] == subject).nonzero().flatten()
            offsets = torch.linspace(0, len(choices) - 1, min(per_subject, len(choices))).round().long()
            indices.extend(choices[offsets].tolist())
        noise = torch.randn((4, len(indices), 256, 16), generator=stream)
        panels[name] = dict(indices=indices, contexts=len(indices), cpu72_noise_digest=state_digest(noise),
                            context_digest=state_digest(pool[indices]))
        for folder in ("reference-probes", *ARMS):
            require(read(directory / folder / f"progress-{name}-indices.json") == indices, f"matched {folder}/{name} progress indices")
    original = read(directory / "original-reference-progress.json")
    require(original["original_sha256"] == ORIGINAL_SHA and original["evaluation_only"] is True
            and original["reference_update"] == 6400 and len(original["points"]) == 1
            and original["judge"] == "common qualified particle V2 step6400 critic", "original6400 constant game reference")
    reference_point = original["points"][0]
    finite(reference_point, "original6400 progress reference")
    require(reference_point["step"] == 6400 and set(reference_point["probes"]) == set(panels),
            "original6400 reference clock/tasks")
    for name in panels:
        require(reference_point["probes"][name]["contexts"] == panels[name]["contexts"]
                and set(reference_point["probes"][name]["clean"]) == {"frozen_start_D"},
                f"original6400 matched clean reference {name}")
    histories = {arm: trace_rows(directory / arm / "progress.jsonl") for arm in ARMS}
    steps = [point["step"] for point in histories[ARMS[0]]]
    require(steps == list(range(6400, plan["final_step"] + 1, 64))
            and [point["step"] for point in histories[ARMS[1]]] == steps, "matched progress horizons")
    for arm, points in histories.items():
        require(read(directory / arm / "progress-latest.json") == points[-1], f"{arm} latest progress")
        for point in points:
            finite(point, f"{arm} progress")
            require(point["native_state_unchanged"] is True and point["output_metrics_used"] is False
                    and point["evaluation_only"] is True and point["output_sigma"] == .125, f"{arm} read-only game probes")
            offset = point["step"] - 6400
            require(point["rolling"] == rolling_losses(rows[arm][:offset]), f"{arm} rolling game units")
            require(set(point["probes"]) == set(panels), f"{arm} fixed progress pools")
            for name in panels:
                require(point["probes"][name]["contexts"] == panels[name]["contexts"], f"{arm} matched progress size")
    require(histories[ARMS[0]][0]["probes"] == histories[ARMS[1]][0]["probes"], "identical initial frozen/live/DV12 progress scores")
    paired = {}
    for first, second in zip(histories[ARMS[0]], histories[ARMS[1]]):
        paired[str(first["step"])] = {name: {mode: dict(native=first["probes"][name][mode]["frozen_start_D"],
            candidate=second["probes"][name][mode]["frozen_start_D"],
            candidate_minus_native=second["probes"][name][mode]["frozen_start_D"] - first["probes"][name][mode]["frozen_start_D"])
            for mode in ("clean", "dv12")} for name in panels}
    endpoint_vs_original = {arm: {name: dict(original=reference_point["probes"][name]["clean"]["frozen_start_D"],
        endpoint=histories[arm][-1]["probes"][name]["clean"]["frozen_start_D"],
        endpoint_minus_original=histories[arm][-1]["probes"][name]["clean"]["frozen_start_D"]
                                - reference_point["probes"][name]["clean"]["frozen_start_D"])
        for name in panels} for arm in ARMS}
    return dict(panels=panels, comparison=paired, endpoint_vs_original6400=endpoint_vs_original,
                reference_budget_limit="Original is a constant6400 quality target; both particle continuations end6656. This does not compare training efficiency at equal6656 budgets.")


def build_review(directory):
    directory = Path(directory).resolve()
    plan, receipt, status = (read(directory / name) for name in ("plan.json", "receipt.json", "status.json"))
    require(receipt["plan"] == plan and plan["schema"] == "supra_bank_game_trust_continuation_v1", "matched run plan")
    require(plan["arms"] == list(ARMS) and plan["start_step"] == 6400 and plan["updates"] == 256
            and plan["final_step"] == 6656 and plan["fractions"] == list(FRACTIONS), "predeclared two-arm256 horizon/fractions")
    require(plan["particlegan_commit"] == PIN, "same pinned native source, no profile migration")
    require(plan["selection"] == "final predeclared fixed horizon; no output-based stopping, selection or optimizer criterion"
            and plan["structural"] == "native learned feature criterion, zero per-context feature harm, no output guard", "objective/structural metric boundary")
    require(status["phase"] == "complete" and status["step"] == status["steps"] == 6656, "completed fixed256 run")
    for flag in ("initial_native_state_exact", "two_update_replay_exact", "frozen_unchanged", "evaluation_state_unchanged",
                 "source_and_inputs_unchanged", "matched_owned_streams"):
        require(receipt[flag] is True, f"GPU runtime witness {flag}")
    inputs = {name: Path(value) for name, value in plan["input_paths"].items()}
    require(set(inputs) == set(plan["input_sha256"]), "complete immutable-input manifest")
    for name, path in inputs.items():
        require(sha(path) == plan["input_sha256"][name], f"immutable input {name}")
    require(plan["input_sha256"]["original_reference"] == ORIGINAL_SHA, "fixed original ordinary-LoRA checkpoint")
    for name, value in plan["application_source_sha256"].items():
        require(sha(directory / "source" / name) == value, f"executed application snapshot {name}")
    for name, value in plan["particlegan_source_sha256"].items():
        require(sha(directory / "source/particlegan" / name) == value, f"native snapshot {name}")
    upstream = ROOT.parent / "ParticleGAN-pr223-supra"
    committed = committed_python_hashes(upstream, PIN)
    require({str(Path(name).relative_to("particlegan")): value for name, value in committed.items()}
            == plan["particlegan_source_sha256"], "actual native cabe git tree")
    require(state_digest({Path(name).name: value for name, value in plan["particlegan_source_sha256"].items()
                          if Path(name).parent == Path(".")}) == plan["native_source_digest"], "native source digest")
    initial, data = load(inputs["final.pt"]), torch.load(inputs["data.pt"], map_location="cpu", weights_only=False)
    starting_run, starting_review = read(inputs["run.json"]), read(inputs["qualification-review.json"])
    require(starting_review["qualified"] is True and state_digest(initial)
            == plan["initial_native_digest"] == starting_review["final_native_digest"], "qualified identical6400 starting state")
    require(starting_run["particlegan_commit"] == PIN and starting_run["particlegan_source_digest"] == plan["native_source_digest"],
            "qualified source pin")
    require(state_digest(data) == initial["config"]["dataset_digest"], "cached dataset semantics")
    five = read(inputs["qualified_five_step_reference"])
    require(five["exact_native_full_replay"] is True and five["final_native_digest"]
            == "ede3822d70475b7dc51555d8ecece6d135cfb0253f872865e33c26fb860130e5", "qualified native five-step reference")
    initial_frozen = state_digest(frozen_parameters(initial))
    states, rows, metrics, trace_reports = {}, {}, {}, {}
    for arm in ARMS:
        folder = directory / arm
        result, wrapper = read(folder / "receipt.json"), load(folder / "final.pt")
        require(result == receipt["results"][arm], f"{arm} receipt consistency")
        require(wrapper["schema"] == "supra_game_trust_experimental_checkpoint_v1" and wrapper["formulation"] == arm
                and wrapper["plan"] == plan, f"{arm} explicitly tagged native checkpoint")
        state = wrapper["native"]
        require(state["policy"]["completed_steps"] == 6656 and state["config"] == initial["config"], f"{arm} native law/architecture/horizon")
        require(state_digest(state) == result["final_native_digest"] and sha(folder / "final.pt") == result["final_checkpoint_sha256"],
                f"{arm} final native checkpoint witness")
        require(state_digest(frozen_parameters(state)) == initial_frozen, f"{arm} immutable FAST/average tensors/buffers")
        audit_text_contexts(state, data, arm)
        require(tuple(state["policy"]["table"].shape) == (128, 4) and state["policy"]["table_requires_grad"]
                and state["policy"]["routing"]["config"]["max_context_harm"] == 0.
                and state["config"]["output_error_guard"] is False and len(state["config"]["sites"]) == 71,
                f"{arm} meaningful particle ownership/sites and native zero feature guard")
        replay = read(folder / "replay-review.json")
        require(replay["initial_native_state_exact"] is True and replay["two_update_replay_exact"] is True, f"{arm} exact resume runtime witness")
        arm_rows = trace_rows(folder / "train.jsonl")
        require(len(arm_rows) == 256 and [row["step"] for row in arm_rows] == list(range(6401, 6657)), f"{arm} fixed native clock")
        finite(arm_rows, f"{arm} trace")
        stream = torch.Generator().set_state(initial["data_rng"].cpu())
        for row in arm_rows:
            hold = row["step"] % 5 == 0
            expected = torch.randint(len(data["holds" if hold else "fit"]["context"]), (4,), generator=stream).tolist()
            require(row["hold"] is hold and row["game_weight"] == (.1 if hold else 1.) and row["batch_indices"] == expected,
                    f"{arm} exact original sampling/preservation at{row['step']}")
            require(row["penalty_calls"] == row["step"] and row["output_sigma"] > 0 and row["bank_grad_norm"] >= 0., f"{arm} native penalty/noise/grad")
            if arm == ARMS[1]:
                audit_trust_record(row["bank_trust"], row)
            else:
                require(row["bank_trust"] is None, "unchanged native control has no projection")
        require(torch.equal(stream.get_state(), state["data_rng"]), f"{arm} saved CPU data RNG")
        for key, value in (("data_rng_digest", state["data_rng"]), ("paired_rng_digest", state["paired_noise_rng"]),
                           ("dv12_rng_digest", state["policy"]["streams"]["noise_generator"])):
            require(rng_digest(value) == result[key], f"{arm} saved RNG digest {key}")
        trust_records = [row["bank_trust"] for row in arm_rows if row["bank_trust"] is not None]
        require(trust_records == result["trust_rows"], f"{arm} trust telemetry trace/receipt pairing")
        metrics[arm] = read(folder / "evaluation.json")
        audit_evaluation(metrics[arm], data, arm)
        require({pool: {key: value for key, value in record.items() if key != "records"} for pool, record in metrics[arm].items()}
                == result["metrics"], f"{arm} raw-record evaluation aggregation")
        states[arm], rows[arm] = state, arm_rows
        trace_reports[arm] = dict(updates=256, preservation_updates=sum(row["hold"] for row in arm_rows),
            accepted_structural_moves=sum(row["move"].get("moves", 0) if row.get("move") else 0 for row in arm_rows),
            dense_bank_gradient_updates=sum(row["dense_gradient_rows"] == 128 for row in arm_rows),
            selected_fraction_counts=dict(Counter(str(record["selected_fraction"]) for record in trust_records)),
            projected_bank_updates=sum(record["selected_fraction"] < 1. for record in trust_records),
            nonzero_bank_updates=sum(record["applied_displacement_norm"] > 0. for record in trust_records))
    require(trace_reports[ARMS[1]]["projected_bank_updates"] > 0, "trust intervention must actually change bank motion")
    require(trace_reports[ARMS[1]]["nonzero_bank_updates"] > 0, "experimental bank must retain meaningful learned motion")
    paired_fields = ("step", "hold", "game_weight", "batch_indices", "base_noise_sums", "paired_rng_digest", "dv12_rng_digest")
    require(all(all(first[key] == second[key] for key in paired_fields) for first, second in zip(rows[ARMS[0]], rows[ARMS[1]])),
            "all256 matched native context/Gaussian/DV12 draws")
    for key in ("data_rng", "paired_noise_rng"):
        require(torch.equal(states[ARMS[0]][key], states[ARMS[1]][key]), f"matched final {key}")
    comparison, pairing = evaluation_comparison(metrics[ARMS[0]], metrics[ARMS[1]], data)
    deltas = {pool: {judge: comparison[pool][judge]["new_minus_old"] for judge in JUDGES} for pool in COUNTS}
    require(deltas == receipt["game_deltas_candidate_minus_native"], "reconstructed endpoint paired game deltas")
    original = read(directory / "evaluation-original.json")
    audit_evaluation(original, data, "original6400")
    gaps = {arm: {pool: {judge: metrics[arm][pool][judge]["g_game"] - original[pool][judge]["g_game"]
            for judge in JUDGES} for pool in COUNTS} for arm in ARMS}
    require(gaps == receipt["game_gaps_to_original_6400_reference"], "reconstructed original6400 game gaps")
    progress = audit_progress(directory, plan, data, rows)
    return dict(schema="supra_bank_game_trust_independent_cpu_review_v1", qualified=True, gpu_used=False,
        run=str(directory), fixed_updates=256, start_step=6400, final_step=6656, source_pin=PIN,
        source_snapshot_files_verified=len(plan["application_source_sha256"]) + len(plan["particlegan_source_sha256"]),
        input_files_verified=len(inputs), native_git_tree_verified=True, trace=trace_reports,
        matched_training_draws=256, native_owner_comparison=owner_comparison(states[ARMS[0]], states[ARMS[1]]),
        comparison=comparison, game_deltas_candidate_minus_native=deltas, paired_context_scores=pairing,
        game_gaps_to_original6400=gaps, progress=progress,
        scoring_critic_digests={"fixed_start_D": state_digest(initial["policy"]["models"]["critic"]),
                               "arm_final_D": state_digest(states[ARMS[0]]["policy"]["models"]["critic"])},
        output_metrics_used_for_optimizer_or_selection=False,
        interpretation="Trust accepts only same-batch training-game-safe bank proposals. Generalization gains or losses remain valid outcomes; no result is required to improve.",
        limits=["CPU review verifies saved tensor/source/stream content and recomputes raw-record aggregates and trust decisions; actual GPU forwards and moment preservation are runtime witnesses.",
                "Private progress CPU72/DV12 and final CUDA72 panels are distinct. The same fixed6400 and same native-final critics are shared by both particle endpoints and original6400 reference.",
                "Only256 late updates on one task/stream. Original ordinary LoRA is a fixed6400 quality target, not an equal6656-budget retrain.",
                "Recorded training_seconds includes endpoint evaluation and extra original-reference evaluation in native arm; it must not support comparative training-speed claims.",
                "Retained native moments plus projected parameter motion define an explicit experimental optimizer law. No production architecture/optimizer promotion follows from replay checks."],
        reviewer_sha256=sha(Path(__file__)), artifact_sha256={name: sha(directory / name) for name in
            ("plan.json", "receipt.json", "evaluation-original.json", "original-reference-progress.json")})


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "outputs/e22-bank-game-trust-256")
    parser.add_argument("--output", type=Path, help="Default: RUN/independent-review.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    output = args.output or args.run / "independent-review.json"
    require(not output.exists(), "preserve existing review evidence; choose another output")
    report = build_review(args.run)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ("qualified", "gpu_used", "trace", "game_deltas_candidate_minus_native", "game_gaps_to_original6400")}))


if __name__ == "__main__":
    main()

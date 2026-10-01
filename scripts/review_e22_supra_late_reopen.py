#!/usr/bin/env python3
"""Independently qualify the fixed late-reopen diagnostic using CPU artifacts.

This script never runs a model, trains, changes a native owner, or selects a
checkpoint. Either sign of the learned-game differences is a valid result.
GPU replay and forward calculations remain explicitly runtime witnesses.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
import math
from pathlib import Path
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.review_e22_supra_pr223 import (
    COUNTS, JUDGES, PIN, audit_text_contexts, canonical, close,
    committed_python_hashes, diagnostic_value, finite, frozen_parameters,
    load, read, sha, state_digest, trace_rows,
)

DRIVER = "scripts/diagnose_e22_supra_late_reopen.py"
DRIVER_SHA = "6a4ee6f87bfa8b333f0e1eef372219afeb9ddffc85a0bf69ba48a2292d53d3bd"
ARMS = {
    "native": (6000, "native_replay_v1"),
    "cancel_generator_drift_release": (6221, "cancel_generator_drift_release6222_v1"),
    "omit_moment_rescale": (6387, "omit_event6387_moment_rescale_v1"),
    "omit_anchor_latch": (6387, "omit_event6387_anchor_latch_v1"),
    "omit_ladder_reopens": (6387, "omit_event6387_ladder_reopens_v1"),
    "omit_all_event_actions": (6387, "omit_event6387_all_three_actions_v1"),
}
FIELDS = ("hold", "game_weight", "batch_indices", "base_noise_sums", "paired_rng_digest", "dv12_rng_digest")
GROUPS = {"0.0": "generator", "0.1": "router", "0.2": "table", "0.3": "noise", "1.0": "critic"}
CHECKS = 0


def require(value, label):
    global CHECKS
    CHECKS += 1
    if not value:
        raise ValueError(label)


def equal(first, second, label):
    require(state_digest(exact_owners(first)) == state_digest(exact_owners(second)), label)


def exact_owners(value):
    """Decode transient KA2 objects by content, never their address repr.

Native checkpoint digests retain their established format. Transient action
records additionally contain an anchor owner with live-module references.
Its full instance fields and referenced module tensors/flags must be compared
by content, including values hidden by ordinary tensor/module repr truncation.
"""
    if isinstance(value, torch.nn.Module):
        return dict(type=type(value).__module__ + "." + type(value).__qualname__,
                    tensors=value.state_dict(),
                    training={name: module.training for name, module in value.named_modules()},
                    requires_grad={name: parameter.requires_grad for name, parameter in value.named_parameters()})
    if isinstance(value, dict):
        return {key: exact_owners(item) for key, item in value.items()}
    if isinstance(value, list):
        return [exact_owners(item) for item in value]
    if isinstance(value, tuple):
        return tuple(exact_owners(item) for item in value)
    if isinstance(value, torch.Tensor) or value is None or isinstance(value, (str, bool, int, float)):
        return value
    if type(value).__module__.startswith("particlegan.") and hasattr(value, "__dict__"):
        return dict(type=type(value).__module__ + "." + type(value).__qualname__,
                    fields=exact_owners(value.__dict__))
    raise TypeError("unknown transient owner " + type(value).__module__ + "." + type(value).__qualname__)


def cross_fork_equal(first, second, label):
    """Keep actual owner checks strict; disclose one nonserialized EMA cache.

The source-pinned KA2 record computes decay and assigns anchor.decay before
every EMA update; when the computed value is1 it skips that update entirely.
Anchor evaluation/start never read the cache. Restore therefore need not
serialize the previously applied decay. Within-action equality still checks
it, so an intervention cannot silently mutate it.
"""
    a, b = exact_owners(first), exact_owners(second)
    cached_a = a["ka2_record"]["anchor"]["fields"].pop("decay")
    cached_b = b["ka2_record"]["anchor"]["fields"].pop("decay")
    equal(a, b, label + " except disclosed unused EMA cache")
    return dict(first=cached_a, second=cached_b, equal=cached_a == cached_b,
                interpretation="Nonserialized previously applied EMA decay; overwritten before every EMA update, never read by evaluation/start.")


def clone(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone(item) for item in value)
    return deepcopy(value)


def fast_from_native(state):
    policy = state["policy"]
    result = {role: {name: value for name, value in module.items()
                     if policy["requires_grad"][role].get(name, False)}
              for role, module in policy["models"].items()}
    result["table"] = {"table": policy["table"]}
    result["noise"] = {"log_output_sigma": policy["output_noise"]}
    return result


def rates(values):
    """Applied group rates are distinct from end-of-step raw settling scales."""
    return {key: dict(role=group["role"], raw_scale=group["raw_scale"],
                     actual_lr=group["lr"], base_lr=group["base_lr"],
                     actual_base_fraction=group["lr"] / group["base_lr"])
            for key, group in values["groups"].items()}


def group_flat(values, role):
    return torch.cat([value.reshape(-1) for value in values["fast"][role].values()])


def native_moment_action(before, omitted):
    expected = clone(before)
    if not omitted:
        for key, group in expected["groups"].items():
            ratio = before["surprise"]["last_ratios"].get(key)
            if ratio is not None and ratio > 1:
                for moment in group["moments"]:
                    for field in ("exp_avg_sq", "max_exp_avg_sq"):
                        if field in moment:
                            moment[field].mul_(1. / (ratio * ratio))
    return expected


def native_anchor_action(before, omitted):
    expected = clone(before)
    surprise = expected["surprise"]
    ratio = before["ka2_record"]["last_ratio"]
    if not omitted and ratio is not None:
        surprise["anchor_event"] = [False]
        surprise["anchor_events"] += 1
    if surprise["anchor_event"] is None:
        return expected
    if ratio is not None and ratio > 3.:
        surprise["anchor_event"][0] = True
    if surprise["anchor_event"][0] and ratio is not None and ratio < 1.75:
        surprise["anchor_event"] = None
        return expected
    expected["controller"]["data_drive"] = 1.
    return expected


def native_ladder_action(before, role, omitted):
    expected = clone(before)
    key = next(key for key, value in GROUPS.items() if value == role)
    tester = expected["groups"][key]["tester"]
    if omitted:
        if tester["anchor"] is None:
            tester["anchor"] = group_flat(before, role).clone()
    else:
        tester.update(anchor=group_flat(before, role).clone(), blocks=[], tau=0.,
                      invalid_block_rows=None, s=1., b=1., r_b=[], r_2b=[],
                      blocks_in_window=0, last_decisive=0, last_decisive_scale=None)
        tester["counts"]["reopens"] += 1
        if "b_anchor" in tester:
            tester["b_anchor"] = None
    expected["groups"][key]["raw_scale"] = tester["s"]
    return expected


def audit_values(values, label):
    require(set(values) == {"fast", "groups", "surprise", "guard", "controller", "ka2_record"}, label + " owners")
    require({key: value["role"] for key, value in values["groups"].items()} == GROUPS, label + " native optimizer layout")
    require({role: len(values["fast"][role]) for role in ("generator", "router", "critic")} ==
            {"generator": 284, "router": 142, "critic": 7}, label + " actual FAST ownership")
    require(tuple(values["fast"]["table"]["table"].shape) == (128, 4), label + " particle bank retained")
    for key, value in values["groups"].items():
        require(value["tester"]["s"] == value["raw_scale"], label + "/" + key + " raw scale is tester state")
        require(value["lr"] > 0 and value["base_lr"] > 0 and math.isfinite(value["lr"]), label + "/" + key + " finite actual LR")


def audit_begin_rates(before, after, label):
    equal(before["fast"], after["fast"], label + " begin leaves FAST values unchanged")
    table_scale = after["groups"]["0.2"]["raw_scale"]
    damping = 1. / (1. + after["controller"]["payoff_error"] ** 2)
    for key, group in after["groups"].items():
        raw = group["raw_scale"]
        expected = group["base_lr"] * (max(raw, .75 * table_scale) if key == "1.0" else raw)
        if key == "1.0":
            expected *= damping
        close(group["lr"], expected, label + "/" + key + " actual native floor/damping rate")
    return dict(before=rates(before), after=rates(after), critic_table_floor=.75 * table_scale,
                critic_payoff_damping=damping)


def audit_game(value, data, label, *, full):
    require(value["full_panel"] is full and value["evaluation_only"] is True
            and value["output_metrics_used"] is False and value["output_sigma"] == .125
            and value["native_state_unchanged"] is True
            and value["source"] == "FAST current particle G", label + " private game-only FAST evaluation")
    require(set(value["pools"]) == set(COUNTS), label + " all evaluation pools")
    finite(value, label)
    for pool, count in COUNTS.items():
        context = data[pool]["context"]
        selected = list(range(count))
        if not full:
            selected = []
            for subject in context[:, 4097].unique(sorted=True):
                indices = (context[:, 4097] == subject).nonzero().flatten()
                selected.extend(indices[[0, len(indices) - 1]].tolist())
            selected = sorted(set(selected))
        result = value["pools"][pool]
        records = result["records"]
        require(result["count"] == len(records) == len(selected) and [row["index"] for row in records] == selected,
                label + "/" + pool + " declared contexts/count/order")
        for row in records:
            index = row["index"]
            require(row["source_caption_id"] == int(context[index, 4097])
                    and row["time"] == float(context[index, 4096]), label + "/" + pool + " context identity")
            require(set(row) == {"index", "source_caption_id", "time", *JUDGES}, label + "/" + pool + " no output scoring fields")
            for judge in JUDGES:
                require(set(row[judge]) == {"g_game", "d_game", "score_gap"}, label + "/" + pool + " learned-game score fields")
                close(row[judge]["g_game"] - row[judge]["d_game"], row[judge]["score_gap"],
                      label + "/" + pool + "/" + judge + " paired logistic identity", 2e-6)
        for judge in JUDGES:
            for metric in ("g_game", "d_game", "score_gap"):
                close(result[judge][metric], sum(row[judge][metric] for row in records) / len(records),
                      label + "/" + pool + "/" + judge + "/" + metric + " raw aggregate")


def paired_changes(first, second):
    result = {}
    for pool in COUNTS:
        pairs = list(zip(first["pools"][pool]["records"], second["pools"][pool]["records"]))
        result[pool] = {}
        for judge in JUDGES:
            changes = []
            subjects = defaultdict(list)
            for a, b in pairs:
                require(all(a[key] == b[key] for key in ("index", "source_caption_id", "time")), pool + " common-panel pairing")
                change = b[judge]["g_game"] - a[judge]["g_game"]
                changes.append(change)
                subjects[str(a["source_caption_id"])].append(change)
            result[pool][judge] = dict(contexts=len(changes), mean_game_arm_minus_native=sum(changes) / len(changes),
                contexts_improved=sum(change < 0 for change in changes), contexts_equal=sum(change == 0 for change in changes),
                contexts_worsened=sum(change > 0 for change in changes),
                subjects={subject: dict(contexts=len(values), mean_game_arm_minus_native=sum(values) / len(values))
                          for subject, values in subjects.items()})
    return result


def build_review(directory, pg_root):
    directory = Path(directory).resolve()
    plan, receipt, status = [read(directory / name) for name in ("plan.json", "receipt.json", "status.json")]
    require(plan["schema"] == receipt["schema"] == "supra_late_native_reopen_causal_v1"
            and receipt["plan"] == plan, "diagnostic schema and fixed plan")
    require(receipt["qualified"] is True and all(value is True for value in receipt["checks"].values())
            and status["qualified"] is True and status["phase"] == "complete" and status["step"] == 6400,
            "complete GPU runtime witnesses")
    require(plan["native_pin"] == PIN and plan["start_step"] == 6000 and plan["fixed_endpoint"] == 6400
            and plan["drift_decision_row"] == 6222 and plan["late_completed_clock"] == 6387
            and plan["late_first_affected_row"] == 6388 and plan["schedule"] == "editing_only_v1", "predeclared clocks/law")
    require({arm: (value["fork"], value["law"]) for arm, value in plan["arms"].items()} ==
            {arm: values for arm, values in ARMS.items() if arm != "native"}, "exact opt-in diagnostic arms")
    require(set(receipt["results"]) == set(status["arms"]) == set(ARMS), "all fixed arms completed")
    for item in (plan, receipt):
        require(item["raw_output_scoring"] is False and item["checkpoint_selection"] is False, "no output criterion or checkpoint selection")
    require(plan["native_recipe_changes"] is False and plan["core_source_changes"] is False, "native law/source ownership preserved")
    require(plan["application_source_sha256"][DRIVER] == DRIVER_SHA, "independently audited diagnostic source pin")
    pg_root = Path(pg_root).resolve()
    sys.path.insert(0, str(pg_root))
    import particlegan
    require(Path(particlegan.__file__).resolve().parent.parent == pg_root,
            "pinned native dataclass import for CPU checkpoint decoding")
    require(subprocess.check_output(["git", "-C", str(pg_root), "rev-parse", "HEAD"], text=True).strip() == PIN,
            "native commit")
    require(committed_python_hashes(pg_root, PIN) == plan["native_source_sha256"], "native source archive pin")
    artifacts = {}

    def artifact(path, expected=None):
        path = Path(path)
        digest = sha(path)
        if expected is not None:
            require(digest == expected, "artifact SHA " + str(path))
        artifacts[str(path.relative_to(directory))] = digest
        return digest

    for name, digest in plan["application_source_sha256"].items():
        require(sha(ROOT / name) == digest, "held application source " + name)
        artifact(directory / "source" / name, digest)
    for name, digest in plan["native_source_sha256"].items():
        require(sha(pg_root / name) == digest, "native working source " + name)
        artifact(directory / "source/native" / name, digest)
    inputs = {key: Path(value) for key, value in plan["input_paths"].items()}
    require(set(inputs) == set(plan["input_sha256"]), "complete immutable input manifest")
    for key, path in inputs.items():
        require(sha(path) == plan["input_sha256"][key], "immutable input " + key)
    parent_review, parent_plan, parent_receipt = [read(inputs[key]) for key in ("parent_review", "parent_plan", "parent_receipt")]
    require(parent_review["qualified"] is True and parent_review["fixed_updates"] == 6400
            and parent_receipt["qualified"] is True and parent_receipt["plan"] == parent_plan, "qualified native trajectory parent")
    for key, name in (("initial", "checkpoint-06000.pt"), ("final", "final.pt"), ("trace", "train.jsonl"),
                      ("final_evaluation", "evaluation-v3.json"), ("parent_plan", "plan.json"), ("parent_receipt", "receipt.json")):
        require(plan["input_sha256"][key] == parent_review["artifact_sha256"][name], "parent reviewed artifact " + key)
    for key in ("data", "judge", "control"):
        require(plan["input_sha256"][key] == parent_plan["input_sha256"][key], "qualified auxiliary " + key)
    require({key: value for key, value in plan["application_source_sha256"].items() if key != DRIVER} ==
            parent_plan["application_source_sha256"] and plan["native_source_sha256"] == parent_plan["particlegan_source_sha256"],
            "unchanged full native application/PG source")
    initial, expected_final, data = [load(inputs[key]) for key in ("initial", "final", "data")]
    require(state_digest(initial) == parent_review["checkpoints"]["6000"]["native_digest"]
            and state_digest(expected_final) == parent_review["final_native_digest"], "qualified native6000/6400 boundaries")
    require(initial["policy"]["completed_steps"] == 6000 and expected_final["policy"]["completed_steps"] == 6400,
            "actual checkpoint clocks")
    require(initial["config"] == expected_final["config"] and initial["config"]["architecture"] == "gated_particle_v3"
            and initial["config"]["particle_profile"] == "pr223_shared_routed_v1"
            and initial["config"]["training_schedule"] == "editing_only_v1"
            and initial["config"]["max_feature_context_harm"] == 0 and initial["config"]["output_error_guard"] is False,
            "unchanged particles/learning law/native guards")
    original = trace_rows(inputs["trace"])
    require(len(original) == 6400 and [row["step"] for row in original] == list(range(1, 6401)), "qualified full native rows")
    old_game = read(inputs["final_evaluation"])
    critics = {label: load(inputs[key])["policy"]["models"]["critic"] for label, key in zip(JUDGES, ("judge", "control"))}
    require(plan["critic_digests"] == {label: state_digest(value) for label, value in critics.items()}
            and plan["critic_digests"]["fixed_start_D"] == parent_plan["critic_digests"]["D1856"]
            and plan["critic_digests"]["arm_final_D"] == parent_plan["critic_digests"]["control_final"], "two immutable common learned judges")
    for label, critic in critics.items():
        equal(critic["scale"], data["coordinate_scale"], "fixed learned-game coordinate units " + label)
    forks = {}
    require(set(receipt["forks"]) == {"6221", "6387"}, "both declared native forks")
    for step, metadata in receipt["forks"].items():
        path = directory / metadata["path"]
        artifact(path, metadata["file_sha256"])
        state = load(path)
        require(state["policy"]["completed_steps"] == int(step) and state_digest(state) == metadata["native_digest"], "native fork boundary " + step)
        require(state["config"] == initial["config"], "strict fork learning law " + step)
        equal(frozen_parameters(state), frozen_parameters(initial), "frozen fork models/averages " + step)
        audit_text_contexts(state, data, "fork" + step)
        forks[int(step)] = state
    require(forks[6387]["policy"]["surprise"]["anchor_event"] is None, "target late fork has no prior anchor latch")
    finals, games, snapshots, actions = {}, {}, {}, {}
    action_reports = {}
    for arm, (start, law) in ARMS.items():
        result = read(directory / arm / "result.json")
        require(result == receipt["results"][arm] and result["arm"] == arm and result["experimental_law"] == law
                and result["fork_step"] == start and result["rows"] == 6400 - start, "fixed arm contract " + arm)
        origin = initial if arm == "native" else forks[start]
        require(result["origin_native_digest"] == state_digest(origin), "native origin " + arm)
        artifact(directory / arm / "result.json")
        trace_path = directory / arm / "train.jsonl"
        artifact(trace_path, result["trace_sha256"])
        rows = trace_rows(trace_path)
        require(len(rows) == 6400 - start and [row["step"] for row in rows] == list(range(start + 1, 6401)), "fixed declared rows " + arm)
        generator = torch.Generator(device="cpu")
        generator.set_state(origin["data_rng"])
        for row in rows:
            finite(row, arm + "row" + str(row["step"]))
            reference = original[row["step"] - 1]
            require(row["hold"] is False and row["game_weight"] == 1. and row["output_sigma"] == .125, "editing-only native row " + arm)
            require(row["batch_indices"] == torch.randint(len(data["fit"]["context"]), (4,), generator=generator).tolist(),
                    "independent Generator7 sampled batch " + arm + str(row["step"]))
            require(all(row[field] == reference[field] for field in FIELDS), "same paired training streams " + arm + str(row["step"]))
            require(not row.get("move") or row["move"].get("moves", 0) == 0, "no structural jumps " + arm)
            if arm == "native":
                require(row == reference, "every native replay row exact " + str(row["step"]))
        path = directory / arm / result["final"]["path"]
        artifact(path, result["final"]["file_sha256"])
        wrapper = load(path)
        require(set(wrapper) == {"schema", "experimental_law", "origin_native_digest", "native"}
                and wrapper["schema"] == "supra_native_event_diagnostic_state_v1"
                and wrapper["experimental_law"] == law and wrapper["origin_native_digest"] == state_digest(origin), "versioned diagnostic checkpoint " + arm)
        final = wrapper["native"]
        finals[arm] = final
        require(state_digest(final) == result["final"]["native_digest"] and final["policy"]["completed_steps"] == 6400
                and final["config"] == initial["config"] and final["policy"]["recipe"] == initial["policy"]["recipe"], "actual final native law/state " + arm)
        equal(frozen_parameters(final), frozen_parameters(initial), "all frozen final models/averages " + arm)
        audit_text_contexts(final, data, "final" + arm)
        equal(final["data_rng"], generator.get_state(), "independent final fit-only RNG " + arm)
        for key in ("data_rng", "paired_noise_rng"):
            equal(final[key], expected_final[key], "final application stream " + arm + key)
        for key in ("streams", "cpu_rng", "cuda_rng"):
            equal(final["policy"][key], expected_final["policy"][key], "final native stream " + arm + key)
        equal(final["policy"]["routing"]["stream"], expected_final["policy"]["routing"]["stream"], "final private structural stream " + arm)
        require(final["policy"]["routing"]["counters"]["moves"] == result["structural_moves"] == 0,
                "native routed structural ownership retained " + arm)
        equal(final["policy"]["models"]["critic"]["scale"], data["coordinate_scale"], "FAST critic units " + arm)
        equal(final["policy"]["optimizers"][1]["regularizer"]["ema"]["scale"], data["coordinate_scale"], "actual KA2 anchor units " + arm)
        require(result["actual_rates"] == [[group["lr"] for group in optimizer["param_groups"]]
                                           for optimizer in final["policy"]["optimizers"]], "actual final rates " + arm)
        require(result["surprise"] == diagnostic_value(final["policy"]["surprise"])
                and result["guard"] == diagnostic_value(final["policy"]["reopen_guard"]), "actual final detector/guard state " + arm)
        game_path = directory / arm / "game-06400.json"
        artifact(game_path, result["game_sha256"])
        game = read(game_path)
        require(game == result["full_game"] and game["step"] == 6400 and game["native_digest"] == state_digest(final), "final runtime evaluation native witness " + arm)
        audit_game(game, data, arm, full=True)
        games[arm] = game
        snapshots[arm] = {}
        expected_ready = {step for step in (6221, 6222, 6223, 6387, 6388, 6400) if step >= start}
        require(set(result["ready_snapshots"]) == {str(step) for step in expected_ready}, "all declared ready snapshots " + arm)
        for step, metadata in result["ready_snapshots"].items():
            path = directory / arm / metadata["filename"]
            artifact(path, metadata["sha256"])
            payload = load(path)
            require(payload["schema"] == "supra_native_event_ready_values_v1" and payload["arm"] == arm
                    and payload["experimental_law"] == law and payload["step"] == int(step) and payload["phase"] == "ready", "actual ready phase " + arm + step)
            values = payload["values"]
            audit_values(values, arm + step)
            require(state_digest(values["fast"]) == metadata["fast_digest"], "ready FAST digest " + arm + step)
            snapshots[arm][int(step)] = values
        equal(snapshots[arm][6400]["fast"], fast_from_native(final), "final ready FAST equals actual native owner " + arm)
        if start != 6000:
            equal(snapshots[arm][start]["fast"], fast_from_native(origin), "initial ready FAST equals actual native fork " + arm)
        actions[arm] = defaultdict(list)
        event_reports = []
        for metadata in result["actions"]["records"]:
            path = directory / arm / metadata["filename"]
            artifact(path, metadata["sha256"])
            payload = load(path)
            name = payload["action"]
            require(payload["schema"] == "supra_native_event_action_values_v1" and payload["arm"] == arm and name == metadata["action"], "action ownership " + arm + name)
            before, after = payload["before"], payload["after"]
            audit_values(before, arm + name + "before")
            audit_values(after, arm + name + "after")
            equal(before["fast"], after["fast"], "native action cannot write FAST values " + arm + name)
            require(state_digest(before["fast"]) == metadata["before_fast_digest"] == metadata["after_fast_digest"], "action FAST receipt " + arm + name)
            for side, values in (("before", before), ("after", after)):
                require(metadata[side + "_groups"] == {key: {field: group[field] for field in ("role", "lr", "base_lr", "raw_scale")}
                                                        for key, group in values["groups"].items()}, "action raw/applied rate receipt " + arm + name)
            omitted = metadata["omitted"]
            if name == "moments":
                require(omitted is (arm in ("omit_moment_rescale", "omit_all_event_actions")), "predeclared moment law " + arm)
                equal(after, native_moment_action(before, omitted), "own-group second moment1/r²; first moments/FAST/other owners intact " + arm)
            elif name == "anchor":
                require(omitted is (arm in ("omit_anchor_latch", "omit_all_event_actions")), "predeclared anchor law " + arm)
                equal(after, native_anchor_action(before, omitted), "exact KA2 latch-only action " + arm)
            elif name.startswith("ladder-"):
                require(omitted is (arm in ("omit_ladder_reopens", "omit_all_event_actions")), "predeclared ladder law " + arm)
                equal(after, native_ladder_action(before, name.removeprefix("ladder-"), omitted), "exact native tester restart/begin action " + arm + name)
            elif name.startswith("begin-row-"):
                require(omitted is False, "begin wrapper only observes native action " + arm)
                event_reports.append(dict(action=name, **audit_begin_rates(before, after, arm + name)))
            elif name == "generator-release6222":
                require(omitted is (arm == "cancel_generator_drift_release"), "predeclared generator scale-only cancellation " + arm)
                require(before["groups"]["0.0"]["raw_scale"] == .5
                        and after["groups"]["0.0"]["raw_scale"] == (.5 if omitted else 1.)
                        and after["groups"]["0.0"]["tester"]["last"]["decision"] == "drift"
                        and after["groups"]["0.0"]["tester"]["last"]["step"] == 6222, "actual native drift decision " + arm)
            else:
                require(False, "unknown diagnostic action " + name)
            actions[arm][name].append(payload)
        counts = Counter({key: len(actions[arm][name]) for key, name in
                          (("moments", "moments"), ("anchor", "anchor"), ("generator_release", "generator-release6222"))})
        counts["ladder"] = sum(len(value) for name, value in actions[arm].items() if name.startswith("ladder-"))
        require(dict(counts) == result["actions"]["counts"], "independent action counts " + arm)
        if arm != "cancel_generator_drift_release":
            require(dict(counts) == dict(moments=1, anchor=1, ladder=5, generator_release=1 if arm == "native" else 0), "single target late event " + arm)
        else:
            require(counts["generator_release"] == 1, "single earlier generator cancellation")
        action_reports[arm] = dict(counts=dict(counts), begin_rates=event_reports,
            ready_rates={str(step): rates(value) for step, value in snapshots[arm].items()},
            final_fast_differs_from_native=state_digest(fast_from_native(final)) != state_digest(fast_from_native(expected_final)))
    equal(finals["native"], expected_final, "400-update replay exact full final native state")
    for pool, count in COUNTS.items():
        require(len(games["native"]["pools"][pool]["records"]) == len(old_game[pool]["records"]) == count, "complete baseline qualified full game " + pool)
        for got, expected in zip(games["native"]["pools"][pool]["records"], old_game[pool]["records"]):
            require(all(got[judge] == expected[judge] for judge in JUDGES), "baseline re-evaluates every qualified raw game " + pool)
    for step, state in forks.items():
        equal(snapshots["native"][step]["fast"], fast_from_native(state), "captured baseline fork FAST " + str(step))
    native_release = actions["native"]["generator-release6222"][0]
    cancelled = actions["cancel_generator_drift_release"]["generator-release6222"][0]
    cross_fork_cache = {}
    cross_fork_cache["generator6222"] = cross_fork_equal(cancelled["before"], native_release["before"], "earlier arm native-identical before drift cancellation")
    expected = clone(native_release["after"])
    expected["ka2_record"]["anchor"].decay = cancelled["before"]["ka2_record"]["anchor"].decay
    expected["groups"]["0.0"]["tester"]["s"] = .5
    expected["groups"]["0.0"]["raw_scale"] = .5
    equal(cancelled["after"], expected, "earlier arm differs only generator tester.s; all decision memory/moments/owners retained")
    equal(snapshots["native"][6222]["fast"], snapshots["cancel_generator_drift_release"][6222]["fast"], "cancelled decision leaves update6222 FAST identical")
    native_begin = actions["native"]["begin-row-6388"][0]
    for arm in ARMS:
        if ARMS[arm][0] != 6387:
            continue
        begin = actions[arm]["begin-row-6388"][0]
        cross_fork_cache[arm] = cross_fork_equal(begin["before"], native_begin["before"], "same complete immediate preevent owner " + arm)
        cross_fork_equal(actions[arm]["moments"][0]["before"], actions["native"]["moments"][0]["before"], "same accepted detector/guard event before first omitted action " + arm)
        expected = clone(native_begin["after"])
        expected["ka2_record"]["anchor"].decay = begin["before"]["ka2_record"]["anchor"].decay
        if arm in ("omit_moment_rescale", "omit_all_event_actions"):
            for key in GROUPS:
                expected["groups"][key]["moments"] = clone(begin["before"]["groups"][key]["moments"])
        if arm in ("omit_anchor_latch", "omit_all_event_actions"):
            anchor = actions[arm]["anchor"][0]["after"]
            expected["surprise"]["anchor_event"] = clone(anchor["surprise"]["anchor_event"])
            expected["surprise"]["anchor_events"] = anchor["surprise"]["anchor_events"]
            expected["controller"]["data_drive"] = anchor["controller"]["data_drive"]
        if arm in ("omit_ladder_reopens", "omit_all_event_actions"):
            for key in GROUPS:
                expected["groups"][key]["tester"] = clone(begin["before"]["groups"][key]["tester"])
                expected["groups"][key]["raw_scale"] = begin["before"]["groups"][key]["raw_scale"]
            table_scale = expected["groups"]["0.2"]["raw_scale"]
            for key, group in expected["groups"].items():
                group["lr"] = group["base_lr"] * (max(group["raw_scale"], .75 * table_scale) if key == "1.0" else group["raw_scale"])
                if key == "1.0":
                    group["lr"] /= 1. + expected["controller"]["payoff_error"] ** 2
        equal(begin["after"], expected, "complete immediate action difference exactly declared; detector/guard unchanged " + arm)
    native_small = {}
    for step in (6221, 6222, 6223, 6387, 6388):
        path = directory / "native" / f"game-{step:05d}.json"
        artifact(path)
        value = read(path)
        require(value["step"] == step, "fixed temporal private game step")
        audit_game(value, data, "native-small" + str(step), full=False)
        native_small[str(step)] = {pool: {judge: value["pools"][pool][judge]["g_game"] for judge in JUDGES} for pool in COUNTS}
    comparisons = {arm: paired_changes(games["native"], games[arm]) for arm in ARMS if arm != "native"}
    for name in ("plan.json", "receipt.json", "status.json"):
        artifact(directory / name)
    return dict(schema="supra_late_native_reopen_independent_cpu_review_v1", qualified=True, gpu_used=False,
        run=str(directory), native_pin=PIN, checks=CHECKS, fixed_final_step=6400,
        diagnostic_source_sha256=DRIVER_SHA, immutable_inputs_verified=len(inputs),
        source_snapshot_files_verified=len(plan["application_source_sha256"]) + len(plan["native_source_sha256"]),
        baseline_full_native_exact=True, baseline_native_digest=state_digest(finals["native"]),
        baseline_exact_native_updates=400, laws={arm: law for arm, (_, law) in ARMS.items()},
        action_contracts=action_reports, temporal_native_small_games=native_small,
        nonserialized_ka2_decay_cache=cross_fork_cache,
        paired_learned_games=comparisons, output_metrics_used_for_selection=False,
        artifact_sha256=artifacts, reviewer_sha256=sha(__file__),
        reviewer_dependency_sha256={"scripts/review_e22_supra_pr223.py": sha(ROOT / "scripts/review_e22_supra_pr223.py")},
        limits="One fixed trajectory. Native guards/optimizer decisions may diverge after intervention as consequences. CPU independently qualifies stored owners, exact action arithmetic, sampled streams, and raw game reductions; GPU replay and forward scores remain source-pinned runtime witnesses. Neither any LR rise nor a negative held-out game change is required for qualification. No production-law or ordinary-LoRA win claim.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "outputs/e22-particle-gated-late-reopen-causal")
    parser.add_argument("--particlegan-root", type=Path, default=ROOT.parent / "ParticleGAN-pr223-supra")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(8)
    result = build_review(args.run, args.particlegan_root)
    path = args.output or args.run / "independent-review.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(qualified=True, checks=result["checks"], output=str(path),
                         paired_learned_games=result["paired_learned_games"]), allow_nan=False), flush=True)


if __name__ == "__main__":
    main()

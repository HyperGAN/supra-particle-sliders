#!/usr/bin/env python3
"""Qualify the fixed PR223/archived-f459 Supra comparison using CPU artifacts.

This review never evaluates a model, advances training, or selects a winner.
Training-state differences and either sign of the measured game change are
valid results. Only the declared matched-input and evidence contracts must pass.
"""
from __future__ import annotations

from argparse import ArgumentParser
from collections import defaultdict
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import tarfile

import torch

ROOT = Path(__file__).resolve().parents[1]
PIN = "bdf05d1be0f68cfdb0c71e81e7e0d3cce477572f"
OLD_PIN = "f459cb6d6aaaabeb1af076ec53ad7a963618de90"
PROFILE = "pr223_shared_routed_v1"
STEPS = (400, 800, 1200, 1600)
COUNTS = {"fit": 240, "test": 240, "holds": 30, "preservation": 60}
JUDGES = ("fixed_start_D", "arm_final_D")


class ReviewError(ValueError):
    pass


def require(condition, label):
    if not condition:
        raise ReviewError(label)


def read(path):
    return json.loads(Path(path).read_text())


def canonical(value):
    return json.loads(json.dumps(value))


def diagnostic_value(value):
    """Match reporting of undefined acquisition statistics without state edits."""
    if isinstance(value, torch.Tensor):
        return diagnostic_value(value.detach().cpu().tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: diagnostic_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [diagnostic_value(item) for item in value]
    return value


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def state_digest(value):
    """Independent native content digest; device is deliberately not encoded."""
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(str(key).encode())
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode())
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode())

    visit(value)
    return digest.hexdigest()


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False, mmap=True)


def finite(value, label="root"):
    if isinstance(value, float):
        require(math.isfinite(value), f"nonfinite {label}")
    elif isinstance(value, dict):
        for key, item in value.items():
            finite(item, f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            finite(item, f"{label}[{index}]")


def close(actual, expected, label, tolerance=1e-12):
    require(math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance),
            f"{label}: {actual!r} != {expected!r}")


def committed_python_hashes(repository, pin):
    contents = subprocess.check_output(["git", "-C", str(repository), "archive", pin, "particlegan"])
    with tarfile.open(fileobj=io.BytesIO(contents)) as archive:
        return {member.name: hashlib.sha256(archive.extractfile(member).read()).hexdigest()
                for member in archive.getmembers() if member.isfile() and member.name.endswith(".py")}


def trace_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def frozen_parameters(state):
    """Both immutable FAST/averaged tensors, including conditioning buffers."""
    policy = state["policy"]
    return {family: {role: {name: tensor for name, tensor in policy[family][role].items()
                           if not policy["requires_grad"][role].get(name, False)}
                     for role in ("generator", "encoder")}
            for family in ("models", "averages")}


def audit_text_contexts(state, data, label):
    for family in ("models", "averages"):
        encoder = state["policy"][family]["encoder"]
        require(torch.equal(encoder["contexts"], data["text_contexts"])
                and torch.equal(encoder["masks"], data["text_masks"]), f"{label} immutable {family} text contexts/masks")


def owner_comparison(old, new):
    """Report common-owner equality; differences are evidence, never failures."""
    first, second = old["policy"], new["policy"]
    common = {key: state_digest(first[key]) == state_digest(second[key])
              for key in first.keys() & second.keys() if key != "recipe"}
    return dict(common_owner_equal=common,
                different_common_owners=sorted(key for key, equal in common.items() if not equal),
                only_new_policy_fields=sorted(second.keys() - first.keys()),
                only_old_policy_fields=sorted(first.keys() - second.keys()),
                recipe_equal=canonical(first["recipe"]) == canonical(second["recipe"]),
                application_rng_equal={key: state_digest(old[key]) == state_digest(new[key])
                                       for key in ("data_rng", "paired_noise_rng")},
                model_role_equal={key: state_digest(first["models"][key]) == state_digest(second["models"][key])
                                  for key in first["models"].keys() & second["models"].keys()})


def rate_report(state):
    """Raw settling scales and actually applied LRs are distinct evidence."""
    policy = state["policy"]
    table_scales = [tester["s"] for index, (testers, roles) in enumerate(zip(policy["lr_settle"], policy["roles"]))
                    if index != 1 for tester, role in zip(testers, roles)
                    if role == "table" and tester is not None and tester["s"] is not None]
    floor = .75 * max(table_scales) if table_scales else None
    groups, contracted = [], {}
    for index, (optimizer, bases, roles, testers) in enumerate(zip(
            policy["optimizers"], policy["initial_lrs"], policy["roles"], policy["lr_settle"])):
        for group_index, (group, base, role, tester) in enumerate(zip(optimizer["param_groups"], bases, roles, testers)):
            scale = tester["s"] if tester is not None else 1.
            if scale is not None and scale < 1. and role in {"generator", "encoder", "router", "critic"}:
                contracted[f"{index}.{group_index}"] = dict(role=role, scale=scale)
            groups.append(dict(optimizer=index, group=group_index, role=role, base_lr=base,
                raw_tester_scale=scale, saved_effective_lr=group["lr"], saved_effective_scale=group["lr"] / base,
                critic_table_support_floor=floor if index == 1 else None))
    return dict(groups=groups, raw_contracted_network=contracted,
                configured_network_lr_floor=policy["recipe"]["network_lr_floor"],
                interpretation="Settled guard reads raw network tester scales. Stationarity D LR uses max(rawD,.75*tableScale), then critic payoff damping. Saved LRs were applied at begin_step; raw scales include end-of-step observations. The recipe network_lr_floor is not applied in this stationarity branch.")


def audit_config(state, initial, data, step):
    config, policy = state["config"], state["policy"]
    require(policy["completed_steps"] == step, f"checkpoint horizon {step}")
    require(state_digest(config) == state_digest(initial["config"]), f"configuration drift at {step}")
    require(canonical(config["recipe"]) == canonical(policy["recipe"]), f"native/application recipe drift at {step}")
    require(config["particle_profile"] == PROFILE and config["architecture"] == "linear_modulated_v2",
            f"profile/architecture changed at {step}")
    require(config["bank"] == [128, 4] and tuple(policy["table"].shape) == (128, 4), f"particle bank changed at {step}")
    require(len(config["sites"]) == 71 and tuple(config["sites"]) == tuple(policy["routing"]["config"]["sites"]),
            f"sequential routing sites changed at {step}")
    require(policy["table_requires_grad"] is True and policy["birth_death"] is None
            and policy["row_evidence"] is None and policy["routing"] is not None, f"routed ownership changed at {step}")
    require(config["output_error_guard"] is False and config["max_feature_context_harm"] == 0.
            and policy["routing"]["config"]["max_context_harm"] == 0., f"native feature guard changed at {step}")
    require(config["checkpoint_selection"] == "final fixed update horizon; evaluation only", "checkpoint-selection boundary")
    require(policy["recipe"]["birth_death_backend"] == "auto" and policy["recipe"]["reopen_guard"] == "settled",
            f"shared recipe changed at {step}")
    backend = policy["backend_selection"]
    require(backend["actual_backend"] == backend["sampling_backend"] == "routed"
            and backend["selection_reason"] == "routed_rows_owns_controls"
            and backend["generator_noise_factor"] == 1., f"active backend changed at {step}")
    require(policy["reopen_guard"] is not None, f"missing settled guard at {step}")
    require(torch.equal(policy["models"]["critic"]["scale"], data["coordinate_scale"]), f"critic scale changed at {step}")


def audit_trace(rows, old, data, final):
    require(len(rows) == len(old) == 1600, "complete fixed1600 traces required")
    require([row["step"] for row in rows] == list(range(1, 1601)), "new native update clock")
    require([row["step"] for row in old] == list(range(1, 1601)), "old native update clock")
    finite(rows, "new trace")
    finite(old, "old trace")
    stream = torch.Generator().manual_seed(7)
    paired = ("batch_indices", "base_noise_sums", "paired_rng_digest", "hold", "game_weight")
    for row, control in zip(rows, old):
        step = row["step"]
        hold = step % 5 == 0
        pool = data["holds" if hold else "fit"]["context"]
        indices = torch.randint(len(pool), (4,), generator=stream).tolist()
        require(row["batch_indices"] == control["batch_indices"] == indices, f"CPU data stream at {step}")
        require(row["hold"] is hold and row["game_weight"] == (.1 if hold else 1.), f"preservation schedule at {step}")
        require(all(row[key] == control[key] for key in paired), f"matched sampling/paired stream at {step}")
        require(row["penalty_calls"] == step and row["output_sigma"] > 0 and row["bank_grad_norm"] >= 0,
                f"native penalty clock/noise/gradient at {step}")
    require(torch.equal(stream.get_state(), final["data_rng"]), "saved final CPU data stream")
    same = [row["step"] for row, control in zip(rows, old) if row == control]
    changed = [row["step"] for row, control in zip(rows, old) if row != control]
    return dict(updates=1600, preservation_updates=sum(row["hold"] for row in rows),
                data_and_paired_streams_matched=True, all_values_finite=True,
                exact_old_trace_rows=len(same), first_different_step=changed[0] if changed else None,
                different_trace_rows=len(changed),
                dv12_rng_digests_equal_steps=sum(row["dv12_rng_digest"] == control["dv12_rng_digest"]
                                                  for row, control in zip(rows, old)),
                new_accepted_moves=sum(bool(row.get("move") and row["move"].get("moves", 0)) for row in rows),
                old_accepted_moves=sum(bool(row.get("move") and row["move"].get("moves", 0)) for row in old))


def audit_evaluation(evaluation, data, label):
    require(set(evaluation) == set(COUNTS), f"{label} complete evaluation pools")
    finite(evaluation, label)
    for pool, count in COUNTS.items():
        result, context = evaluation[pool], data[pool]["context"]
        records = result["records"]
        require(result["count"] == len(records) == len(context) == count, f"{label}/{pool} count")
        require([row["index"] for row in records] == list(range(count)), f"{label}/{pool} record order")
        for index, row in enumerate(records):
            require(row["source_caption_id"] == int(context[index, 4097]) and row["time"] == float(context[index, 4096]),
                    f"{label}/{pool}/{index} context identity")
            require(row["mse"] >= 0 and row["teacher_rms"] > 0, f"{label}/{pool}/{index} invalid diagnostic magnitude")
            close(row["rmse"] ** 2, row["mse"], f"{label}/{pool}/{index} RMSE", 2e-7)
            close(row["relative_rmse"], row["rmse"] / max(row["teacher_rms"], 1e-12),
                  f"{label}/{pool}/{index} relative RMSE", 2e-7)
            for judge in JUDGES:
                scores = row[judge]
                close(scores["g_game"] - scores["d_game"], scores["score_gap"],
                      f"{label}/{pool}/{index}/{judge} paired logistic identity", 2e-6)
        mse = sum(row["mse"] for row in records) / count
        teacher_power = sum(row["teacher_rms"] ** 2 for row in records) / count
        close(result["rmse"], math.sqrt(mse), f"{label}/{pool} RMSE aggregate")
        close(result["relative_rms"], math.sqrt(mse / teacher_power), f"{label}/{pool} relative RMS aggregate")
        for judge in JUDGES:
            for metric in ("g_game", "d_game", "score_gap"):
                close(result[judge][metric], sum(row[judge][metric] for row in records) / count,
                      f"{label}/{pool}/{judge}/{metric} mean")


def evaluation_comparison(old, new, data):
    comparisons, pairing = {}, {}
    for pool, count in COUNTS.items():
        first, second = old[pool], new[pool]
        comparisons[pool] = dict(count=count, old_rmse=first["rmse"], new_rmse=second["rmse"],
                                 rmse_change=second["rmse"] - first["rmse"])
        groups = defaultdict(list)
        for a, b in zip(first["records"], second["records"]):
            require(all(a[key] == b[key] for key in ("index", "source_caption_id", "time", "teacher_rms")),
                    f"{pool} endpoint pairing")
            groups[a["source_caption_id"]].append((a, b))
        for judge in JUDGES:
            comparisons[pool][judge] = dict(old_g_game=first[judge]["g_game"], new_g_game=second[judge]["g_game"],
                                             new_minus_old=second[judge]["g_game"] - first[judge]["g_game"])
        pairing[pool] = {}
        for judge in JUDGES:
            changes = [b[judge]["g_game"] - a[judge]["g_game"] for a, b in zip(first["records"], second["records"])]
            subjects = {str(key): dict(source_prompt=data["prompts"][key], contexts=len(records),
                mean_game_new_minus_old=sum(b[judge]["g_game"] - a[judge]["g_game"] for a, b in records) / len(records),
                contexts_improved=sum(b[judge]["g_game"] < a[judge]["g_game"] for a, b in records))
                for key, records in groups.items()}
            pairing[pool][judge] = dict(contexts=count, mean_game_new_minus_old=sum(changes) / count,
                contexts_improved=sum(value < 0 for value in changes), contexts_equal=sum(value == 0 for value in changes),
                contexts_worsened=sum(value > 0 for value in changes), subjects=subjects)
    return comparisons, pairing


def rolling_losses(rows, window=100):
    result = {}
    for name, hold in (("edit", False), ("preservation", True)):
        selected = [row for row in rows if row["hold"] is hold][-window:]
        if selected:
            result[name] = dict(updates=len(selected), first_step=selected[0]["step"], last_step=selected[-1]["step"],
                g_game=sum(row["loss_g"] / row["game_weight"] for row in selected) / len(selected),
                d_game=sum(row["loss_d_game"] / row["game_weight"] for row in selected) / len(selected),
                penalty=sum(row["penalty"] for row in selected) / len(selected),
                bank_grad_norm=sum(row["bank_grad_norm"] for row in selected) / len(selected))
    return result


def audit_progress(directory, rows, old_rows, data, initial, plan):
    stream = torch.Generator().manual_seed(72)
    panels = {}
    for name, pool_name, per_subject in (("edit", "fit", 2), ("preservation", "holds", 4)):
        pool = data[pool_name]["context"]
        indices = []
        for subject in pool[:, 4097].unique(sorted=True):
            available = (pool[:, 4097] == subject).nonzero().flatten()
            offsets = torch.linspace(0, len(available) - 1, min(per_subject, len(available))).round().long()
            indices.extend(available[offsets].tolist())
        noise = torch.randn((4, len(indices), 256, 16), generator=stream)
        panels[name] = dict(contexts=len(indices), indices=indices, gaussian_cpu72_digest=state_digest(noise),
                            context_digest=state_digest(pool[indices]))
        for location in [directory] + [directory / f"control-probes-{step}" for step in STEPS]:
            require(read(location / f"progress-{name}-indices.json") == indices, f"private probe indices {location}/{name}")
    control = read(directory / "comparison-control-progress.json")
    progress = trace_rows(directory / "progress.jsonl")
    require([point["step"] for point in control] == list(STEPS), "control progress horizons")
    require([point["step"] for point in progress] == [2] + list(range(200, 1601, 200)), "new progress horizons")
    require(read(directory / "progress-latest.json") == progress[-1], "latest progress copy")
    for label, points, training_rows in (("new", progress, rows), ("old", control, old_rows)):
        for point in points:
            finite(point, f"{label} progress")
            require(point["native_state_unchanged"] is True and point["output_metrics_used"] is False
                    and point["evaluation_only"] is True and point["output_sigma"] == .125
                    and point["source"] == "FAST current particle G", f"{label} probe boundary at {point['step']}")
            require(point["rolling"] == rolling_losses(training_rows[:point["step"]]), f"{label} rolling game normalization")
            require(set(point["probes"]) == set(panels), f"{label} probe pools")
            for name, scores in point["probes"].items():
                require(scores["contexts"] == panels[name]["contexts"], f"{label} probe count")
        for step in STEPS:
            if label == "old":
                require(trace_rows(directory / f"control-probes-{step}/progress.jsonl") == [points[list(STEPS).index(step)]],
                        f"control probe copy at {step}")
    by_step = {point["step"]: point for point in progress}
    comparison = {str(old["step"]): {name: {mode: dict(
        old_game=old["probes"][name][mode]["frozen_start_D"],
        new_game=by_step[old["step"]]["probes"][name][mode]["frozen_start_D"],
        new_minus_old=by_step[old["step"]]["probes"][name][mode]["frozen_start_D"] - old["probes"][name][mode]["frozen_start_D"])
        for mode in ("clean", "dv12")} for name in panels} for old in control}
    # Construction is checked against frozen executed source, not an unrecorded
    # claim that CPU and CUDA generators produce interchangeable random panels.
    runner = (directory / "source/scripts/experiment_e22_supra_pr223.py").read_text()
    monitor = (directory / "source/scripts/monitor_e22_supra_particle_convergence.py").read_text()
    evaluator = (directory / "source/scripts/diagnose_e22_supra_training_controls.py").read_text()
    require(runner.count("monitor.dv12_state = common_dv12.clone()") == 2
            and 'initial["policy"]["streams"]["noise_generator"]' in runner, "common private DV12 construction")
    require("stream = torch.Generator().manual_seed(72)" in monitor and "record=False" in monitor
            and 'before != state_digest(checkpoint(loop))' in monitor, "read-only private progress construction")
    require("stream = torch.Generator(device=device).manual_seed(72)" in evaluator
            and "torch.randn((4, len(context), 256, 16)" in evaluator, "common private CUDA final panels")
    return dict(primary_judge=plan["fixed_critic_digest"], comparison=comparison, panels=panels,
                common_private_dv12_state_digest=state_digest(initial["policy"]["streams"]["noise_generator"]),
                progress_native_state_unchanged_runtime_witnesses=len(control) + len(progress),
                final_gaussian_panels="private CUDA generator72 reset independently in each endpoint evaluation",
                panel_evidence_limit="CPU progress panels and indices are reconstructed. GPU scores and private CUDA panels are recorded source/runtime witnesses, not CPU model reruns.")


def build_review(directory):
    directory = Path(directory).resolve()
    plan, run, receipt, status = (read(directory / name) for name in ("plan.json", "run.json", "receipt.json", "status.json"))
    require(receipt["qualified"] is True and all(value is True for value in receipt["checks"].values()), "GPU qualification checks")
    require(receipt["plan"] == plan, "final receipt plan")
    require({key: value for key, value in plan.items() if key != "fixed_critic_digest"}
            == {key: value for key, value in run.items() if key != "config"}, "run provenance; only judge digest may be added")
    require(plan["schema"] == "supra_pr223_matched_1600_v1" and plan["fixed_updates"] == 1600
            and plan["start_step"] == 2 and plan["comparison_steps"] == list(STEPS)
            and plan["progress_every"] == 200, "predeclared fixed budget/report horizons")
    require(plan["particlegan_commit"] == PIN and plan["control_commit"] == OLD_PIN and plan["profile"] == PROFILE,
            "truthful ParticleGAN source/profile labels")
    require(plan["selection"] == "predeclared1600 horizon; no output-based stopping, hyperparameter or checkpoint selection"
            and plan["output_metrics"] == "final evaluation only", "output metric boundary")
    require(status["phase"] == "complete" and status["step"] == status["steps"] == 1600
            and status["backend"] == "routed", "completed fixed-horizon status")
    inputs = {name: Path(path) for name, path in plan["input_paths"].items()}
    require(set(inputs) == set(plan["input_sha256"]), "complete input hash manifest")
    for name, path in inputs.items():
        require(sha(path) == plan["input_sha256"][name], f"immutable input {name}")
    for name, expected in plan["application_source_sha256"].items():
        require(sha(directory / "source" / name) == expected, f"application source snapshot {name}")
    initial_directory = inputs["shared_initial"].parent
    smoke, smoke_review = (read(initial_directory / name) for name in ("result.json", "independent-review.json"))
    require(smoke["qualified"] is True and all(value is True for value in smoke["checks"].values())
            and smoke_review["qualified"] is True and all(value is True for value in smoke_review["checks"].values()),
            "qualified fresh shared smoke")
    require(sha(initial_directory / "result.json") == smoke_review["result_sha256"]
            and sha(inputs["shared_initial"]) == smoke_review["checkpoint_sha256"], "smoke review byte witnesses")
    pg_repository = Path(smoke["plan"]["imported_particlegan"]).parent.parent
    require(committed_python_hashes(pg_repository, PIN) == plan["particlegan_source_sha256"], "PR223 pinned git source tree")
    for name, expected in plan["particlegan_source_sha256"].items():
        require(sha(directory / "source/particlegan" / name) == expected, f"PR223 source snapshot {name}")
    old_directory = Path(plan["control_directory"])
    old_run, old_review, old_receipt = (read(old_directory / name) for name in ("run.json", "qualification-review.json", "receipt.json"))
    require(old_review["all_checks_passed"] is True and old_run["particlegan_commit"] == OLD_PIN
            and old_run["fixed_updates"] == old_review["fixed_updates"] == 1600, "qualified f459 historical control")
    for name, expected in old_review["file_hashes"].items():
        require(sha(old_directory / name) == expected, f"historical qualified artifact {name}")
    old_manifest = read(old_directory / "source/sha256.json")
    require(old_manifest == old_receipt["plan"]["source_sha256"], "historical source manifest")
    for name, expected in old_manifest.items():
        require(sha(old_directory / "source" / name) == expected, f"historical application snapshot {name}")
    old_package = Path(old_run["particlegan_root"]) / "particlegan"
    old_hashes = {path.name: sha(path) for path in sorted(old_package.glob("*.py"))}
    require(state_digest(old_hashes) == old_run["particlegan_source_digest"], "historical native source digest")
    old_committed = committed_python_hashes(pg_repository, OLD_PIN)
    require(old_hashes == {Path(name).name: value for name, value in old_committed.items()
                           if Path(name).parent == Path("particlegan")}, "historical f459 git source tree")
    data = torch.load(inputs["data"], map_location="cpu", weights_only=False)
    initial, old_initial = load(inputs["shared_initial"]), load(old_directory / "checkpoint-00002.pt")
    require(state_digest(initial) == smoke["checkpoint_native_digest"], "qualified shared step2 native digest")
    require(state_digest(data) == initial["config"]["dataset_digest"] == old_initial["config"]["dataset_digest"], "dataset content digest")
    require(initial["policy"]["completed_steps"] == old_initial["policy"]["completed_steps"] == 2, "fresh step2 horizons")
    initial_comparison = owner_comparison(old_initial, initial)
    require(all(initial_comparison["common_owner_equal"].values()) and all(initial_comparison["application_rng_equal"].values()),
            "common fresh step2 tensors/optimizers/controllers/RNGs exact")
    require(initial_comparison["common_owner_equal"] == plan["initial_common_owners_exact"], "recorded warm-initialization provenance")
    require(initial_comparison["only_new_policy_fields"] == ["backend_selection", "reopen_guard"]
            and not initial_comparison["only_old_policy_fields"], "only declared additional initialization state")
    old_config, new_config = canonical(old_initial["config"]), canonical(initial["config"])
    recipe_diff = {key for key in old_config["recipe"].keys() | new_config["recipe"].keys()
                   if old_config["recipe"].get(key) != new_config["recipe"].get(key)}
    require(recipe_diff == {"birth_death_backend", "reopen_guard", "birth_death_cells", "birth_death_metric_rank",
                            "birth_death_chunk", "birth_death_parent_policy"}, "declared shared recipe-only changes")
    require({key: value for key, value in old_config.items() if key != "recipe"}
            == {key: value for key, value in new_config.items() if key not in ("recipe", "particle_profile")},
            "same V2 architecture/physics/application settings")
    require(canonical(initial["config"]) == run["config"], "run initial config provenance")
    audit_config(initial, initial, data, 2)
    audit_text_contexts(initial, data, "initial")
    audit_text_contexts(old_initial, data, "old initial")
    judge = load(inputs["judge"])
    require(judge["policy"]["completed_steps"] == 1856
            and state_digest(judge["policy"]["models"]["critic"]) == plan["fixed_critic_digest"], "common frozen1856 judge")
    final, old_final = load(directory / "final.pt"), load(inputs["old_1600"])
    require(sha(directory / "final.pt") == receipt["final_checkpoint_sha256"]
            and state_digest(final) == receipt["final_native_digest"], "final native checkpoint witness")
    require(state_digest(old_final) == receipt["old_final_native_digest"] == old_review["final_native_state_digest_verified"],
            "historical native endpoint witness")
    frozen = state_digest(frozen_parameters(initial))
    require(state_digest(frozen_parameters(final)) == state_digest(frozen_parameters(old_final)) == frozen,
            "frozen host/teacher unchanged in both arms")
    audit_text_contexts(final, data, "new final")
    audit_text_contexts(old_final, data, "old final")
    rows, old_rows = trace_rows(directory / "train.jsonl"), trace_rows(inputs["old_trace"])
    require(rows[:2] == smoke["rows"][:2] == old_rows[:2], "qualified fresh trace prefix")
    trace = audit_trace(rows, old_rows, data, final)
    require(trace["exact_old_trace_rows"] == receipt["exact_old_trace_rows"] == status["exact_old_trace_rows"],
            "independently recomputed exact-row count")
    comparisons = {}
    for step in STEPS:
        new, old = load(directory / f"checkpoint-{step:05d}.pt"), load(inputs[f"old_{step}"])
        audit_config(new, initial, data, step)
        require(old["policy"]["completed_steps"] == step and old["config"] == old_initial["config"], f"historical snapshot {step}")
        require(state_digest(frozen_parameters(new)) == state_digest(frozen_parameters(old)) == frozen,
                f"immutable FAST/averaged tensors and buffers changed at {step}")
        audit_text_contexts(new, data, f"new checkpoint{step}")
        audit_text_contexts(old, data, f"old checkpoint{step}")
        comparison = owner_comparison(old, new)
        require(all(comparison["application_rng_equal"].values()), f"matched saved data/paired streams at {step}")
        comparison["new_native_digest"] = state_digest(new)
        comparison["old_native_digest"] = state_digest(old)
        comparison["native_rates"] = {"old": rate_report(old), "new": rate_report(new)}
        comparisons[str(step)] = comparison
        if step == 1600:
            require(comparison["new_native_digest"] == receipt["final_native_digest"], "last checkpoint equals fixed final")
    final_comparison = comparisons["1600"]
    require({key: final_comparison["common_owner_equal"][key] for key in receipt["model_and_owner_equal"]}
            == receipt["model_and_owner_equal"], "independent model/optimizer/controller equality report")
    guard = final["policy"]["reopen_guard"]
    require(diagnostic_value(guard) == receipt["controller"]["reopen_guard"] == status["guard_activity"]["reopen_guard"], "saved settled guard witness")
    require(guard["epoch_rebases"] == receipt["controller"]["epoch_rebases"], "settled epoch count witness")
    require(final["policy"]["surprise"]["fires"] == receipt["controller"]["optimizer_surprise_fires"]
            and final["policy"]["surprise"]["anchor_events"] == receipt["controller"]["anchor_release_events"], "saved reopen counts")
    require(final["policy"]["routing"]["counters"]["moves"] == receipt["controller"]["accepted_moves"], "saved structural move count")
    require(rate_report(final)["raw_contracted_network"] == receipt["controller"]["contracted_network"],
            "recorded raw network contraction witness")
    progress = audit_progress(directory, rows, old_rows, data, initial, plan)
    old_metrics, new_metrics = (read(directory / f"evaluation-{arm}.json") for arm in ("old", "new"))
    audit_evaluation(old_metrics, data, "old")
    audit_evaluation(new_metrics, data, "new")
    metrics, pairing = evaluation_comparison(old_metrics, new_metrics, data)
    require(metrics == receipt["comparison"] == status["comparison"], "recomputed game/RMSE comparison")
    require(receipt["evaluation_state_unchanged"] is True and receipt["checks"]["new_evaluation_read_only"] is True
            and receipt["checks"]["old_evaluation_read_only"] is True, "final evaluation runtime state witnesses")
    require(receipt["critic_labels"] == {"fixed_start_D": "common frozen V2 cabe step1856", "arm_final_D": "common PR223 final1600"},
            "same two critics applied to both endpoints")
    return dict(schema="supra_pr223_matched_1600_independent_cpu_review_v1", qualified=True, gpu_used=False,
        run=str(directory), fixed_updates=1600, source_pins={"old": OLD_PIN, "new": PIN}, profile=PROFILE,
        source_snapshot_files_verified=len(old_manifest) + len(plan["application_source_sha256"]) + len(plan["particlegan_source_sha256"]),
        source_git_trees_verified=True, immutable_inputs_verified=len(inputs), dataset_content_digest=state_digest(data),
        initialized_common_state=initial_comparison, trace=trace, checkpoints=comparisons,
        model_and_owner_equal=receipt["model_and_owner_equal"], guarded_metadata={"backend_selection": final["policy"]["backend_selection"],
            "reopen_guard": diagnostic_value(guard), "guarded_policy_digest": state_digest(guard),
            "surprise_state_equal_old": final_comparison["common_owner_equal"]["surprise"]},
        progress=progress, comparison=metrics, paired_context_scores=pairing,
        scoring_critic_digests={"common_frozen1856": plan["fixed_critic_digest"],
                               "common_new_final1600": state_digest(final["policy"]["models"]["critic"])},
        final_served_sources={"old": old_final["policy"]["served_source"], "new": final["policy"]["served_source"]},
        final_native_digest=receipt["final_native_digest"], evaluation_state_unchanged_runtime_witnesses=True,
        output_metrics_used_for_selection=False,
        limits=["CPU review verifies persisted tensors, raw records, source snapshots and runtime receipts; it does not rerun GPU forwards or 1600 native updates.",
                "Matched cached f459 trajectory is a historical control admitted by exact fresh step2 common-state replay, not a newly rerun old optimizer arm.",
                "Progress compares FAST G under frozen1856 D; final evaluation uses each native served source under the same1856 and same newfinal1600 critics.",
                "The two endpoint final evaluations reset the same private CUDA72 panels; progress instead uses private CPU72 panels and common private DV12.",
                "One task/stream with correlated contexts. Game judges are endogenous diagnostics; output error is reporting only and does not select a checkpoint.",
                "The active backend is routed. This test does not activate Atlas feature cells or claim superiority to ordinary LoRA."],
        artifact_sha256={name: sha(directory / name) for name in ("plan.json", "run.json", "receipt.json", "train.jsonl", "final.pt",
                                                                  "evaluation-old.json", "evaluation-new.json", "progress.jsonl")},
        reviewer_sha256=sha(Path(__file__)))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "outputs/e22-pr223-shared-1600")
    parser.add_argument("--output", type=Path, help="Default: RUN/independent-review.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    output = args.output or args.run / "independent-review.json"
    require(not output.exists(), "preserve existing review evidence; choose a new output")
    report = build_review(args.run)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(qualified=True, gpu_used=False, output=str(output), trace=report["trace"],
                         model_and_owner_equal=report["model_and_owner_equal"], comparison=report["comparison"])), flush=True)


if __name__ == "__main__":
    main()

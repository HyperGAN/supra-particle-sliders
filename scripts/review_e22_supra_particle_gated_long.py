#!/usr/bin/env python3
"""CPU qualification of versioned editing-only V3 and both fixed endpoints.

Verifies source/input provenance, checkpoint ownership, sampling, all stored
570-context game/ablation records, all three declared ordinary LoRA references,
and clean export tensors. GPU forward/replay checks remain runtime witnesses.
Either sign of a measured game change is a valid result.
"""
import argparse
from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path
import sys

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.review_e22_supra_particle_gated import (
    COUNTS, JUDGES, PIN, PROFILE, V2, V3, audit_contributions, audit_hookup,
    close, load, read, require, rows, sha,
)
from scripts.review_e22_supra_pr223 import (
    audit_evaluation, audit_text_contexts, canonical, committed_python_hashes,
    finite, frozen_parameters, rolling_losses, state_digest,
)
from scripts.e22_supra_historical_control import reconstruct

ORIGINALS = {
    "converged_1600": (1600, "original1600", "0c97f2f2fd05bf17be99c901a87a05447f0448ad6edae06ecba4e5d623fed997"),
    "published_1600": (1600, "originalpublished1600", "134f5a9d12206b74f39a2c3f9c90fdb38477a1c3204b5deea5152b1cbe49a47f"),
    "converged_6400": (6400, "original6400", "24a98ba055d0b5298136d67dc7491a339684da03416c510d1d443fe730eb7069"),
}


def paired_report(old_eval, new_eval, comparison, *, label, ordinary=False):
    """Recompute every paired aggregate and subject result from raw records."""
    result = {}
    old_prefix = "original" if ordinary else "control"
    delta_name = "new_minus_original" if ordinary else "new_minus_control"
    for pool, count in COUNTS.items():
        require(comparison[pool]["count"] == count, label + "/" + pool + " count")
        close(comparison[pool][old_prefix + "_rmse"], old_eval[pool]["rmse"], label + "/" + pool + " reference diagnostic")
        close(comparison[pool]["gated_rmse"], new_eval[pool]["rmse"], label + "/" + pool + " V3 diagnostic")
        close(comparison[pool]["rmse_change_evaluation_only"], new_eval[pool]["rmse"] - old_eval[pool]["rmse"],
              label + "/" + pool + " diagnostic difference")
        pairs = list(zip(old_eval[pool]["records"], new_eval[pool]["records"]))
        require(len(pairs) == count, label + "/" + pool + " raw count")
        for old, new in pairs:
            require(all(old[key] == new[key] for key in ("index", "source_caption_id", "time", "teacher_rms")),
                    label + "/" + pool + " same context/teacher")
        result[pool] = {}
        for judge in JUDGES:
            changes = [new[judge]["g_game"] - old[judge]["g_game"] for old, new in pairs]
            close(comparison[pool][judge][old_prefix + "_g_game"], old_eval[pool][judge]["g_game"],
                  label + "/" + pool + "/" + judge + " reference game")
            close(comparison[pool][judge]["gated_g_game"], new_eval[pool][judge]["g_game"],
                  label + "/" + pool + "/" + judge + " V3 game")
            close(comparison[pool][judge][delta_name], sum(changes) / count,
                  label + "/" + pool + "/" + judge + " paired game difference")
            subjects = defaultdict(list)
            for old, new in pairs:
                subjects[str(old["source_caption_id"])].append(new[judge]["g_game"] - old[judge]["g_game"])
            result[pool][judge] = dict(mean_game_new_minus_reference=sum(changes) / count,
                contexts_improved=sum(value < 0 for value in changes), contexts_worsened=sum(value > 0 for value in changes),
                subjects={subject: dict(contexts=len(values), mean_game_new_minus_reference=sum(values) / len(values))
                          for subject, values in subjects.items()})
    return result


def continuation_contributions(report, data):
    """Reuse the qualified reduction checks with an explicit judge-key mapping.

The second critic here is the declared control's final critic (1600 or6400),
not D400. The temporary key only adapts the old reviewer schema; original
record labels and the returned report retain their actual critic identity.
"""
    require(set(report) == set(COUNTS), "all 570 contribution contexts")
    adjusted = deepcopy(report)
    def rename(value):
        if isinstance(value, dict):
            if "common_control_final_D" in value:
                require("common_D400" not in value, "unambiguous contribution final judge")
                value["common_D400"] = value.pop("common_control_final_D")
            for item in value.values():
                rename(item)
        elif isinstance(value, list):
            for item in value:
                rename(item)
    rename(adjusted)
    result = audit_contributions(adjusted, data)
    for pool in result.values():
        for arm in ("code_removal", "mass_only"):
            pool[arm]["common_control_final_D"] = pool[arm].pop("common_D400")
    return result


def review_export(directory, receipt, state):
    source = state["policy"]["served_source"]
    selected = state["policy"]["models" if source == "fast" else "averages"]
    expected = {"generator." + name: tensor for name, tensor in selected["generator"].items()
                if state["policy"]["requires_grad"]["generator"].get(name, False)}
    expected.update({"router." + name: tensor for name, tensor in selected["router"].items() if name != "log_mass"})
    expected.update({"bank.table": state["policy"]["table" if source == "fast" else "averaged_table"],
                     "bank.log_mass": selected["router"]["log_mass"]})
    export = directory / "final.safetensors"
    require(sha(export) == receipt["final_export_sha256"] and len(expected) == 428, "clean export file/schema")
    with safe_open(str(export), framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
        config = json.loads(metadata["config"])
        require(metadata["format"] == "supra_particlegan_clean_v3" and config["architecture"] == V3,
                "explicit V3 clean export formula")
        require(config == receipt["export"]["config"] and config["completed_steps"] == state["policy"]["completed_steps"]
                and config["served_source"] == source and config["sites"] == state["config"]["sites"],
                "clean export horizon/source/sites")
        require(config["rank"] == 16 and config["z_dim"] == 4 and config["num_particles"] == 128
                and config["cfg"] == 3 and config["sampling"] == "clean"
                and config["routed_geometry"] == "mass_atoms_v1", "export dimensions/native routing")
        require(json.loads(metadata["native_pins"])["backend_sha256"] == read(ROOT / "backend.lock.json")["sha256"],
                "export backend source pin")
        require(set(handle.keys()) == set(expected), "all 428 export names")
        for name, tensor in expected.items():
            value = handle.get_tensor(name)
            require(value.dtype == tensor.dtype == torch.float32 and torch.equal(value, tensor), "clean export tensor " + name)


def build_review(directory, pg_root):
    directory = Path(directory).resolve()
    plan, run, receipt, status = [read(directory / name) for name in ("plan.json", "run.json", "receipt.json", "status.json")]
    horizon, start = plan["fixed_updates"], plan["start_step"]
    require(plan["schema"] == "supra_particle_gated_v3_editing_only_long_v1" and horizon == 6400
            and start == 1600 and plan["architecture"] == V3
            and plan["profile"] == PROFILE and plan["particlegan_commit"] == PIN, "predeclared continuation law/profile/horizon")
    require(receipt["qualified"] is True and all(value is True for value in receipt["checks"].values()), "GPU runtime qualification")
    require(receipt["plan"] == plan and status["phase"] == "complete" and status["step"] == status["steps"] == horizon,
            "complete fixed-horizon receipt")
    require({key: run[key] for key in plan} == plan and run["fixed_updates"] == horizon, "run/plan provenance")
    config = run["config"]
    require(plan["training_schedule"] == config["training_schedule"] == "editing_only_v1"
            and plan["schedule_start_step"] == config["schedule_start_step"] == 1600
            and plan["endpoint_steps"] == [5440, 6400] and plan["endpoint_edit_updates"] == {"5440": 5120, "6400": 6080}
            and plan["inherited_edit_updates"] == 1280 and plan["inherited_preservation_updates"] == 320
            and plan["historical_control_edit_updates"] == 5120 and plan["historical_control_preservation_updates"] == 1280,
            "predeclared versioned schedule and editing budgets")
    require(config["architecture"] == V3 and config["particle_profile"] == PROFILE
            and config["max_feature_context_harm"] == 0 and config["output_error_guard"] is False,
            "V3 strict learned-feature guards")
    require(config["preservation_game_weight"] == .1 and config["penalty_weight_unchanged"] is True
            and config["probe_interval"] == 100 and config["checkpoint_selection"] == "final fixed update horizon; evaluation only",
            "unchanged native game/probe/selection contract")
    require(config["recipe"]["birth_death_backend"] == "auto" and config["recipe"]["reopen_guard"] == "settled", "shared native recipe")
    for name, digest in plan["application_source_sha256"].items():
        require(sha(directory / "source" / name) == sha(ROOT / name) == digest, "application source " + name)
    committed = committed_python_hashes(pg_root, PIN)
    require(plan["particlegan_source_sha256"] == committed, "native commit tree")
    for name, digest in committed.items():
        require(sha(directory / "source/native" / name) == sha(Path(pg_root) / name) == digest, "native source " + name)
    require(run["particlegan_source_digest"] == state_digest(committed), "native source digest")
    inputs = {name: Path(path) for name, path in plan["input_paths"].items()}
    for name, path in inputs.items():
        require(sha(path) == plan["input_sha256"][name], "immutable input " + name)
    initial_receipt, initial_review, control_review, common_review, common_result = [read(inputs[name]) for name in
        ("initial_receipt", "initial_review", "control_review", "common_initial_review", "common_initial_result")]
    require(initial_receipt["qualified"] and initial_review["qualified"]
            and control_review.get("qualified", control_review.get("all_checks_passed", False))
            and common_review["qualified"] and common_result["qualified"], "qualified inherited references")
    require(plan["input_sha256"]["initial_trace"] == initial_review["artifact_sha256"]["train.jsonl"], "qualified inherited trace link")
    require(plan["input_sha256"]["control_trace"] == control_review["artifact_sha256"]["train.jsonl"], "qualified control suffix trace link")
    initial, control, final, common = [load(inputs[name] if name != "final" else directory / "final.pt")
                                      for name in ("initial", "control", "final", "common_initial")]
    require(state_digest(initial) == receipt["initial_native_digest"] == initial_receipt["final_native_digest"]
            == initial_review["final_native_digest"] and sha(inputs["initial"]) == initial_receipt["final_checkpoint_sha256"],
            "qualified initial full content")
    require(state_digest(control) == receipt["control_native_digest"] == control_review["final_native_digest"], "qualified control full content")
    require(state_digest(final) == receipt["final_native_digest"] and sha(directory / "final.pt") == receipt["final_checkpoint_sha256"],
            "final full native content/file")
    require(state_digest(common) == common_result["checkpoint_native_digest"]
            and sha(inputs["common_initial"]) == common_review["checkpoint_sha256"], "qualified common private-monitor DV12 stream")
    require(initial["policy"]["completed_steps"] == start and control["policy"]["completed_steps"]
            == final["policy"]["completed_steps"] == horizon, "native clocks")
    expected_config = deepcopy(initial["config"])
    expected_config.update(training_schedule="editing_only_v1", schedule_start_step=1600,
        pre_switch_edit_updates=1280, pre_switch_preservation_updates=320,
        sampling="original CPU generator 7; preservation every fifth update through1600; editing only after1600")
    require(expected_config == final["config"] and canonical(config) == canonical(final["config"]), "explicit V3 schedule is the only config change")
    scheduled = load(directory / "schedule-01600.pt")
    require(scheduled["config"] == expected_config and scheduled["policy"]["completed_steps"] == 1600
            and state_digest({key: value for key, value in initial.items() if key != "config"})
            == state_digest({key: value for key, value in scheduled.items() if key != "config"}),
            "schedule opt-in preserves every native owner and RNG exactly")
    require(sha(directory / "schedule-01600.pt") == receipt["schedule_checkpoint_sha256"], "tagged native boundary file")
    require(control["config"]["architecture"] == V2
            and control["config"].get("particle_profile", "pr155_routed") == plan["control_profile"], "declared V2 control profile")
    control_provenance = read(inputs["control_provenance"])
    require(control_provenance["particlegan_commit"] == plan["control_particlegan_commit"], "control source/profile provenance")
    if horizon == 1600:
        require(plan["control_profile"] == PROFILE and plan["control_particlegan_commit"] == PIN,
                "1600 control matched shared source/profile")
        require(canonical({key: value for key, value in config.items() if key != "architecture"})
                == canonical({key: value for key, value in control["config"].items() if key != "architecture"}),
                "1600 only generator law differs")
    data = load(inputs["data"])
    require(state_digest(data) == initial["config"]["dataset_digest"] == control["config"]["dataset_digest"]
            == common["config"]["dataset_digest"], "same cached task data")
    frozen = state_digest(frozen_parameters(initial))
    def audit_game_scale(state, label):
        expected_scale = data["coordinate_scale"]
        fast_scale = state["policy"]["models"]["critic"]["scale"]
        anchor_scale = state["policy"]["optimizers"][1]["regularizer"]["ema"]["scale"]
        require(torch.equal(fast_scale, expected_scale) and torch.equal(anchor_scale, expected_scale)
                and torch.equal(fast_scale, initial["policy"]["models"]["critic"]["scale"]),
                label + " fixed FAST critic/KA2 anchor coordinate scale")
    checkpoint_digests = {}
    checkpoint_steps = sorted(set(range(start + 400, horizon + 1, 400)) | {5440})
    for step in checkpoint_steps:
        path = directory / f"checkpoint-{step:05d}.pt"
        state = load(path)
        require(state["policy"]["completed_steps"] == step and state["config"] == expected_config, f"checkpoint{step} clock/config")
        require(state_digest(frozen_parameters(state)) == frozen, f"checkpoint{step} four frozen owners")
        audit_text_contexts(state, data, f"checkpoint{step}")
        audit_game_scale(state, f"checkpoint{step}")
        checkpoint_digests[str(step)] = dict(native_digest=state_digest(state), file_sha256=sha(path))
    for label, state in (("initial", initial), ("control", control), ("final", final), ("commonstep2", common)):
        require(state_digest(frozen_parameters(state)) == frozen, label + " four immutable owners")
        audit_text_contexts(state, data, label)
        audit_game_scale(state, label)
    require(state_digest(load(directory / f"checkpoint-{horizon:05d}.pt")) == state_digest(final), "final/checkpoint boundary equality")
    require(initial["policy"]["roles"] == final["policy"]["roles"] == control["policy"]["roles"]
            and initial["policy"]["recipe"] == final["policy"]["recipe"], "unchanged V3 native role ownership/recipe")
    if horizon == 1600:
        require(final["policy"]["recipe"] == control["policy"]["recipe"], "1600 common native recipe")
    guard = final["policy"]["routing"]
    require(guard["config"]["max_context_harm"] == 0 and guard["config"].get("output_error_guard", False) is False,
            "native routed per-context zero-feature-harm state")
    control_rows, control_chain = reconstruct(inputs, data, torch=torch, state_digest=state_digest,
        sha=sha, committed_python_hashes=committed_python_hashes, repository=pg_root, require=require)
    recorded_chain = read(directory / "control-chain.json")
    derived_trace = rows(directory / "control-trace-full.jsonl")
    require(control_rows == derived_trace and sha(directory / "control-trace-full.jsonl")
            == recorded_chain["reconstructed_trace_sha256"] == plan["control_reconstructed_trace_sha256"],
            "all6400 derived control rows/hash")
    require({key: value for key, value in recorded_chain.items() if key != "reconstructed_trace_sha256"} == control_chain
            and plan["control_trace_parts"] == control_chain["control_trace_parts"]
            and plan["control_program_digest"] == control_chain["complete_recorded_program_digest"],
            "exact qualified historical chain and program digest")
    actual_rows, inherited_rows = rows(directory / "train.jsonl"), rows(inputs["initial_trace"])
    require(len(actual_rows) == len(control_rows) == horizon and len(inherited_rows) == start
            and actual_rows[:start] == inherited_rows, "complete fixed-horizon trace with exact inherited prefix")
    finite(actual_rows, "training trace")
    stream = torch.Generator().manual_seed(7)
    for index, (row, reference) in enumerate(zip(actual_rows, control_rows), 1):
        hold = index <=1600 and index % 5 == 0
        pool = data["holds" if hold else "fit"]["context"]
        ids = torch.randint(len(pool), (4,), generator=stream).tolist()
        require(row["step"] == index and row["hold"] is hold and row["game_weight"] == (.1 if hold else 1.)
                and row["batch_indices"] == ids, f"native sampling/preservation at{index}")
        fields = ("batch_indices", "base_noise_sums", "paired_rng_digest", "hold", "game_weight") if index <=1600 else (
            "base_noise_sums", "paired_rng_digest")
        require(all(row[name] == reference[name] for name in fields),
                f"matched paired input streams at{index}")
        require(row["penalty_calls"] == index and row["dense_gradient_rows"] == (0 if index == 1 else 128), f"penalty/dense bank at{index}")
    require(torch.equal(final["data_rng"], stream.get_state())
            and torch.equal(final["paired_noise_rng"], control["paired_noise_rng"]), "final fit sampling and common Gaussian stream")
    edited_program = [{key: row[key] for key in ("step", "hold", "game_weight", "batch_indices")} for row in actual_rows[1600:]]
    require(edited_program == read(directory / "editing-sampling-program.json")
            and sha(directory / "editing-sampling-program.json") == plan["editing_sampling_program_sha256"]
            and state_digest(edited_program) == plan["editing_sampling_program_digest"], "predeclared4800 fit-only draws")
    require(sum(not row["hold"] for row in actual_rows) == 6080 and sum(row["hold"] for row in actual_rows) == 320,
            "actual full cumulative editing/preservation budget")
    dv12_matches = sum(row["dv12_rng_digest"] == reference["dv12_rng_digest"] for row, reference in zip(actual_rows, control_rows))
    require(dv12_matches == receipt["dv12_streams_equal"], "recorded DV12 equality count")
    moves = final["policy"]["routing"]["counters"]["moves"]
    require(moves == receipt["controller"]["accepted_moves"], "native accepted moves")
    require(final["policy"]["surprise"]["fires"] == receipt["controller"]["optimizer_surprise_fires"]
            and final["policy"]["surprise"]["anchor_events"] == receipt["controller"]["anchor_release_events"]
            and final["policy"]["reopen_guard"]["epoch_rebases"] == receipt["controller"]["epoch_rebases"],
            "saved native guard/reopen counters")
    if moves == 0:
        require(dv12_matches == horizon, "no-move matched native DV12 program")
    judge = load(inputs["judge"])
    require(judge["policy"]["completed_steps"] == 1856
            and state_digest(judge["policy"]["models"]["critic"]) == plan["critic_digests"]["D1856"], "same frozen D1856")
    if "scoring_critic_digests" in control_review:
        require(plan["critic_digests"]["D1856"] == control_review["scoring_critic_digests"]["common_frozen1856"], "qualified common-critic chain")
    require(state_digest(control["policy"]["models"]["critic"]) == plan["critic_digests"]["control_final"], "same control-final critic")
    old_eval, new_eval = read(directory / "evaluation-v2.json"), read(directory / "evaluation-v3.json")
    audit_evaluation(old_eval, data, "control")
    audit_evaluation(new_eval, data, "gated")
    paired = paired_report(old_eval, new_eval, receipt["comparison"], label="V2 control")
    require(status["comparison"] == receipt["comparison"] and status["original_comparison"] == receipt["original_comparison"],
            "dashboard final comparison copies")
    provenance = read(inputs["data_provenance"])
    from supra.runtime import MODEL_ID, MODEL_REV, T5_REV, VAE_REV, TARGETS
    expected_ordinary = {name.removeprefix("teacher."): tensor for name, tensor in
        initial["policy"]["models"]["encoder"].items() if name.startswith("teacher.")
        and name.endswith((".down.weight", ".up.weight"))}
    require(len(expected_ordinary) == 142 and all(not bool(tensor.count_nonzero())
            for name, tensor in expected_ordinary.items() if name.endswith(".up.weight")),
            "original-reference host reconstructed from pure teacher")
    original_report = {}
    ordinary_tensors, ordinary_evaluations = {}, {}
    require(set(receipt["original_comparison"]) == set(plan["ordinary_reference_metadata"]) == set(ORIGINALS), "all three declared ordinary targets")
    for label, (step, input_name, digest) in ORIGINALS.items():
        path = inputs[input_name]
        require(sha(path) == plan["input_sha256"][input_name] == digest, label + " declared original SHA")
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            metadata = {key: json.loads(value) for key, value in (handle.metadata() or {}).items()}
            expected = dict(format="supra-native-lora-v1", rank=16, alpha=16, targets=list(TARGETS), model_id=MODEL_ID,
                model_revision=MODEL_REV, text_encoder_revision=T5_REV, vae_revision=VAE_REV,
                step=step, prompts_sha256=provenance["prompts_sha256"])
            require(metadata == plan["ordinary_reference_metadata"][label]
                    and all(metadata.get(key) == value for key, value in expected.items()), label + " metadata/task/source pins")
            require(set(handle.keys()) == set(expected_ordinary), label + " exact142 ordinary adapter names")
            for key, tensor in expected_ordinary.items():
                value = handle.get_tensor(key)
                require(value.shape == tensor.shape and value.dtype == tensor.dtype == torch.float32
                        and bool(torch.isfinite(value).all()), label + "/" + key + " ordinary shape/dtype/finiteness")
            ordinary_tensors[label] = {key: handle.get_tensor(key) for key in handle.keys()}
        evaluation = read(directory / f"evaluation-original-{label}.json")
        ordinary_evaluations[label] = evaluation
        audit_evaluation(evaluation, data, label)
        original_report[label] = dict(path=str(path), checkpoint_step=step, sha256=digest,
            same_total_update_budget=step == horizon, original_edit_updates=step * 4 // 5,
            v3_edit_updates=6080, same_editing_budget=False,
            artifact_label="published artifact; trajectory not inferred from file path"
                if label == "published_1600" else "converged trajectory",
            adapter_tensor_digest=state_digest(ordinary_tensors[label]),
            paired_games=paired_report(evaluation, new_eval, receipt["original_comparison"][label], label=label, ordinary=True))
    ordinary_identity = {}
    for first, tensors in ordinary_tensors.items():
        equal_to = [second for second, other in ordinary_tensors.items() if set(tensors) == set(other)
                    and all(torch.equal(value, other[key]) for key, value in tensors.items())]
        ordinary_identity[first] = equal_to
        original_report[first]["same_adapter_tensors_as"] = equal_to
        for second in equal_to:
            require(ordinary_evaluations[first] == ordinary_evaluations[second], "identical adapter tensors give identical recorded evaluation")
    endpoint_reports = {}
    require(set(receipt["endpoints"]) == {"5440", "6400"} and status["endpoints"] == receipt["endpoints"],
            "both predeclared complete endpoint receipts")
    for step, edit_updates in ((5440, 5120), (6400, 6080)):
        point = read(directory / f"endpoint-{step:05d}.json")
        evaluation_path = directory / f"evaluation-v3-step-{step:05d}.json"
        evaluation = read(evaluation_path)
        audit_evaluation(evaluation, data, f"fixed endpoint{step}")
        require(point == receipt["endpoints"][str(step)] and point["step"] == step
                and point["editing_updates"] == edit_updates and point["preservation_updates"] == 320
                and point["training_schedule"] == "editing_only_v1", f"endpoint{step} counts/schedule/copy")
        require(point["native_digest"] == checkpoint_digests[str(step)]["native_digest"]
                and point["checkpoint_sha256"] == checkpoint_digests[str(step)]["file_sha256"]
                and point["evaluation_sha256"] == sha(evaluation_path), f"endpoint{step} checkpoint/evaluation linkage")
        require(point["native_state_unchanged"] and point["evaluation_only"]
                and point["output_metrics_used_for_selection"] is False, f"endpoint{step} read-only common-game contract")
        prefix = actual_rows[:step]
        require(sum(not row["hold"] for row in prefix) == edit_updates and sum(row["hold"] for row in prefix) == 320,
                f"endpoint{step} actual native edit counts")
        endpoint_reports[str(step)] = dict(total_updates=step, editing_updates=edit_updates, preservation_updates=320,
            same_editing_budget_as_historical6400=edit_updates == 5120,
            same_total_update_budget_as_historical6400=step == 6400,
            paired_v2_games=paired_report(old_eval, evaluation, point["comparison"], label=f"endpoint{step}/V2"),
            paired_ordinary_games={label: paired_report(ordinary_evaluations[label], evaluation,
                point["original_comparison"][label], label=f"endpoint{step}/{label}", ordinary=True) for label in ORIGINALS})
        if step == 6400:
            require(evaluation == new_eval and point["comparison"] == receipt["comparison"]
                    and point["original_comparison"] == receipt["original_comparison"], "final endpoint is exact complete final evaluation")
    require(status["editing_updates"] == 6080 and status["preservation_updates"] == 320
            and status["training_schedule"] == "editing_only_v1", "final live budget card")
    contributions = continuation_contributions(read(directory / "particle-contribution.json"), data)
    hookup = audit_hookup(read(directory / "audit-final.json"), horizon, config["sites"])
    progress = rows(directory / "progress.jsonl")
    require([item["step"] for item in progress] == sorted(set(range(start + 200, horizon + 1, 200)) | {5440}), "fixed progress horizons plus5440")
    progress_panels = {}
    probe_stream = torch.Generator().manual_seed(72)
    for name, pool_name, per_subject in (("edit", "fit", 2), ("preservation", "holds", 4)):
        pool = data[pool_name]["context"]
        indices = []
        for subject in pool[:, 4097].unique(sorted=True):
            available = (pool[:, 4097] == subject).nonzero().flatten()
            offsets = torch.linspace(0, len(available) - 1, min(per_subject, len(available))).round().long()
            indices.extend(available[offsets].tolist())
        require(read(directory / f"progress-{name}-indices.json") == indices, "fixed " + name + " monitor subjects/contexts")
        gaussian = torch.randn((4, len(indices), 256, 16), generator=probe_stream)
        progress_panels[name] = dict(contexts=len(indices), context_digest=state_digest(pool[indices]),
            gaussian_cpu72_digest=state_digest(gaussian), indices=indices)
    for item in progress:
        finite(item, "fixed game progress")
        require(item["native_state_unchanged"] and item["evaluation_only"] and item["output_metrics_used"] is False,
                "read-only game progress")
        require(item["source"] == "FAST current particle G" and item["output_sigma"] == .125,
                "common progress units/panel sigma")
        require(item["rolling"] == rolling_losses(actual_rows[:item["step"]]), "independent unweighted progress losses")
        for name in progress_panels:
            require(item["probes"][name]["contexts"] == progress_panels[name]["contexts"], "progress probe count")
    require(read(directory / "progress-latest.json") == progress[-1], "progress latest copy")
    review_export(directory, receipt, final)
    artifacts = ["plan.json", "run.json", "receipt.json", "status.json", "train.jsonl", "final.pt", "final.safetensors",
        "evaluation-v2.json", "evaluation-v3.json", "particle-contribution.json", "audit-final.json", "progress.jsonl"]
    artifacts += [f"evaluation-original-{label}.json" for label in ORIGINALS]
    artifacts += [f"checkpoint-{step:05d}.pt" for step in checkpoint_steps]
    artifacts += ["schedule-01600.pt", "control-chain.json", "control-trace-full.jsonl", "editing-sampling-program.json"]
    artifacts += [name for step in (5440, 6400) for name in
        (f"endpoint-{step:05d}.json", f"evaluation-v3-step-{step:05d}.json")]
    runtime_names = ("initial_strict_full_native_restore", "two_update_rows_exact", "two_update_full_native_replay_exact",
        "v3_clean_export_reload_exact", "control_strict_full_native_restore", "all_v3_evaluation_native_state_immutable",
        "v3_full_native_state_immutable_after_control_eval", "schedule_opt_in_changes_config_only",
        "schedule_opt_in_exact_config", "tagged_schedule_strict_cpu_restore", "initial_native_unchanged_after_reference_preflight")
    guard_activity = dict(surprise_fires=final["policy"]["surprise"]["fires"],
        surprise_log=final["policy"]["surprise"]["log"], anchor_release_events=final["policy"]["surprise"]["anchor_events"],
        epoch_rebases=final["policy"]["reopen_guard"]["epoch_rebases"],
        controller_reopens=final["policy"]["controller"].get("reopens"),
        accepted_moves=moves, lr_settling_owners=[dict(player=player, group=group, role=role,
            raw_scale=settler.get("s"), counts=settler.get("counts"))
            for player, (roles, settlers) in enumerate(zip(final["policy"]["roles"], final["policy"]["lr_settle"]))
            for group, (role, settler) in enumerate(zip(roles, settlers)) if settler is not None])
    return dict(schema="supra_particle_gated_v3_editing_only_long_cpu_review_v1", qualified=True, gpu_used=False,
        run=str(directory), start_step=start, fixed_updates=horizon, particlegan_pin=PIN, profile=PROFILE,
        control_profile=plan["control_profile"], control_particlegan_commit=plan["control_particlegan_commit"],
        source_snapshot_files_verified=len(plan["application_source_sha256"]) + len(committed), immutable_inputs_verified=len(inputs),
        frozen_four_owner_digest=frozen, fixed_game_coordinate_scale_digest=state_digest(data["coordinate_scale"]),
        checkpoints=checkpoint_digests,
        native_resume_runtime_witnesses={name: receipt["checks"][name] for name in runtime_names},
        trace=dict(updates=horizon, inherited_exact_updates=start, editing_updates=6080, preservation_updates=320,
            historical_prefix_contexts_matched=1600, future_fit_draws_replayed_exactly=True,
            paired_gaussian_streams_matched=True, dv12_matches=dv12_matches, accepted_moves=moves),
        paired_games=paired, ordinary_references=original_report, ordinary_tensor_identity=ordinary_identity,
        contributions=contributions, hookup=hookup, endpoints=endpoint_reports,
        private_progress_panels=progress_panels, guard_activity=guard_activity, control_chain=control_chain,
        final_native_digest=receipt["final_native_digest"], export_tensors=428,
        artifact_sha256={name: sha(directory / name) for name in artifacts}, reviewer_sha256=sha(__file__),
        reviewer_dependency_sha256={name: sha(ROOT / "scripts" / name) for name in
            ("review_e22_supra_particle_gated.py", "review_e22_supra_pr223.py", "e22_supra_historical_control.py")},
        output_metrics_used_for_selection=False,
        limits="CPU checks sources, immutable inputs/native checkpoints, full qualified historical control chain, fit-only schedule, and every raw game/ablation aggregate. CUDA BF16 forwards, actual gradient gates, two-update native recovery, frozen common Gaussian panels, original-reference serving and clean reload are audited source/runtime witnesses, not CPU reruns. Historical V2 differs in source/profile and sampling;5440 matches5120 editing updates,6400 has6080 editing updates. The320 preservation prefix is retained, with no future preservation training. One task/stream does not establish general superiority.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "outputs/e22-particle-gated-v3-editing-only-6400")
    parser.add_argument("--particlegan-root", type=Path, default=ROOT.parent / "ParticleGAN-pr223-supra")
    parser.add_argument("--output", type=Path, help="fresh review path; default is RUN/independent-review.json")
    args = parser.parse_args()
    output = args.output if args.output is not None else args.run / "independent-review.json"
    if output.exists():
        parser.error("preserve the existing review; choose a fresh --output path")
    torch.set_num_threads(4)
    result = build_review(args.run, args.particlegan_root)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("qualified", "gpu_used", "start_step", "fixed_updates", "paired_games", "ordinary_references", "trace")}), flush=True)


if __name__ == "__main__":
    main()

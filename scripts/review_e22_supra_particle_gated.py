#!/usr/bin/env python3
"""CPU artifact qualification of the fixed-400 native particle-gate experiment.

Recomputes source/input/state digests, frozen owners, trace sampling, every
stored endpoint/ablation aggregation and export tensors. CUDA BF16 forwards,
fresh-prefix parity and exact resumed updates remain executed runtime witnesses.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.review_e22_supra_pr223 import (
    audit_evaluation, audit_text_contexts, committed_python_hashes, finite,
    frozen_parameters, state_digest, canonical,
)

PIN = "bdf05d1be0f68cfdb0c71e81e7e0d3cce477572f"
V2, V3 = "linear_modulated_v2", "gated_particle_v3"
PROFILE = "pr223_shared_routed_v1"
COUNTS = dict(fit=240, test=240, holds=30, preservation=60)
JUDGES = ("fixed_start_D", "arm_final_D")


class ReviewError(RuntimeError):
    pass


def require(value, message):
    if not value:
        raise ReviewError(message)


def close(actual, expected, message, tolerance=1e-12):
    require(math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance),
            f"{message}: {actual!r} != {expected!r}")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False, mmap=True)


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def audit_contributions(report, data):
    require(set(report) == set(COUNTS), "all 570 contribution contexts")
    result = {}
    for pool, expected_count in COUNTS.items():
        item, context = report[pool], data[pool]["context"]
        require(item["contexts"] == expected_count and item["gaussian_panels"] == 4
                and item["output_sigma"] == .125, f"{pool} contribution pool/panels/sigma")
        arms = item["per_arm"]
        require(set(arms) == {"clean", "zero_particle_codes", "mass_only_routing"}, f"{pool} ablation arms")
        clean = arms["clean"]["records"]
        expected_subjects = {str(int(value)) for value in context[:, 4097].unique()}
        for arm, panel in arms.items():
            finite(panel, f"{pool}/{arm}")
            records = panel["records"]
            require(panel["contexts"] == len(records) == expected_count, f"{pool}/{arm} complete records")
            require(set(panel["per_subject"]) == expected_subjects, f"{pool}/{arm} subjects")
            for index, row in enumerate(records):
                require(row["index"] == index and row["source_caption_id"] == int(context[index, 4097])
                        and row["time"] == float(context[index, 4096]), f"{pool}/{arm}/{index} source/time")
                require(row["output_change_mse_evaluation_only"] >= 0, f"{pool}/{arm}/{index} diagnostic power")
                for judge in ("common_D1856", "common_D400"):
                    value, original = row[judge], clean[index][judge]
                    require(len(value["per_draw_g_game"]) == len(value["per_draw_g_game_delta_from_clean"]) == 4,
                            f"{pool}/{arm}/{index}/{judge} four panels")
                    close(value["g_game"], sum(value["per_draw_g_game"]) / 4,
                          f"{pool}/{arm}/{index}/{judge} panel average", 2e-7)
                    close(value["g_game"] - value["d_game"], value["score_gap"],
                          f"{pool}/{arm}/{index}/{judge} logistic identity", 2e-6)
                    panel_deltas = value["per_draw_g_game_delta_from_clean"]
                    close(value["g_game_delta_from_clean"], sum(panel_deltas) / 4,
                          f"{pool}/{arm}/{index}/{judge} paired panel mean", 2e-7)
                    # The CUDA helper averages paired float32 differences.
                    # Subtracting its two separately rounded means is a
                    # different reduction. Bound only that rounding effect;
                    # every underlying paired float32 subtraction is exact.
                    reduction_bound = 4 * torch.finfo(torch.float32).eps * (
                        abs(value["g_game"]) + abs(original["g_game"])) + 1e-12
                    require(abs(value["g_game_delta_from_clean"] - (value["g_game"] - original["g_game"]))
                            <= reduction_bound, f"{pool}/{arm}/{index}/{judge} rounded-mean relation")
                    for draw in range(4):
                        paired_float32 = float(torch.tensor(value["per_draw_g_game"][draw], dtype=torch.float32)
                                               - torch.tensor(original["per_draw_g_game"][draw], dtype=torch.float32))
                        require(value["per_draw_g_game_delta_from_clean"][draw] == paired_float32,
                                f"{pool}/{arm}/{index}/{judge}/draw{draw} exact float32 subtraction")

            def summarize(summary, selected, label):
                count = len(selected)
                require(summary["contexts"] == count and count > 0, label + " count")
                close(summary["output_change_rms_evaluation_only"],
                      math.sqrt(sum(row["output_change_mse_evaluation_only"] for row in selected) / count), label + " RMS")
                for judge in ("common_D1856", "common_D400"):
                    values = summary[judge]
                    for metric in ("g_game", "d_game", "score_gap", "g_game_delta_from_clean"):
                        close(values[metric], sum(row[judge][metric] for row in selected) / count, label + "/" + metric)
                    for draw in range(4):
                        for metric in ("per_draw_g_game", "per_draw_g_game_delta_from_clean"):
                            close(values[metric][draw], sum(row[judge][metric][draw] for row in selected) / count,
                                  label + "/" + metric + "/" + str(draw))
                    changes = [row[judge]["g_game_delta_from_clean"] for row in selected]
                    for metric, condition in (("contexts_game_worsened", lambda x: x > 0),
                                              ("contexts_game_improved", lambda x: x < 0),
                                              ("contexts_game_unchanged", lambda x: x == 0)):
                        require(values[metric] == sum(condition(x) for x in changes), label + "/" + metric)
            summarize(panel, records, f"{pool}/{arm}")
            for subject, summary in panel["per_subject"].items():
                selected = [row for row in records if str(row["source_caption_id"]) == subject]
                summarize(summary, selected, f"{pool}/{arm}/subject{subject}")
        result[pool] = dict(contexts=expected_count, subjects=len(expected_subjects),
            gaussian_panels=4, paired_noise_sha256=item["paired_noise_sha256"],
            code_removal={judge: arms["zero_particle_codes"][judge] for judge in ("common_D1856", "common_D400")},
            mass_only={judge: arms["mass_only_routing"][judge] for judge in ("common_D1856", "common_D400")})
    return result


def audit_hookup(audit, step, sites):
    finite(audit, f"audit{step}")
    require(audit["step"] == step and audit["architecture"] == V3
            and audit["native_state_rng_and_gradients_unchanged"] is True, f"audit{step} boundary/runtime immutability")
    require(set(audit["branches"]) == set(audit["per_site_interventions"]) == set(sites), f"audit{step} 71sites")
    for site, branch in audit["branches"].items():
        require(branch["formula_exact"] is True and branch["code_effect_at_fixed_input_rms"] > 0
                and 0 <= branch["gate_saturation_fraction"] <= 1 and branch["particle_gate_rms"] <= 1,
                f"audit{step}/{site} bounded actual gate/dependence")
        close(branch["code_effect_over_total_delta"], branch["code_effect_at_fixed_input_rms"] /
              max(branch["total_adapter_delta_rms"], 1e-30), f"audit{step}/{site} ratio")
    for mode in ("clean_gradient", "dv12_gradient"):
        gradient = audit[mode]
        require(gradient["bank_rows_nonzero"] == 128 and gradient["router_tensors_nonzero"] == 142
                and gradient["generator_tensors_nonzero"] == 284, f"audit{step}/{mode} native dense owners")
        norms = gradient["parameter_gradient_norms"]
        require(sum(name.startswith("generator.") and value > 0 for name, value in norms.items()) == 284
                and sum(name.startswith("router.") and value > 0 for name, value in norms.items()) == 142
                and norms["bank.table"] > 0, f"audit{step}/{mode} positive individual gradients")
    count = sum(value["output_change_rms_evaluation_only"] > 0 for value in audit["per_site_interventions"].values())
    require(count == audit["final_output_site_effects_nonzero_diagnostic"], f"audit{step} BF16 dependence count")
    require(all(value["output_change_rms_evaluation_only"] >= 0 for value in audit["per_site_interventions"].values()),
            f"audit{step} diagnostic power")
    return dict(native_sites=len(sites), dense_bank_rows=128, router_tensors=142, generator_tensors=284,
                per_site_final_output_effects_nonzero_diagnostic=count, runtime_state_rng_gradients_unchanged=True)


def build_review(directory, pg_root):
    directory = Path(directory).resolve()
    plan, run, receipt, status = [read(directory / name) for name in ("plan.json", "run.json", "receipt.json", "status.json")]
    require(receipt["qualified"] is True and all(value is True for value in receipt["checks"].values()), "GPU runtime qualification")
    require(receipt["plan"] == plan and status["phase"] == "complete" and status["step"] == 400, "complete matched receipt")
    require(plan["schema"] == "supra_particle_gated_v3_matched_400_v1" and plan["fixed_updates"] == 400
            and plan["architectures"] == [V2, V3] and plan["profile"] == PROFILE
            and plan["particlegan_commit"] == PIN, "predeclared architecture/profile/horizon/pin")
    require({key: run[key] for key in plan} == plan and run["fixed_updates"] == 400, "run/plan provenance")
    config = run["config"]
    require(config["architecture"] == V3 and config["particle_profile"] == PROFILE
            and config["max_feature_context_harm"] == 0 and config["output_error_guard"] is False,
            "explicit V3 and zero-feature-harm native guards")
    require(config["preservation_game_weight"] == .1 and config["penalty_weight_unchanged"] is True
            and config["probe_interval"] == 100 and config["checkpoint_selection"] == "final fixed update horizon; evaluation only",
            "native game/probe/selection contract")
    require(config["recipe"]["birth_death_backend"] == "auto" and config["recipe"]["reopen_guard"] == "settled",
            "shared native recipe")
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
    control_review, initial_review, teacher_review = [read(inputs[name]) for name in
                                                     ("control_review", "initial_review", "teacher_review")]
    require(control_review["qualified"] and initial_review["qualified"] and teacher_review["qualified"], "qualified references")
    require(plan["input_sha256"]["control_trace"] == control_review["artifact_sha256"]["train.jsonl"], "qualified trace link")
    require(plan["input_sha256"]["initial"] == initial_review["checkpoint_sha256"], "qualified initial link")
    control = load(inputs["control_checkpoint"])
    initial = load(inputs["initial"])
    two, final = load(directory / "checkpoint-00002.pt"), load(directory / "final.pt")
    require(state_digest(control) == receipt["control_native_digest"] == control_review["checkpoints"]["400"]["new_native_digest"],
            "qualified shared control400")
    require(state_digest(final) == receipt["final_native_digest"] and sha(directory / "final.pt") == receipt["final_checkpoint_sha256"],
            "final full native content/file digests")
    require(control["policy"]["completed_steps"] == final["policy"]["completed_steps"] == 400
            and two["policy"]["completed_steps"] == initial["policy"]["completed_steps"] == 2, "native checkpoint clocks")
    require(two["config"] == final["config"] and canonical(config) == canonical(final["config"]), "native V3 configs")
    require(canonical({key: value for key, value in config.items() if key != "architecture"})
            == canonical({key: value for key, value in control["config"].items() if key != "architecture"}), "only generator law differs")
    data = load(inputs["data"])
    require(state_digest(data) == config["dataset_digest"], "cached data contents")
    frozen = state_digest(frozen_parameters(control))
    for label, state in (("initial", initial), ("v3step2", two), ("v3final", final)):
        require(state_digest(frozen_parameters(state)) == frozen, label + " four immutable model/encoder owners")
        audit_text_contexts(state, data, label)
    for state in (two, final):
        require(state["policy"]["roles"] == control["policy"]["roles"]
                and state["policy"]["recipe"] == control["policy"]["recipe"], "native roles/recipe unchanged")
        guard = state["policy"]["routing"]
        require(guard["config"]["max_context_harm"] == 0 and guard["config"].get("output_error_guard", False) is False,
                "native routed feature guard state")
    require(final["policy"]["served_source"] in ("fast", "averaged"), "served source")
    actual_rows, control_rows = rows(directory / "train.jsonl"), rows(inputs["control_trace"])[:400]
    require(len(actual_rows) == len(control_rows) == 400, "fixed400 native traces")
    finite(actual_rows, "training trace")
    data_stream = torch.Generator().manual_seed(7)
    for index, (row, reference) in enumerate(zip(actual_rows, control_rows), 1):
        hold = index % 5 == 0
        pool = data["holds" if hold else "fit"]["context"]
        ids = torch.randint(len(pool), (4,), generator=data_stream).tolist()
        require(row["step"] == index and row["hold"] is hold and row["game_weight"] == (.1 if hold else 1.)
                and row["batch_indices"] == ids, f"native CPU sampling/preservation at{index}")
        require(all(row[name] == reference[name] for name in
                    ("batch_indices", "base_noise_sums", "paired_rng_digest", "hold", "game_weight")), f"matched streams at{index}")
        require(row["penalty_calls"] == index and row["dense_gradient_rows"] == (0 if index == 1 else 128),
                f"native penalty/dense bank at{index}")
    require(torch.equal(final["data_rng"], data_stream.get_state()), "saved final CPU data stream")
    require(torch.equal(final["data_rng"], control["data_rng"])
            and torch.equal(final["paired_noise_rng"], control["paired_noise_rng"]), "final application stream equality")
    dv12_matches = sum(row["dv12_rng_digest"] == reference["dv12_rng_digest"] for row, reference in zip(actual_rows, control_rows))
    require(receipt["dv12_streams_equal"] == dv12_matches, "DV12 equality count")
    accepted = final["policy"]["routing"]["counters"]["moves"]
    require(accepted == receipt["controller"]["accepted_moves"], "native accepted moves")
    if accepted == 0:
        require(dv12_matches == 400 and state_digest(final["policy"]["streams"]) == state_digest(control["policy"]["streams"]),
                "unchanged no-move DV12 draw program")
    judge = load(inputs["judge"])
    require(state_digest(judge["policy"]["models"]["critic"]) == plan["critic_digests"]["D1856"]
            == control_review["scoring_critic_digests"]["common_frozen1856"], "same qualified D1856")
    require(state_digest(control["policy"]["models"]["critic"]) == plan["critic_digests"]["D400"], "same qualified D400")
    old_eval, new_eval = read(directory / "evaluation-v2.json"), read(directory / "evaluation-v3.json")
    audit_evaluation(old_eval, data, "control")
    audit_evaluation(new_eval, data, "gated")
    pairing = {}
    for pool, count in COUNTS.items():
        for old, new in zip(old_eval[pool]["records"], new_eval[pool]["records"]):
            require(all(old[name] == new[name] for name in ("index", "source_caption_id", "time", "teacher_rms")),
                    pool + " endpoint context/teacher pairing")
        comparison = receipt["comparison"][pool]
        require(comparison["count"] == count, pool + " comparison count")
        close(comparison["control_rmse"], old_eval[pool]["rmse"], pool + " control diagnostic")
        close(comparison["gated_rmse"], new_eval[pool]["rmse"], pool + " gated diagnostic")
        close(comparison["rmse_change_evaluation_only"], new_eval[pool]["rmse"] - old_eval[pool]["rmse"], pool + " diagnostic delta")
        pairing[pool] = {}
        for judge_name in JUDGES:
            records = list(zip(old_eval[pool]["records"], new_eval[pool]["records"]))
            changes = [new[judge_name]["g_game"] - old[judge_name]["g_game"] for old, new in records]
            close(comparison[judge_name]["control_g_game"], old_eval[pool][judge_name]["g_game"], pool + " control score")
            close(comparison[judge_name]["gated_g_game"], new_eval[pool][judge_name]["g_game"], pool + " gated score")
            close(comparison[judge_name]["new_minus_control"], sum(changes) / count, pool + " paired game delta")
            subjects = defaultdict(list)
            for old, new in records:
                subjects[str(old["source_caption_id"])].append(new[judge_name]["g_game"] - old[judge_name]["g_game"])
            pairing[pool][judge_name] = dict(mean_game_new_minus_control=sum(changes) / count,
                contexts_improved=sum(value < 0 for value in changes), contexts_worsened=sum(value > 0 for value in changes),
                subjects={subject: dict(contexts=len(values), mean_game_new_minus_control=sum(values) / len(values))
                          for subject, values in subjects.items()})
    require(status["comparison"] == receipt["comparison"], "final dashboard comparison copy")
    contribution = audit_contributions(read(directory / "particle-contribution.json"), data)
    hookup = {str(step): audit_hookup(read(directory / name), step, config["sites"])
              for step, name in ((2, "audit-step-two.json"), (400, "audit-final.json"))}
    progress = rows(directory / "progress.jsonl")
    require([item["step"] for item in progress] == [100, 200, 300, 400], "predeclared progress horizons")
    for item in progress + [read(directory / "control-progress400.json")]:
        require(item["native_state_unchanged"] and item["evaluation_only"] and item["output_metrics_used"] is False,
                "read-only native-game progress")
    require(read(directory / "progress-latest.json") == progress[-1], "progress latest copy")
    source = final["policy"]["served_source"]
    selected = final["policy"]["models" if source == "fast" else "averages"]
    expected = {"generator." + name: tensor for name, tensor in selected["generator"].items()
                if final["policy"]["requires_grad"]["generator"].get(name, False)}
    expected.update({"router." + name: tensor for name, tensor in selected["router"].items() if name != "log_mass"})
    expected.update({"bank.table": final["policy"]["table" if source == "fast" else "averaged_table"],
                     "bank.log_mass": selected["router"]["log_mass"]})
    export = directory / "final.safetensors"
    require(sha(export) == receipt["final_export_sha256"] and len(expected) == 428, "clean export file/schema")
    with safe_open(str(export), framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
        export_config = json.loads(metadata["config"])
        require(metadata["format"] == "supra_particlegan_clean_v3" and export_config["architecture"] == V3,
                "explicit V3 export formula")
        require(export_config == receipt["export"]["config"] and export_config["completed_steps"] == 400
                and export_config["served_source"] == source and export_config["sites"] == config["sites"],
                "export configuration/horizon/source/sites")
        require(export_config["rank"] == 16 and export_config["z_dim"] == 4 and export_config["num_particles"] == 128
                and export_config["cfg"] == 3 and export_config["sampling"] == "clean"
                and export_config["routed_geometry"] == "mass_atoms_v1", "export dimensions/native routing")
        require(json.loads(metadata["native_pins"])["backend_sha256"] == read(ROOT / "backend.lock.json")["sha256"], "export backend pin")
        require(set(handle.keys()) == set(expected), "all428 export tensor names")
        for name, tensor in expected.items():
            value = handle.get_tensor(name)
            require(value.dtype == tensor.dtype == torch.float32 and torch.equal(value, tensor), "export tensor " + name)
    artifact_names = ["plan.json", "run.json", "receipt.json", "train.jsonl", "final.pt", "checkpoint-00002.pt",
        "evaluation-v2.json", "evaluation-v3.json", "particle-contribution.json", "audit-step-two.json", "audit-final.json",
        "progress.jsonl", "final.safetensors"]
    return dict(schema="supra_particle_gated_v3_matched_400_cpu_review_v1", qualified=True, gpu_used=False,
        run=str(directory), fixed_updates=400, particlegan_pin=PIN, profile=PROFILE,
        source_snapshot_files_verified=len(plan["application_source_sha256"]) + len(committed),
        immutable_inputs_verified=len(inputs), frozen_four_owner_digest=frozen,
        native_replay_initialization_runtime_witnesses={name: receipt["checks"][name] for name in
            ("fresh_v2_step_two_full_native_exact", "fresh_v2_prefix_rows_exact", "fresh_v3_identical_native_step_zero",
             "v3_zero_output_base_exact", "v3_cpu_checkpoint_replay_full_native_exact", "v3_replay_three_rows_exact")},
        trace=dict(updates=400, preservation_updates=80, native_streams_matched=True, dv12_matches=dv12_matches,
                   accepted_moves=accepted), paired_games=pairing, contributions=contribution, hookup=hookup,
        final_native_digest=receipt["final_native_digest"], export_tensors=428,
        artifact_sha256={name: sha(directory / name) for name in artifact_names}, reviewer_sha256=sha(__file__),
        output_metrics_used_for_selection=False,
        limits="CPU verifies recorded states, sources, sampling and all raw record aggregations. Actual CUDA BF16 forwards, input gradients, native prefix equality, three-update exact recovery, clean reload parity and private CUDA final Gaussian panels are executed source/runtime witnesses; CPU does not rerun them. One task/stream and400updates do not establish superiority to ordinary LoRA.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "outputs/e22-particle-gated-v3-400")
    parser.add_argument("--particlegan-root", type=Path, default=ROOT.parent / "ParticleGAN-pr223-supra")
    args = parser.parse_args()
    torch.set_num_threads(4)
    result = build_review(args.run, args.particlegan_root)
    (args.run / "independent-review.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("qualified", "gpu_used", "fixed_updates", "paired_games", "trace")}), flush=True)


if __name__ == "__main__":
    main()

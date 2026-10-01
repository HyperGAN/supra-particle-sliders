"""Verify and summarize the complete five-arm continuation without GPU work."""
from argparse import ArgumentParser
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import torch
from safetensors import safe_open


ARMS = ("native_control", "clean_dv12", "hold_game_units", "no_anchor", "generator_no_amsgrad")
COUNTS = {"fit": 240, "test": 240, "holds": 30, "preservation": 60}
ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_digest(value):
    """The run's complete content digest, independently applied to saved data."""
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


def read(path):
    return json.loads(Path(path).read_text())


def finite(value, location="root"):
    if isinstance(value, float):
        assert math.isfinite(value), f"nonfinite {location}: {value}"
    elif isinstance(value, dict):
        for key, item in value.items():
            finite(item, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            finite(item, f"{location}[{index}]")


def close(actual, expected, label, tolerance=1e-12):
    assert math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance), (label, actual, expected)


def reconstruct_final(original, delta):
    """Reattach immutable hosts to the expressly non-resumable diagnostic delta."""
    policy = dict(original["policy"])
    policy.update({key: value for key, value in delta["policy"].items() if key not in ("models", "averages")})
    for family in ("models", "averages"):
        policy[family] = {role: dict(values) for role, values in original["policy"][family].items()}
        for role, values in delta["policy"][family].items():
            policy[family][role].update(values)
    return dict(policy=policy, data_rng=delta["data_rng"], paired_noise_rng=delta["paired_noise_rng"],
                config=delta["config"])


def export_check(path, receipt, final, original_run):
    export = receipt["export"]
    assert path.stat().st_size == export["bytes"]
    selected = final["policy"]["models" if receipt["served_source"] == "fast" else "averages"]
    trainable = {name for name, flag in final["policy"]["requires_grad"]["generator"].items() if flag}
    expected = {"generator." + name: selected["generator"][name] for name in trainable}
    expected.update({"router." + name: tensor for name, tensor in selected["router"].items() if name != "log_mass"})
    expected["bank.table"] = final["policy"]["table" if receipt["served_source"] == "fast" else "averaged_table"]
    expected["bank.log_mass"] = selected["router"]["log_mass"]
    assert len(expected) == export["tensors"] == 428
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
        assert metadata["format"] == "supra_particlegan_clean_v1"
        config = json.loads(metadata["config"])
        assert config == export["config"]
        assert (config["rank"], config["z_dim"], config["num_particles"], config["cfg"]) == (16, 4, 128, 3.)
        assert config["sampling"] == "clean" and config["routed_geometry"] == "mass_atoms_v1"
        assert len(config["sites"]) == 71 and config["completed_steps"] == 6656
        assert config["extra"]["diagnostic_arm"] == receipt["arm"]
        assert config["extra"]["starting_step"] == 6400 and config["extra"]["fixed_updates"] == 256
        assert config["extra"]["selection"] == "final fixed horizon; output metrics evaluation only"
        pins = json.loads(metadata["native_pins"])
        assert pins["model_revision"] == original_run["model_revision"]
        assert pins["text_encoder_revision"] == original_run["text_revision"]
        assert pins["vae_revision"] == original_run["vae_revision"]
        assert pins["backend_sha256"] == read(ROOT / "backend.lock.json")["sha256"]
        assert set(handle.keys()) == set(expected)
        parameters = 0
        for name, reference in expected.items():
            tensor = handle.get_tensor(name)
            assert tensor.dtype == torch.float32 and torch.equal(tensor, reference), f"export differs: {name}"
            if name == "bank.log_mass":
                assert not torch.isnan(tensor).any() and not torch.isposinf(tensor).any()
                assert torch.isfinite(tensor).any()
            else:
                assert torch.isfinite(tensor).all(), f"nonfinite exported tensor {name}"
            parameters += tensor.numel()
        assert parameters == export["parameters"]
    return dict(tensors=428, parameters=parameters, tensor_values_match_selected_final_state=True,
                native_pins_match=True, schema_and_horizon_match=True, sha256=sha(path))


def evaluate_check(evaluation, receipt, data):
    for pool, expected_count in COUNTS.items():
        result = evaluation[pool]
        records = result["records"]
        assert result["count"] == len(records) == expected_count
        assert [row["index"] for row in records] == list(range(expected_count))
        for index, row in enumerate(records):
            assert row["source_caption_id"] == int(data[pool]["context"][index, 4097])
            assert row["time"] == float(data[pool]["context"][index, 4096])
            close(row["rmse"] ** 2, row["mse"], f"{pool} individual RMSE", tolerance=2e-7)
            for judge in ("fixed_start_D", "arm_final_D"):
                score = row[judge]
                close(score["g_game"] - score["d_game"], score["score_gap"], f"{pool} logistic identity", tolerance=2e-6)
        mean_mse = sum(row["mse"] for row in records) / expected_count
        close(result["rmse"], math.sqrt(mean_mse), f"{pool} aggregate RMSE")
        close(result["relative_rms"], math.sqrt(mean_mse / (sum(row["teacher_rms"] ** 2 for row in records) / expected_count)),
              f"{pool} aggregate relative RMS")
        for judge in ("fixed_start_D", "arm_final_D"):
            for metric in ("g_game", "d_game", "score_gap"):
                close(result[judge][metric], sum(row[judge][metric] for row in records) / expected_count,
                      f"{pool} aggregate {judge}.{metric}")
        assert {key: value for key, value in result.items() if key != "records"} == receipt["metrics"][pool]


def paired_pool(native, candidate, prompts):
    groups = defaultdict(list)
    values = []
    for first, second in zip(native["records"], candidate["records"]):
        identity = ("index", "source_caption_id", "time", "teacher_rms")
        assert all(first[key] == second[key] for key in identity), "evaluation pairing differs"
        improvement = first["fixed_start_D"]["g_game"] - second["fixed_start_D"]["g_game"]
        pair = dict(game_improvement=improvement, native_mse=first["mse"], candidate_mse=second["mse"])
        groups[first["source_caption_id"]].append(pair)
        values.append(improvement)
    subjects = {}
    for caption_id, rows in groups.items():
        mean = sum(row["game_improvement"] for row in rows) / len(rows)
        subjects[str(caption_id)] = dict(source_prompt=prompts[caption_id], contexts=len(rows),
            common_frozen_critic_mean_g_loss_improvement=mean,
            contexts_with_lower_common_g_loss=sum(row["game_improvement"] > 0 for row in rows),
            evaluation_only_native_rmse=math.sqrt(sum(row["native_mse"] for row in rows) / len(rows)),
            evaluation_only_arm_rmse=math.sqrt(sum(row["candidate_mse"] for row in rows) / len(rows)))
    return dict(contexts=len(values), common_frozen_critic_mean_g_loss_improvement=sum(values) / len(values),
                contexts_with_lower_common_g_loss=sum(value > 0 for value in values),
                subjects_with_lower_mean_common_g_loss=sum(row["common_frozen_critic_mean_g_loss_improvement"] > 0 for row in subjects.values()),
                subjects=subjects)


def main():
    parser = ArgumentParser()
    parser.add_argument("--run", type=Path, default=ROOT / "outputs/e22-convergence-gap/training-controls-256-final")
    parser.add_argument("--output", type=Path, help="Default: RUN/qualification-review.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    directory = args.run.resolve()
    aggregate = read(directory / "receipt.json")
    plan = read(directory / "plan.json")
    assert aggregate["plan"] == plan
    assert tuple(plan["predeclared_arms"]) == tuple(plan["selected_arms"]) == ARMS
    assert set(aggregate["arms"]) == set(ARMS), "All five arms must finish before review"
    assert (plan["start_step"], plan["updates"], plan["end_step"]) == (6400, 256, 6656)
    assert sha(directory / "source.py") == plan["script_sha256"], "executed source snapshot differs"
    original_dir = Path(plan["checkpoint"]).parent
    original_run = read(original_dir / "run.json")
    assert plan["original_particlegan_commit"] == original_run["particlegan_commit"]
    package = Path(plan["imported_particlegan"]).parent
    assert state_digest({path.name: sha(path) for path in sorted(package.glob("*.py"))}) == plan["original_particlegan_source_digest"]
    for name, expected in plan["supra_source_sha256"].items():
        assert sha(original_dir / "source" / name) == expected
    original = torch.load(plan["checkpoint"], map_location="cpu", mmap=True, weights_only=False)
    initial_digest = state_digest(original)
    frozen = {role: {name: original["policy"]["models"][role][name] for name, flag in
                    original["policy"]["requires_grad"][role].items() if not flag}
              for role in ("generator", "encoder")}
    frozen_digest = state_digest(frozen)
    data = torch.load(original_dir / "data.pt", map_location="cpu", mmap=True, weights_only=False)
    assert state_digest(data) == original["config"]["dataset_digest"]
    assert original["config"]["output_error_guard"] is False
    evaluations, reports, file_hashes, reference_trace = {}, {}, {}, None
    for name in ("plan.json", "receipt.json", "source.py"):
        file_hashes[name] = sha(directory / name)
    for arm in ARMS:
        path = directory / arm
        receipt = read(path / "receipt.json")
        initial = read(path / "initial-state.json")
        assert receipt == aggregate["arms"][arm]
        assert initial["full_checkpoint_exact"] and initial["completed_steps"] == 6400
        assert initial["state_digest"] == initial_digest and initial["frozen_digest"] == frozen_digest
        assert receipt["arm"] == arm and receipt["completed_steps"] == 6656 and receipt["updates"] == 256
        for flag in ("start_state_exact", "data_and_paired_noise_streams_matched", "frozen_unchanged", "evaluation_state_unchanged",
                     "dv12_stream_checked_on_equal_call_shapes", "dv12_call_shape_program_entirely_matched"):
            assert receipt[flag], f"{arm}: {flag} failed"
        assert receipt["native_dv12_calls"] == 256 * 2 * 71
        assert receipt["dv12_equal_call_shape_steps"] == 256
        assert receipt["native_penalty_calls"] == 6656
        assert receipt["actual_anchor_weight"] == (0. if arm == "no_anchor" else 1.)
        assert receipt["generator_amsgrad"] == ([False] * 4 if arm == "generator_no_amsgrad" else [True] * 4)
        assert receipt["critic_amsgrad"] == [True]
        assert receipt["final_critic_source"] == "fast live training critic"
        assert receipt["served_source"] in ("fast", "averaged")
        assert receipt["clean_dv12_consumed_calls"] == (36352 if arm == "clean_dv12" else None)
        rows = [json.loads(line) for line in (path / "train.jsonl").read_text().splitlines() if line.strip()]
        assert len(rows) == 256 and [row["step"] for row in rows] == list(range(6401, 6657))
        finite(rows, arm + ".trace")
        for index, row in enumerate(rows, 1):
            assert row["arm"] == arm and row["penalty_phase"] == "blend"
            assert row["penalty_calls"] == row["step"] and row["dv12_call_count"] == index * 142
            assert row["hold"] == (row["step"] % 5 == 0)
            assert row["game_weight"] == (.1 if row["hold"] else 1.)
            if arm == "hold_game_units":
                assert row["critic_game_weight"] == row["controller_payoff_weight"] == 1.
        keys = ("step", "hold", "batch_indices", "base_noise_sums", "paired_rng_digest", "dv12_rng_digest",
                "dv12_call_count", "dv12_draw_shape_signature")
        trace = [{key: row[key] for key in keys} for row in rows]
        if reference_trace is None:
            reference_trace = trace
        else:
            assert trace == reference_trace, f"{arm}: paired training streams/draw program differ"
        assert receipt["final_dv12_rng_digest"] == rows[-1]["dv12_rng_digest"]
        assert receipt["final_paired_rng_digest"] == rows[-1]["paired_rng_digest"]
        moves = [row["move"] for row in rows if isinstance(row["move"], dict) and row["move"].get("moves", 0) > 0]
        assert moves == receipt["accepted_moves"]
        delta = torch.load(path / "training-delta.pt", map_location="cpu", mmap=True, weights_only=False)
        assert delta["format"] == "diagnostic_delta_without_frozen_hosts_v1" and delta["resumeable"] is False
        assert delta["config"] == original["config"]
        final = reconstruct_final(original, delta)
        assert state_digest(final) == receipt["final_state_digest"], f"{arm}: reconstructed final full-state digest differs"
        assert final["policy"]["completed_steps"] == 6656
        assert final["policy"]["optimizers"][1]["regularizer"]["record"]["observed_steps"] == 6656
        assert [group["amsgrad"] for group in final["policy"]["optimizers"][0]["param_groups"]] == receipt["generator_amsgrad"]
        assert [group["lr"] for group in final["policy"]["optimizers"][0]["param_groups"]] == receipt["learning_rates"]["generator"]
        assert [group["lr"] for group in final["policy"]["optimizers"][1]["param_groups"]] == receipt["learning_rates"]["critic"]
        evaluation = read(path / "evaluation.json")
        finite(evaluation, arm + ".evaluation")
        evaluate_check(evaluation, receipt, data)
        evaluations[arm] = evaluation
        reports[arm] = dict(full_start_state_digest_matches_original=True, frozen_digest_matches_original=True,
                           reconstructed_full_final_state_digest_matches_receipt=True,
                           all_training_streams_and_draw_programs_match_native=True,
                           fixed_horizon_and_native_clocks_match=True, evaluation_records_and_aggregates_match=True,
                           intervention_flags_match_plan=True,
                           export=export_check(path / "final.safetensors", receipt, final, original_run),
                           accepted_move_updates=len(moves), dense_gradient_rows_min=min(row["dense_gradient_rows"] for row in rows),
                           dense_gradient_rows_max=max(row["dense_gradient_rows"] for row in rows))
        for filename in ("receipt.json", "initial-state.json", "train.jsonl", "evaluation.json", "training-delta.pt", "final.safetensors"):
            file_hashes[str(Path(arm) / filename)] = sha(path / filename)
        print(json.dumps(dict(event="arm_verified", arm=arm, final_step=6656, tensors=428)), flush=True)
        del final, delta
    comparisons, table = {}, []
    for arm in ARMS:
        comparisons[arm] = {pool: paired_pool(evaluations["native_control"][pool], evaluations[arm][pool], data["prompts"])
                            for pool in COUNTS}
        assert len(comparisons[arm]["fit"]["subjects"]) == len(comparisons[arm]["test"]["subjects"]) == 6
        current = evaluations[arm]
        native = evaluations["native_control"]
        table.append(dict(arm=arm, test_common_frozen_D_g_loss=current["test"]["fixed_start_D"]["g_game"],
                          test_common_frozen_D_g_loss_improvement_vs_native=comparisons[arm]["test"]["common_frozen_critic_mean_g_loss_improvement"],
                          test_subjects_with_lower_mean_common_g_loss=comparisons[arm]["test"]["subjects_with_lower_mean_common_g_loss"],
                          test_evaluation_only_rmse=current["test"]["rmse"],
                          test_evaluation_only_rmse_relative_change=current["test"]["rmse"] / native["test"]["rmse"] - 1.,
                          preservation_common_frozen_D_g_loss=current["preservation"]["fixed_start_D"]["g_game"],
                          preservation_evaluation_only_rmse=current["preservation"]["rmse"],
                          accepted_move_updates=reports[arm]["accepted_move_updates"]))
    # Guard against a report being assembled from files still changing.
    assert all(sha(directory / name) == value for name, value in file_hashes.items())
    report = dict(schema="supra_e22_training_controls_qualification_v1", qualified=True,
                  scope="Read-only CPU mmap of original and delta state; exact stream/source/schema/digest checks; evaluation reporting",
                  start_step=6400, updates=256, final_step=6656,
                  initial_full_state_digest=initial_digest, unchanged_frozen_parameter_digest=frozen_digest,
                  common_frozen_critic_digest=state_digest(original["policy"]["models"]["critic"]),
                  checks=reports, paired_common_frozen_critic_comparisons=comparisons, summary_table=table,
                  artifact_sha256=file_hashes, output_metrics_used_for_optimization_or_selection=False, gpu_used=False,
                  interpretation="Positive game improvement is a paired decrease of G loss under the same frozen starting critic. Final arm critics differ and are not a common ranking judge.",
                  limits=["One matched continuation from one learned state, not independent training replicates or seed experiments.",
                          "The six subject averages contain correlated contexts from fixed trajectories.",
                          "A 256-update local intervention does not establish the cause of the original 6400-update gap.",
                          "Shared frozen-critic payoff improvements do not by themselves establish persistent game repair.",
                          "Output errors are evaluation only and did not select updates, structural decisions, guards or checkpoints.",
                          "The diagnostic delta omits immutable hosts and is explicitly not a standalone resumable checkpoint."])
    output = args.output or directory / "qualification-review.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    (directory / "summary-table.json").write_text(json.dumps(table, indent=2) + "\n")
    print(json.dumps(dict(event="qualification_complete", output=str(output), arms=len(ARMS), checks_passed=True,
                          summary_table=table)), flush=True)


if __name__ == "__main__":
    main()

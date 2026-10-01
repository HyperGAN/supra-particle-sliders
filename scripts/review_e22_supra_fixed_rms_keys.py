#!/usr/bin/env python3
"""Independent CPU artifact review of the one fixed256 normalized-key trial.

No model forward, optimizer update, checkpoint selection or GPU work occurs.
GPU replay/restore witnesses are consumed after verifying their exact archived
sources, inputs, full checkpoint digests and paired complete-pool aggregates.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "outputs/e22-particle-fixed-rms-keys-256")
    parser.add_argument("--native", type=Path, default=ROOT / "outputs/e22-particle-preservation-units-256/native")
    args = parser.parse_args()
    receipt, plan = load(args.run / "receipt.json"), load(args.run / "plan.json")
    source_root = Path(plan["particlegan_root"])
    sys.path.insert(0, str(source_root))
    sys.path.insert(1, str(ROOT))
    import torch
    from supra.particle_pilot import state_digest
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    checks = {}
    manifest = load(args.run / "source/sha256.json")
    checks["application_sources_match_archive_and_current"] = all(
        sha(ROOT / name) == digest == sha(args.run / "source" / name)
        for name, digest in manifest["application"].items())
    checks["particlegan_sources_match_archive_and_current"] = all(
        sha(source_root / "particlegan" / name) == digest == sha(args.run / "source/particlegan" / name)
        for name, digest in manifest["particlegan"].items())
    checks["pinned_source_digest_matches"] = state_digest({name: digest for name, digest in
        manifest["particlegan"].items() if "/" not in name}) == plan["particlegan_source_digest"]
    initial_run = Path(plan["reference_run"])
    paths = dict(checkpoint=initial_run / "final.pt", data=initial_run / "data.pt",
                 run=initial_run / "run.json", qualification=initial_run / "receipt.json",
                 native_control_trace=args.native / "train.jsonl")
    checks["all_initial_input_SHA256_match"] = all(sha(paths[name]) == digest
        for name, digest in plan["input_sha256"].items())
    initial = torch.load(paths["checkpoint"], map_location="cpu", weights_only=False, mmap=True)
    wrapped = torch.load(args.run / "final.pt", map_location="cpu", weights_only=False, mmap=True)
    final = wrapped["native"]
    native = torch.load(args.native / "final.pt", map_location="cpu", weights_only=False, mmap=True)
    native_receipt = load(args.native / "receipt.json")
    checks["initial_native_full_digest_match"] = state_digest(initial) == receipt["initial_native_digest"] == plan["initial_native_digest"]
    checks["tagged_final_full_digest_match"] = state_digest(wrapped) == receipt["final_experimental_digest"]
    checks["final_inner_native_digest_match"] = state_digest(final) == receipt["final_native_digest"]
    checks["final_checkpoint_SHA256_match"] = sha(args.run / "final.pt") == receipt["final_checkpoint_sha256"]
    checks["native_control_digest_match"] = state_digest(native) == native_receipt["final_native_digest"]
    routing = plan["experimental_routing"]
    checks["explicit_routing_metadata_consistent"] = (wrapped["format"] == "e22_supra_experimental_fixed_rms_keys_v1"
        and wrapped["experimental_routing"] == routing == final["config"]["experimental_routing"]
        == load(args.run / "run.json")["experimental_routing"] == receipt["experimental_routing"]
        and wrapped["ordinary_clean_export_supported"] is False
        and receipt["ordinary_clean_export_supported"] is False
        and wrapped["dedicated_restore"] == "experimental_e22_fixed_rms_keys.restore_experimental")
    checks["ordinary_clean_export_absent"] = not list(args.run.glob("*.safetensors"))
    base_config = dict(final["config"])
    del base_config["experimental_routing"]
    checks["only_experimental_routing_config_changes"] = base_config == initial["config"]
    checks["native_recipe_roles_shapes_retained"] = all(final["policy"][name] == initial["policy"][name]
        for name in ("recipe", "roles", "row_semantics", "table_location", "table_requires_grad"))
    checks["all_particles_and_sites_retained"] = final["policy"]["table"].shape == (128, 4) and len(final["config"]["sites"]) == 71
    # Includes the frozen text/mask buffers and all teacher tensors as well.
    checks["FAST_and_averaged_encoders_unchanged"] = all(
        state_digest(initial["policy"][family]["encoder"]) == state_digest(final["policy"][family]["encoder"])
        for family in ("models", "averages"))
    frozen_names = [name for name, trainable in initial["policy"]["requires_grad"]["generator"].items() if not trainable]
    checks["all_FAST_and_averaged_frozen_host_tensors_equal"] = all(
        torch.equal(initial["policy"][family]["generator"][name], final["policy"][family]["generator"][name])
        for family in ("models", "averages") for name in frozen_names)
    checks["trainability_flags_retained"] = initial["policy"]["requires_grad"] == final["policy"]["requires_grad"]
    sampling = load(args.run / "sampling-program.json")
    rows = [json.loads(row) for row in (args.run / "train.jsonl").read_text().splitlines()]
    control_rows = [json.loads(row) for row in paths["native_control_trace"].read_text().splitlines()]
    checks["exact_fixed256_horizon"] = (len(rows) == 256 and [r["step"] for r in rows] == list(range(1857, 2113))
        and final["policy"]["completed_steps"] == 2112 and initial["policy"]["completed_steps"] == 1856)
    checks["sampling_program_digest_match"] = state_digest(sampling) == receipt["sampling_program_digest"]
    checks["all_steps_match_declared_sampling"] = len(rows) == len(sampling) and all(
        all(row[key] == value for key, value in expected.items()) for row, expected in zip(rows, sampling))
    checks["all_steps_match_native_data_paired_DV12_streams"] = len(rows) == len(control_rows) and all(
        all(row[key] == control[key] for key in ("step", "hold", "batch_indices", "base_noise_sums", "paired_rng_digest", "dv12_rng_digest"))
        for row, control in zip(rows, control_rows))
    data_stream = torch.Generator().set_state(initial["data_rng"])
    expected_indices = []
    for step in range(1857, 2113):
        expected_indices.append(torch.randint(30 if step % 5 == 0 else 240, (4,), generator=data_stream).tolist())
    checks["CPU_recomputed_data_sampling_and_final_stream_exact"] = (
        expected_indices == [r["batch_indices"] for r in rows] and torch.equal(data_stream.get_state(), final["data_rng"]))
    rng_digest = lambda value: hashlib.sha256(bytes(value.cpu().tolist())).hexdigest()
    checks["stored_final_stream_digests_match"] = (rng_digest(final["data_rng"]) == receipt["final_data_rng_digest"]
        and rng_digest(final["paired_noise_rng"]) == receipt["final_paired_rng_digest"]
        and rng_digest(final["policy"]["streams"]["noise_generator"]) == receipt["final_dv12_rng_digest"])
    checks["final_data_paired_and_all_private_policy_streams_match_native"] = (
        torch.equal(final["data_rng"], native["data_rng"]) and torch.equal(final["paired_noise_rng"], native["paired_noise_rng"])
        and all(torch.equal(value, native["policy"]["streams"][name]) for name, value in final["policy"]["streams"].items()))
    checks["all128_bank_rows_receive_game_gradients_every_update"] = all(r["dense_gradient_rows"] == 128 for r in rows)
    checks["no_output_metric_training_or_structural_guard"] = (final["config"]["output_error_guard"] is False
        and final["config"]["max_feature_context_harm"] == 0 and final["policy"]["routing"]["config"]["max_context_harm"] == 0
        and all(r["critic_game_weight"] == r["game_weight"] == r["controller_payoff_weight"] for r in rows))
    restore_review = load(args.run / "restore-review.json")
    witness_fields = ("initial_native_state_exact", "native_control_four_update_replay_exact", "native_control_trace_exact", "four_update_replay_exact")
    checks["qualified_GPU_native_and_experimental_prefix_replay"] = all(
        receipt[field] is True and restore_review[field] is True for field in witness_fields)
    checks["qualified_GPU_dedicated_restore_output_state_witness"] = all(receipt[field] is True for field in
        ("dedicated_experimental_restore_exact", "evaluation_state_unchanged", "frozen_unchanged", "source_and_inputs_unchanged"))
    metrics, baseline = load(args.run / "evaluation.json"), load(args.native / "evaluation.json")
    pools = {}
    aggregate_pass, identities_pass = True, True
    for name, expected_count in (("fit", 240), ("test", 240), ("holds", 30), ("preservation", 60)):
        actual, control = metrics[name], baseline[name]
        ar, cr = actual["records"], control["records"]
        aggregate_pass &= len(ar) == actual["count"] == expected_count
        for judge in ("fixed_start_D", "arm_final_D"):
            for key in ("g_game", "d_game", "score_gap"):
                aggregate_pass &= abs(sum(r[judge][key] for r in ar) / len(ar) - actual[judge][key]) < 1e-12
        aggregate_pass &= abs((sum(r["mse"] for r in ar) / len(ar)) ** .5 - actual["rmse"]) < 1e-12
        identities_pass &= len(ar) == len(cr) and all(all(a[k] == c[k] for k in
            ("index", "source_caption_id", "time", "teacher_rms")) for a, c in zip(ar, cr))
        per_subject = defaultdict(list)
        differences = []
        for a, c in zip(ar, cr):
            value = a["fixed_start_D"]["g_game"] - c["fixed_start_D"]["g_game"]
            differences.append(value); per_subject[a["source_caption_id"]].append(value)
        pools[name] = dict(count=len(ar), native_fixed_game=control["fixed_start_D"]["g_game"],
            experimental_fixed_game=actual["fixed_start_D"]["g_game"],
            game_delta=sum(differences) / len(differences),
            contexts_improved=sum(v < 0 for v in differences), contexts_worsened=sum(v > 0 for v in differences),
            subject_game_deltas={str(k): sum(v) / len(v) for k, v in per_subject.items()},
            reporting_only_native_RMSE=control["rmse"], reporting_only_experimental_RMSE=actual["rmse"])
    checks["all_complete_pool_counts_and_aggregates_recomputed"] = aggregate_pass
    checks["all_complete_pool_context_identities_and_teachers_paired"] = identities_pass
    monitor = [load(args.run / file) for file in ("routing-switch-native.json", "routing-switch-experimental.json", "final-monitor.json")]
    checks["monitor_GPU_state_immutability_and_fixed12_12_contexts"] = all(row["native_state_unchanged"] is True
        and row["output_metrics_used"] is False and all(pool["contexts"] == 12 for pool in row["probes"].values()) for row in monitor)
    checks["monitor_receipt_matches_raw_files"] = monitor == [receipt["zero_update_native_game"], receipt["zero_update_experimental_game"], receipt["final_fixed_game_probe"]]
    change = {pool: {mode: dict(native_zero=monitor[0]["probes"][pool][mode]["frozen_start_D"],
        experimental_zero=monitor[1]["probes"][pool][mode]["frozen_start_D"], experimental_final=monitor[2]["probes"][pool][mode]["frozen_start_D"],
        switch_delta=monitor[1]["probes"][pool][mode]["frozen_start_D"] - monitor[0]["probes"][pool][mode]["frozen_start_D"],
        subsequent_256_update_delta=monitor[2]["probes"][pool][mode]["frozen_start_D"] - monitor[1]["probes"][pool][mode]["frozen_start_D"])
        for mode in ("clean", "dv12")} for pool in monitor[0]["probes"]}
    report = dict(passed=all(checks.values()), checks=checks, no_GPU_forward_or_optimizer_update=True,
        review_script_sha256=sha(__file__), checkpoint_sha256=receipt["final_checkpoint_sha256"],
        common_judge="same frozen V2 step1856 critic; arm_final_D scores are not compared across runs",
        pools=pools, fixed_monitor_switch_and_training=change,
        accepted_moves=receipt["accepted_moves"], structural_variants=Counter(r["move"].get("variant", "unselected")
            for r in rows if isinstance(r["move"], dict)),
        conclusion="The local frozen-query Jacobian repair did not yield a convergence win in this fixed256 warm continuation. Common-critic complete-pool game is worse in all four pools; retain native routing. Small edit probes improve from the switched start while preservation regresses, and the routing-switch shock is explicit.",
        limitations=["Native/experimental CUDA replay and dedicated forward equality are verified GPU witnesses with independently checked source/input/state provenance; CPU review does not rerun BF16 inference.",
                     "One task and stream, fixed256 continuation. This does not rule out a separately declared fresh-initialization experiment, but no additional trial or production promotion is implied.",
                     "Private Gaussian evaluation panels and teachers are paired; common-critic per-context means do not provide independent training replicates or an external image-quality judge."])
    (args.run / "independent-review.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(passed=report["passed"], checks=checks, pools=pools, monitor=change), indent=2), flush=True)
    if not report["passed"]:
        raise RuntimeError("independent artifact qualification failed")


if __name__ == "__main__":
    main()

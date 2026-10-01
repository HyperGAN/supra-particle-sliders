"""Independent receipt review rejects incomplete or falsely paired evidence."""
import json
import math

import pytest

from scripts.review_e22_supra_full import (
    ReviewError, audit_pool, audit_probe_receipt, build_review,
)


def pool():
    result = {}
    for arm, errors in (("old_lora", [1., 3.]), ("new_particlegan", [2., 4.])):
        records = [dict(index=index, prompt=prompt, t=.1 + index * .1, mse=error,
                        base_power=4., base_gap_mse=gap)
                   for index, (prompt, error, gap) in enumerate(zip(("a", "b"), errors, (2., 4.)))]
        mean = sum(errors) / 2
        result[arm] = dict(contexts=2, records=records, mse=mean, rmse=math.sqrt(mean),
                           base_power=4., relative_rms=math.sqrt(mean / 4),
                           base_gap_mse=3., velocity_ratio=mean / 3,
                           per_subject={row["prompt"]: dict(contexts=1, mse=row["mse"],
                                                            velocity_ratio=row["mse"] / row["base_gap_mse"])
                                        for row in records},
                           common_new_critic=dict(draws=4, contexts=2, per_draw_generator_loss=[1., 2., 3., 4.],
                                                  generator_loss=2.5, evaluation_only=True,
                                                  critic_state_sha256="same critic", paired_noise_sha256="same noise",
                                                  output_sigma=.125, noise_rng="private", law="clean"))
    return result


def test_complete_paired_record_aggregates_are_accepted():
    audit_pool(pool(), 2)


def test_incomplete_evaluation_is_rejected_before_any_artifact_read(tmp_path):
    folder = tmp_path / "evaluation"
    folder.mkdir()
    (folder / "status.json").write_text(json.dumps({"phase": "rendering"}))
    with pytest.raises(ReviewError, match="evaluation is incomplete"):
        build_review(tmp_path)


def test_reported_rmse_is_recomputed_from_records():
    report = pool()
    report["new_particlegan"]["rmse"] = 0.
    with pytest.raises(ReviewError, match="MSE/RMSE aggregation"):
        audit_pool(report, 2)


def test_individually_consistent_reports_still_require_identical_base_inputs():
    report = pool()
    new = report["new_particlegan"]
    new["records"][0]["base_power"] = 8.
    new["base_power"] = 6.
    new["relative_rms"] = math.sqrt(new["mse"] / 6)
    with pytest.raises(ReviewError, match="live target/base quantities differ"):
        audit_pool(report, 2)


def test_four_draw_scores_cannot_use_different_noise_between_arms():
    report = pool()
    report["new_particlegan"]["common_new_critic"]["paired_noise_sha256"] = "different noise"
    with pytest.raises(ReviewError, match="paired_noise_sha256 differs"):
        audit_pool(report, 2)


def test_missing_original_probes_cannot_be_reported_as_zero_error(tmp_path):
    baseline = tmp_path / "adapter.safetensors"
    with pytest.raises(ReviewError, match="missing original probes"):
        audit_probe_receipt({"available": True, "maximum_absolute_metric_difference": 0.}, baseline, [])
    result = audit_probe_receipt({"available": False, "reason": "not published"}, baseline, [])
    assert result == {"available": False, "reason": "not published"}


def test_6400_review_cannot_allow_historical_source_changes(tmp_path):
    folder = tmp_path / "evaluation"
    folder.mkdir()
    (folder / "status.json").write_text(json.dumps({"phase": "complete"}))
    (folder / "comparison.json").write_text(json.dumps({"completed_steps": 6400}))
    (tmp_path / "run.json").write_text(json.dumps({"fixed_updates": 6400}))
    with pytest.raises(ReviewError, match="only for the original 1600-step"):
        build_review(tmp_path, allow_source_changes=True)

"""Comparison receipts must never confuse training budgets or missing evidence."""
import json

import pytest
import torch
from safetensors.torch import save_file

pytest.importorskip("particlegan")
from scripts.evaluate_e22_supra_full import (
    baseline_probe_reproduction, comparison_horizon, render_labels,
)


def baseline(tmp_path, step):
    path = tmp_path / "adapter.safetensors"
    save_file({"weight": torch.ones(1)}, str(path), metadata={"step": json.dumps(step)})
    return path


@pytest.mark.parametrize("horizon", [1600, 6400])
def test_matching_declared_budget_and_baseline_label(horizon, tmp_path):
    path = baseline(tmp_path, horizon)
    resolved = comparison_horizon({"fixed_updates": horizon}, horizon, path)
    assert resolved == horizon
    assert render_labels(resolved)[1:3] == (f"Original LoRA / {horizon}", f"New ParticleGAN / {horizon}")


def test_original_1600_baseline_cannot_be_compared_as_6400(tmp_path):
    with pytest.raises(ValueError, match="original baseline step 1600"):
        comparison_horizon({"fixed_updates": 6400}, 6400, baseline(tmp_path, 1600))


def test_partial_checkpoint_cannot_replace_declared_final_horizon(tmp_path):
    with pytest.raises(ValueError, match="expects fixed step 6400, got 1600"):
        comparison_horizon({"fixed_updates": 6400}, 1600, baseline(tmp_path, 6400))


def test_override_cannot_change_declared_budget(tmp_path):
    with pytest.raises(ValueError, match="explicit horizon differs"):
        comparison_horizon({"fixed_updates": 6400}, 6400, baseline(tmp_path, 6400), expected_steps=1600)


def test_missing_run_requires_explicit_horizon(tmp_path):
    path = baseline(tmp_path, 6400)
    with pytest.raises(ValueError, match="declared run horizon"):
        comparison_horizon({}, 6400, path)
    assert comparison_horizon({}, 6400, path, expected_steps=6400) == 6400


@pytest.mark.parametrize("receipt", [{"score": .1, "step": 6400}, {"velocity_probes": []}, None])
def test_missing_published_probes_are_unavailable_not_zero_error(receipt, tmp_path):
    path = tmp_path / "validation.json"
    if receipt is not None:
        path.write_text(json.dumps(receipt))
    result = baseline_probe_reproduction(path, [{"name": "knight", "t": .2, "trained_mse": .1}])
    assert result["available"] is False
    assert result["reason"]
    assert "maximum_absolute_metric_difference" not in result


def test_published_reproduction_matches_probe_identity_and_reports_real_difference(tmp_path):
    rows = [{"name": "knight", "t": .2, "trained_mse": .1},
            {"name": "fruit", "t": .8, "relative_drift": .02}]
    path = tmp_path / "validation.json"
    path.write_text(json.dumps({"velocity_probes": rows}))
    current = [dict(rows[1], relative_drift=.0205), dict(rows[0], t=.200000003)]
    result = baseline_probe_reproduction(path, current)
    assert result["available"] is True
    assert result["compared_records"] == 2
    assert result["maximum_absolute_metric_difference"] == pytest.approx(.0005)
    assert len(result["receipt_sha256"]) == 64


def test_partial_probe_receipt_cannot_report_success(tmp_path):
    path = tmp_path / "validation.json"
    path.write_text(json.dumps({"velocity_probes": [{"name": "knight", "t": .2, "trained_mse": .1}]}))
    result = baseline_probe_reproduction(path, [])
    assert result["available"] is False
    assert "maximum_absolute_metric_difference" not in result

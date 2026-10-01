"""Qualification rejects incomplete, disconnected or unpaired experiments."""
from copy import deepcopy

import pytest
import torch

from scripts.review_e22_supra_full import ReviewError
from scripts.review_e22_supra_particle_architecture import UPDATES, audit_trace, build_review


@pytest.fixture
def traces():
    data = {"fit": {"context": torch.zeros(240, 1)}, "holds": {"context": torch.zeros(30, 1)}}
    stream = torch.Generator().manual_seed(7)
    rows = []
    for step in range(1, UPDATES + 1):
        hold = step % 5 == 0
        rows.append(dict(step=step, hold=hold, game_weight=.1 if hold else 1., penalty_calls=step,
                         batch_indices=torch.randint(30 if hold else 240, (4,), generator=stream).tolist(),
                         dense_gradient_rows=0 if step == 1 else 128, bank_grad_norm=0. if step == 1 else .001,
                         output_sigma=.125, base_noise_sums=[1., 2.], paired_rng_digest=f"paired-{step}",
                         dv12_rng_digest=f"dv12-{step}"))
    return rows, deepcopy(rows), data


def test_complete_dense_paired_trace_is_accepted(traces):
    review = audit_trace(*traces)
    assert review["updates"] == 1600
    assert review["preservation_updates"] == 320
    assert review["dv12_rng_digests_equal_steps"] == 1600


def test_missing_receipt_is_rejected_without_reading_training_artifacts(tmp_path):
    with pytest.raises(ReviewError, match="receipt is not complete"):
        build_review(tmp_path)


def test_disconnected_bank_is_rejected(traces):
    rows, old, data = traces
    rows[638]["dense_gradient_rows"] = 0
    with pytest.raises(ReviewError, match="not dense at 639"):
        audit_trace(rows, old, data)


def test_changed_paired_noise_is_rejected(traces):
    rows, old, data = traces
    rows[-1]["paired_rng_digest"] = "different"
    with pytest.raises(ReviewError, match="training draws differ at 1600"):
        audit_trace(rows, old, data)


def test_even_identically_changed_data_rows_must_match_original_stream(traces):
    rows, old, data = traces
    rows[0]["batch_indices"] = old[0]["batch_indices"] = [0, 0, 0, 0]
    with pytest.raises(ReviewError, match="CPU data stream changed at 1"):
        audit_trace(rows, old, data)


def test_nonfinite_trace_is_rejected(traces):
    rows, old, data = traces
    rows[-1]["bank_grad_norm"] = float("nan")
    with pytest.raises(AssertionError, match="nonfinite"):
        audit_trace(rows, old, data)

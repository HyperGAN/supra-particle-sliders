"""Saved-force diagnostic algebra only; no checkpoint, CUDA or training run."""
import math

import pytest
import torch
from torch import nn
from particlegan.gan_loss import GANLoss

from scripts.diagnose_e22_supra_token_game_force import (
    ARMS, CARD, COMMON, INDICES, SIGMA, allocation, force, mechanism_decision,
    module_witness, panel_subset, read, self_cauchy, unit_descent_slopes, validate_card,
)
import scripts.diagnose_e22_supra_token_game_force as observer


class ScalarTokenCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("scale", torch.tensor([2.], dtype=torch.float64))
        self.score = nn.Identity()

    def features(self, error, condition):
        return .7 * error - .2 * error.square()

    def forward(self, error, condition):
        return self.features(error, condition).mean(1)


def fixture():
    critic = ScalarTokenCritic().eval().requires_grad_(False)
    error = torch.tensor([[[.4], [.4]], [[.1], [1.3]]], dtype=torch.float64)
    condition = torch.zeros(2, 1, dtype=torch.float64)
    panel = torch.tensor([-.3, .3, -.2, .2], dtype=torch.float64).reshape(4, 1, 1, 1).expand(4, 2, 2, 1)
    return critic, error, condition, panel


def test_native_per_context_derivative_and_units_are_exact_public_rpgan():
    critic, error, condition, panel = fixture()
    value = force(critic, error, condition, panel, GANLoss())
    scalar = error[0, 0, 0].clone().requires_grad_(True)
    # A homogeneous context has native pooled score and ordinary paired RpGAN.
    real = SIGMA * panel[:, 0, 0, 0]
    fake = real + scalar / critic.scale[0]
    score = lambda x: .7*x - .2*x.square()
    expected = GANLoss().g_loss(score(fake), score(real))
    derivative = torch.autograd.grad(expected, scalar)[0]
    torch.testing.assert_close(value["games"][0], expected, rtol=0, atol=2e-15)
    torch.testing.assert_close(value["gradient"][0].sum(), derivative, rtol=0, atol=2e-15)
    token = force(critic, error, condition, panel, GANLoss(), tokenwise=True)
    torch.testing.assert_close(token["games"][0], value["games"][0], rtol=0, atol=2e-15)
    assert token["games"][1] > value["games"][1]  # Jensen for heterogeneous local gaps.


def test_origin_identity_and_force_observation_preserve_rng_owner_and_inputs():
    critic, error, condition, panel = fixture()
    before = module_witness(critic)
    rng = torch.get_rng_state().clone()
    originals = [x.clone() for x in (error, condition, panel)]
    origin = force(critic, error, condition, panel, GANLoss(), origin=True)
    torch.testing.assert_close(origin["games"], torch.full((2,), math.log(2), dtype=torch.float64), atol=2e-15, rtol=0)
    assert origin["gradient"].count_nonzero() > 0  # Stale fixed D need not be stationary.
    assert before == module_witness(critic) and torch.equal(rng, torch.get_rng_state())
    for value, original in zip((error, condition, panel), originals):
        assert torch.equal(value, original) and value.grad is None


def test_native_self_cauchy_control_and_equal_norm_common_game_directions():
    native = torch.tensor([[[3., 4.]], [[0., 0.]]], dtype=torch.float64)
    token = torch.tensor([[[4., -3.]], [[1., 0.]]], dtype=torch.float64)
    control = self_cauchy(native, token)
    assert control[0]["native_direction_slope"] == -5.
    assert control[0]["token_direction_slope"] == 0.
    assert control[0]["token_minus_native_slope"] == 5.
    assert control[1]["native_direction_slope"] is None
    common = unit_descent_slopes(token, native, token)
    assert common[0]["native_direction_slope"] == 0.
    assert common[0]["token_direction_slope"] == -5.


def test_selected_panels_retain_original_draw_and_context_addresses():
    batches = [torch.arange(4*4*2).reshape(4, 4, 2, 1) + batch*1000 for batch in range(60)]
    selected = panel_subset(batches)
    for position, index in enumerate(INDICES):
        assert torch.equal(selected[:, position], batches[index//4][:, index%4])


def test_descriptive_error_power_never_affects_mechanism_decision():
    rows = [dict(arm=arm, own_native_radial_derivative=.1,
        common_games={judge: dict(token_minus_native_slope=-.01) for judge in COMMON},
        descriptive_power_allocation={"arbitrary": 100000}) for arm in ARMS for _ in range(12)]
    before = mechanism_decision(rows)
    for row in rows:
        row["descriptive_power_allocation"] = {"arbitrary": -99999}
    assert mechanism_decision(rows) == before
    assert before["tokenwise_local_common_descent_supported"]
    for row in rows:
        row["common_games"]["D6400"]["token_minus_native_slope"] = .01
    assert mechanism_decision(rows)["next_action"] == "stop_this_token_granularity_mechanism_before_quality_training"


def test_allocation_bins_have_explicit_zero_norm_convention():
    error = torch.arange(1., 11.).reshape(1, 10, 1)
    gradient = torch.zeros_like(error)
    result = allocation(error, gradient)[0]
    assert result["largest_10_percent"]["descriptive_residual_power_fraction"] == pytest.approx(100/385)
    assert result["largest_10_percent"]["game_force_power_fraction"] is None


def test_completion_write_overrun_retains_incomplete_evidence(tmp_path, monkeypatch):
    clock = [0.]
    original = observer.write

    def delayed_completion(path, value):
        original(path, value)
        if path.name == "completion.json" and value["complete"]:
            clock[0] = 120.01

    monkeypatch.setattr(observer, "write", delayed_completion)
    with pytest.raises(TimeoutError):
        observer.finish(tmp_path, dict(qualified=True, complete=True), started=0, clock=lambda: clock[0])
    assert read(tmp_path / "report.json")["budget_overrun"]
    completion = read(tmp_path / "completion.json")
    assert not completion["complete"] and not completion["qualified"]
    assert completion["report_sha256"] == observer.sha(tmp_path / "report.json")


@pytest.mark.parametrize("key,value", [("fit_indices", list(range(12))), ("execution_budget_seconds", 121),
    ("sigma", .1), ("common_judges", ["D6400"])])
def test_fixed_card_rejects_scope_or_budget_overrides(key, value):
    card = read(CARD)
    validate_card(card)
    card[key] = value
    with pytest.raises(ValueError):
        validate_card(card)

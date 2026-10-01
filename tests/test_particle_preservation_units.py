"""Hold-unit intervention preserves native wiring and the task's G gradient weight."""
import pytest
import torch

from scripts.experiment_e22_supra_preservation_units import game_units_update, rolling_game_summary
from supra.particle_pilot import checkpoint, frozen_digest, restore, state_digest
from supra.particle_training import make_training_loop, training_update
from supra.runtime import TARGETS, model_module


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def loop():
    mod = model_module()
    model = mod.SupraDiT(d_model=8, depth=1, n_heads=2, ctx_dim=768,
                        mlp_ratio=2, num_tokens=256, patch=2)
    mod.attach_supra_lora(model, rank=16, alpha=16, targets=TARGETS)
    context = torch.linspace(-.8, .8, 4 * 4100).reshape(4, 4100)
    context[:, 4096] = torch.tensor([.2, .4, .6, .8])
    context[:, 4097], context[:, 4098], context[:, 4099] = 1, 2, 1
    holds = context.clone()
    holds[:, 4097:4099] = 3
    guard = context.clone()
    guard[:, :4096] += .031
    data = dict(text_contexts=torch.linspace(-.3, .7, 4 * 3 * 768).reshape(4, 3, 768),
                text_masks=torch.ones(4, 3), coordinate_scale=torch.ones(16),
                fit=dict(context=context, targets=torch.zeros(4, 4, 32, 32)),
                holds=dict(context=holds), guard=dict(context=guard))
    return make_training_loop(model, data, device="cpu", probe_interval=1000)


@pytest.mark.parametrize("completed_steps", [0, 4])
def test_native_mode_and_gradient_telemetry_preserve_stock_full_state(loop, completed_steps):
    loop.policy.completed_steps = completed_steps
    saved = checkpoint(loop)
    expected_row = training_update(loop)
    expected_state = state_digest(checkpoint(loop))
    restore(loop, saved)
    row = game_units_update(loop, mode="native")
    assert all(row[key] == value for key, value in expected_row.items())
    assert state_digest(checkpoint(loop)) == expected_state


def test_hold_units_restore_native_D_and_controller_units_while_G_stays_weighted(loop):
    # Populate nonzero branch outputs first, so the preservation gradients are
    # measurable rather than using a zero-residual untouched host.
    training_update(loop)
    loop.policy.completed_steps = 4
    saved = checkpoint(loop)
    original_frozen = frozen_digest(loop)
    native = game_units_update(loop, mode="native")
    restore(loop, saved)
    candidate = game_units_update(loop, mode="hold_units")
    assert native["hold"] and candidate["hold"]
    assert candidate["game_weight"] == native["game_weight"] == .1
    assert candidate["critic_game_weight"] == candidate["controller_payoff_weight"] == 1.
    assert native["critic_game_weight"] == native["controller_payoff_weight"] == .1
    assert candidate["loss_g"] == pytest.approx(.1 * candidate["loss_g_unweighted"], rel=1e-7)
    assert candidate["controller_observed_g"] == candidate["loss_g_unweighted"]
    assert candidate["controller_observed_d"] == candidate["loss_d_game_unweighted"]
    assert native["controller_observed_g"] == native["loss_g"]
    assert native["controller_observed_d"] == native["loss_d_game"]
    assert candidate["loss_d_game_unweighted"] == native["loss_d_game_unweighted"]
    # These small preservation residuals cause cancellation between the real
    # and fake FP32 derivatives. The two backward scaling orders can differ by
    # a few rounding units while still restoring the exact tenfold game weight.
    assert candidate["critic_gradient_geometry"]["game_gradient_norm"] == pytest.approx(
        native["critic_gradient_geometry"]["game_gradient_norm"] * 10, rel=1e-3, abs=1e-9)
    assert candidate["critic_gradient_geometry"]["penalty_gradient_norm"] == native["critic_gradient_geometry"]["penalty_gradient_norm"]
    for key in ("batch_indices", "base_noise_sums", "paired_rng_digest", "dv12_rng_digest"):
        assert native[key] == candidate[key]
    assert frozen_digest(loop) == original_frozen


def test_hold_units_resume_exactly_across_preservation_update(loop):
    loop.policy.completed_steps = 4
    saved = checkpoint(loop)
    expected_rows = [game_units_update(loop, mode="hold_units") for _ in range(2)]
    expected_state = state_digest(checkpoint(loop))
    restore(loop, saved)
    assert [game_units_update(loop, mode="hold_units") for _ in range(2)] == expected_rows
    assert state_digest(checkpoint(loop)) == expected_state


def test_rolling_summaries_do_not_mix_weighted_hold_losses_with_fit_losses():
    geometry = dict(game_gradient_norm=1., penalty_gradient_norm=2., game_penalty_gradient_cosine=-.5,
                    penalty_to_game_gradient_ratio=2.)
    rows = [dict(step=4, hold=False, loss_g_unweighted=1., loss_d_game_unweighted=2.,
                 penalty=.1, bank_grad_norm=.2, output_sigma=.125, critic_gradient_geometry=geometry),
            dict(step=5, hold=True, loss_g_unweighted=3., loss_d_game_unweighted=4.,
                 penalty=.1, bank_grad_norm=.02, output_sigma=.125, critic_gradient_geometry=geometry)]
    result = rolling_game_summary(rows)
    assert result["fit"]["loss_g_unweighted"] == 1.
    assert result["holds"]["loss_g_unweighted"] == 3.
    assert result["fit"]["updates"] == result["holds"]["updates"] == 1

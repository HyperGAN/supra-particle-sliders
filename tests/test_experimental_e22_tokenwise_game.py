"""Payoff/penalty units and complete native replay for paired-token game."""
from copy import deepcopy
import math

import pytest
import torch
import torch.nn.functional as F

from particlegan import RoutedBatch, get_recipe
from supra.particle_game import ConditionalTokenCritic, apply_critic_penalty
from supra.particle_pilot import checkpoint as native_checkpoint, restore as native_restore, state_digest
from supra.particle_training import training_update as native_training_update
from scripts.experimental_e22_tokenwise_game import (
    FORMULATION, checkpoint, configure, restore, token_logits, training_update,
)
from test_experimental_e22_bank_game_trust import native_loop


@pytest.fixture(autouse=True)
def serial_cpu():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.autograd.set_multithreading_enabled(False):
            yield
    finally:
        torch.set_num_threads(previous)


def loop():
    result = native_loop()
    result.fit_context = result.context
    result.fit_targets = torch.zeros(4, 4, 32, 32)
    result.hold_context = result.context + .05
    return result


def test_identical_pairs_and_homogeneous_scores_preserve_native_game_and_gradient():
    loss = get_recipe("e22_routed").make_loss()
    with torch.random.fork_rng(devices=[]):
        critic = ConditionalTokenCritic(torch.ones(16), condition_dim=2, width=8, feature_dim=4)
    error = torch.linspace(-.5, .5, 2 * 256 * 16).reshape(2, 256, 16)
    condition = torch.tensor([[.25, -.25], [.5, -.5]])
    pooled, local = critic(error, condition), token_logits(critic, error, condition)
    assert local.shape == (512, 1)
    for real, fake in ((pooled, pooled), (local, local)):
        assert loss.g_loss(fake, real) == loss.d_loss(real, fake)
        # The public mean over512 scores differs from scalar log2 by one FP32
        # reduction ulp. No custom payoff or zero-residual special case is used.
        torch.testing.assert_close(loss.g_loss(fake, real), torch.tensor(math.log(2)), atol=2e-7, rtol=0)
    fake_scalar = torch.tensor(.25, requires_grad=True)
    native = loss.g_loss(fake_scalar.reshape(1, 1), torch.zeros(1, 1))
    token = loss.g_loss(fake_scalar.expand(256, 1), torch.zeros(256, 1))
    assert torch.equal(native, token)
    assert torch.equal(torch.autograd.grad(native, fake_scalar)[0], torch.autograd.grad(token, fake_scalar)[0])


def test_heterogeneous_paired_game_jensen_and_exact_patch_gradient_factor():
    loss = get_recipe("e22_routed").make_loss()
    real = torch.tensor([[.5], [-.5]]).expand(2, 256)
    fake = torch.stack((torch.linspace(-1, 1, 256), torch.linspace(1, -1, 256))).requires_grad_(True)
    pooled_g = loss.g_loss(fake.mean(1, keepdim=True), real.mean(1, keepdim=True))
    token_g = loss.g_loss(fake.flatten()[:, None], real.flatten()[:, None])
    pooled_d = loss.d_loss(real.mean(1, keepdim=True), fake.mean(1, keepdim=True))
    token_d = loss.d_loss(real.flatten()[:, None], fake.flatten()[:, None])
    assert token_g > pooled_g and token_d > pooled_d
    gradient = torch.autograd.grad(token_g, fake)[0]
    expected = -(real - fake).sigmoid() / (2 * 256)
    assert torch.allclose(gradient, expected, atol=1e-10, rtol=1e-7)
    assert torch.equal(token_g, F.softplus(real - fake).mean())
    assert not torch.equal(token_g, loss.g_loss(fake.flatten()[:, None], real.flip(0).flatten()[:, None]))


def test_pooled_native_ka2_value_gradients_ema_and_record_remain_exact_before_learning():
    native, experimental = loop(), loop()
    initial = deepcopy(native_checkpoint(native))
    native_restore(experimental, initial)
    configure(experimental)
    assert state_digest(native_checkpoint(experimental)["policy"]) == state_digest(initial["policy"])
    panels = torch.linspace(-.125, .125, 4 * 256 * 16).reshape(4, 256, 16)
    fake = panels + torch.linspace(-.3, .4, 4 * 256 * 16).reshape(4, 256, 16)
    results = []
    for current in (native, experimental):
        policy = current.policy
        target = torch.zeros(4, 4, 32, 32)
        policy.begin_step(target, routed=RoutedBatch(current.context, target,
                          current.guard_context, current.guard_targets))
        policy.observe_critic_pair(panels, fake)
        penalty = apply_critic_penalty(policy.penalty, policy.D, panels, fake, current.context)
        grads = torch.autograd.grad(penalty, tuple(policy.D.parameters()), allow_unused=True)
        results.append((penalty.detach(), grads, deepcopy(policy.penalty.last_stats),
                        state_digest(policy.opt_d.state_dict()["regularizer"])))
    assert torch.equal(results[0][0], results[1][0])
    assert all((a is None and b is None) or torch.equal(a, b) for a, b in zip(results[0][1], results[1][1]))
    assert results[0][2:] == results[1][2:]


@pytest.mark.parametrize("shared", (False, True), ids=("legacy_routed", "shared_auto_settled"))
def test_full_native_two_update_resume_including_hold_and_matched_owned_streams(shared, monkeypatch):
    if shared:
        # Reuse the same real two-site128x4 public fixture with the currently
        # integrated routed profile, rather than changing any native owner.
        import test_experimental_e22_bank_game_trust as fixture
        original_recipe = fixture.get_recipe

        def shared_recipe(name, **options):
            return original_recipe(name, birth_death_backend="auto", reopen_guard="settled", **options)

        monkeypatch.setattr(fixture, "get_recipe", shared_recipe)
    native = loop()
    initial = deepcopy(native_checkpoint(native))
    native_rows = [native_training_update(native) for _ in range(5)]
    experimental = loop()
    with pytest.raises(ValueError):
        training_update(experimental)
    native_restore(experimental, initial)
    configure(experimental)
    tagged_initial = deepcopy(checkpoint(experimental))
    before = state_digest(tagged_initial)
    with pytest.raises(ValueError):
        restore(experimental, initial)
    assert state_digest(checkpoint(experimental)) == before
    wrong = deepcopy(tagged_initial)
    wrong["formulation"] = "context_mean_v1"
    with pytest.raises(ValueError):
        restore(experimental, wrong)
    rows = [training_update(experimental) for _ in range(3)]
    mid = deepcopy(checkpoint(experimental))
    rows += [training_update(experimental), training_update(experimental)]
    final = state_digest(checkpoint(experimental))
    assert rows[-1]["hold"] and rows[-1]["game_weight"] == .1
    assert all(row["game_aggregation"] == FORMULATION and row["dense_gradient_rows"] == 128 for row in rows)
    assert rows[0]["penalty"] == native_rows[0]["penalty"]
    for first, second in zip(rows, native_rows):
        assert all(first[key] == second[key] for key in ("batch_indices", "base_noise_sums", "paired_rng_digest",
            "dv12_rng_digest", "hold", "game_weight", "penalty_calls"))
    restore(experimental, mid)
    assert [training_update(experimental), training_update(experimental)] == rows[3:]
    assert state_digest(checkpoint(experimental)) == final
    assert experimental.policy.D.forward.__func__ is ConditionalTokenCritic.forward

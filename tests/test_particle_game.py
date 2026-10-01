"""Application tests for native velocity patches and conditional KA2 units."""

from copy import deepcopy

import pytest
import torch

from particlegan import get_recipe, init
from particlegan.ka2 import WARMUP_CALLS
from supra.particle_game import ConditionalTokenCritic, ConditionalTokenPenaltyView, apply_critic_penalty, patchify


@pytest.fixture(autouse=True)
def serial_backward():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.autograd.set_multithreading_enabled(False):
            yield
    finally:
        torch.set_num_threads(threads)


def test_native_patch_coordinates_preserve_the_complete_velocity_and_its_gradient():
    velocity = torch.arange(4096, dtype=torch.float32).reshape(1, 4, 32, 32).requires_grad_(True)
    patches = patchify(velocity)
    assert patches.shape == (1, 256, 16)
    expected = torch.tensor([0, 1, 32, 33, 1024, 1025, 1056, 1057,
                             2048, 2049, 2080, 2081, 3072, 3073, 3104, 3105], dtype=torch.float32)
    torch.testing.assert_close(patches[0, 0], expected, rtol=0, atol=0)
    restored = patches.reshape(1, 16, 16, 4, 2, 2).permute(0, 3, 1, 4, 2, 5).reshape_as(velocity)
    torch.testing.assert_close(restored, velocity, rtol=0, atol=0)
    patches.sum().backward()
    torch.testing.assert_close(velocity.grad, torch.ones_like(velocity), rtol=0, atol=0)


def pair(phase):
    with torch.random.fork_rng(devices=[]):
        critic = ConditionalTokenCritic(torch.ones(16), condition_dim=5, width=8, feature_dim=4).double()
        init.deterministic_orthogonal_(critic, seed=1)
    ema = deepcopy(critic)
    with torch.no_grad():
        for parameter in ema.parameters():
            parameter.mul_(.7)
    recipe = get_recipe("e22_routed", num_particles=8, z_dim=2, batch_size=3,
                        row_evidence_gate=False, particle_birth_death=False)
    optimizer = recipe.make_critic_optimizer(critic, ema_critic=ema, foreach=False)
    penalty = recipe.make_critic_penalty(optimizer, collect_stats=True, coeff=1.7, kappa=1e-4)
    if phase == "blend":
        optimizer.record.calls = WARMUP_CALLS - 1
        optimizer.record.observed_steps = WARMUP_CALLS - 1
        optimizer.record.anchor_started = True
        optimizer.record.last_sur = .1
    return critic, optimizer, penalty


def measure(tokens, phase):
    critic, optimizer, penalty = pair(phase)
    real = torch.linspace(-.2, .3, 48, dtype=torch.float64).reshape(3, 1, 16).repeat(1, tokens, 1)
    fake = real + .4
    condition = torch.linspace(-.2, .5, 15, dtype=torch.float64).reshape(3, 5)
    value = apply_critic_penalty(penalty, critic, real, fake, condition)
    gradients = torch.autograd.grad(value, tuple(critic.parameters()), allow_unused=True)
    return value.detach(), gradients, deepcopy(penalty.last_stats), optimizer.record.calls


@pytest.mark.parametrize("phase", ["a", "blend"])
def test_conditioned_token_penalty_and_EMA_keep_units_when_tokens_are_replicated(phase):
    value, gradients, stats, calls = measure(1, phase)
    for tokens in (8, 128, 256):
        actual, actual_gradients, actual_stats, actual_calls = measure(tokens, phase)
        torch.testing.assert_close(actual, value, rtol=3e-12, atol=1e-14)
        for actual_gradient, gradient in zip(actual_gradients, gradients):
            if gradient is None:
                assert actual_gradient is None
            else:
                torch.testing.assert_close(actual_gradient, gradient, rtol=4e-12, atol=1e-14)
        assert actual_calls == calls
        assert actual_stats["phase"] == stats["phase"] == phase
        assert actual_stats["applied"]
        assert actual_stats["prox"] == pytest.approx(stats["prox"], rel=4e-12, abs=1e-14)
    assert calls == (1 if phase == "a" else 800)
    # The penalty has a trainable source/time path, including through its EMA
    # counterpart, while the regularizer's differentiated input stays 16-wide.
    assert gradients[2] is not None and gradients[2].abs().sum() > 0
    assert (stats["prox"] > 0) == (phase == "blend")


def test_view_keeps_context_scores_and_critic_features_respond_to_conditioning():
    critic, _, _ = pair("a")
    error = torch.linspace(-.2, .3, 96, dtype=torch.float64).reshape(3, 2, 16)
    condition = torch.linspace(-.2, .5, 15, dtype=torch.float64).reshape(3, 5)
    view = ConditionalTokenPenaltyView(critic, 2)
    assert dict(view.named_children()) == {"critic": critic}
    torch.testing.assert_close(view(error.flatten(0, 1), condition),
                               critic(error, condition).repeat_interleave(2, 0), rtol=0, atol=0)
    assert not torch.equal(critic.features(error, condition), critic.features(error, condition + .2))

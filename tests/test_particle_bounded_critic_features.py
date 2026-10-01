"""Bounded critic extension retains the initial game and native KA2 units."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
pytest.importorskip("particlegan.routing")
from particlegan import Recipe
from experimental_e22_bounded_critic_features import (
    BoundedFeatureCritic, FORMULATION, checkpoint, extend_native_state, restore,
)
from supra.particle_game import ConditionalTokenCritic, apply_critic_penalty
from supra.particle_pilot import state_digest


def fixture():
    recipe = Recipe(name="e22_routed")
    with torch.random.fork_rng(devices=[]):
        native = ConditionalTokenCritic(torch.ones(16), condition_dim=3, width=12, feature_dim=5)
        capacity = BoundedFeatureCritic(torch.ones(16), condition_dim=3, width=12, feature_dim=5)
    optimizer = recipe.make_critic_optimizer(native, ema_critic=deepcopy(native), foreach=False)
    # Populate the old native Adam moments before migration.
    error = torch.linspace(-1, 1, 2 * 7 * 16).reshape(2, 7, 16)
    cond = torch.ones(2, 3)
    native(error, cond).sum().backward()
    optimizer.step()
    state, opt_state = extend_native_state(native.state_dict(), optimizer.state_dict(), capacity)
    capacity.load_state_dict(state, strict=True)
    cap_optimizer = recipe.make_critic_optimizer(capacity, ema_critic=deepcopy(capacity), foreach=False)
    cap_optimizer.load_state_dict(opt_state)
    return recipe, native, optimizer, capacity, cap_optimizer, error, cond


def test_initial_function_old_derivatives_moments_and_ema_are_exact():
    _, native, optimizer, capacity, cap_optimizer, error, cond = fixture()
    error = error.requires_grad_(True)
    a, b = native(error, cond), capacity(error, cond)
    assert torch.equal(a, b)
    assert torch.equal(native.features(error, cond), capacity.features(error, cond))
    left = torch.autograd.grad(a.sum(), (error,) + tuple(native.parameters()))
    right = torch.autograd.grad(b.sum(), (error,) + tuple(capacity.parameters()))
    assert all(torch.equal(x, y) for x, y in zip(left, right[:-1]))
    assert right[-1].norm() > 0
    migrated = cap_optimizer.state_dict()
    assert state_digest(migrated["state"]) == state_digest(optimizer.state_dict()["state"])
    assert len(migrated["param_groups"][0]["params"]) == len(optimizer.state_dict()["param_groups"][0]["params"]) + 1
    assert torch.equal(capacity.bypass[0], torch.zeros(5))
    assert torch.equal(optimizer.ema_critic(error, cond), cap_optimizer.ema_critic(error, cond))


def test_features_stay_bounded_and_native_ka2_sees_new_input_gradient():
    recipe, native, optimizer, capacity, cap_optimizer, error, cond = fixture()
    with torch.no_grad():
        capacity.bypass[0].fill_(1000.)
    bounded = capacity.features(error * 1e6, cond * 1e6)
    assert torch.isfinite(bounded).all() and bounded.abs().max() <= 2
    ordinary = recipe.make_critic_penalty(optimizer, collect_stats=True)
    changed = recipe.make_critic_penalty(cap_optimizer, collect_stats=True)
    a = apply_critic_penalty(ordinary, native, error * .1, error + .3, cond)
    b = apply_critic_penalty(changed, capacity, error * .1, error + .3, cond)
    assert not torch.equal(a, b)
    with torch.no_grad():
        capacity.bypass[0].fill_(.2)
    # The native input-gradient regularizer differentiates all the way into
    # the new bounded feature gain. It is not hidden outside the cap path.
    b = apply_critic_penalty(changed, capacity, error * .1, error + .3, cond)
    grad = torch.autograd.grad(b, capacity.bypass[0])[0]
    assert torch.isfinite(grad).all() and grad.norm() > 0


def test_explicit_checkpoint_tag_prevents_native_state_reinterpretation():
    _, native, optimizer, capacity, cap_optimizer, _, _ = fixture()
    ordinary = checkpoint(native, optimizer, formulation="native_critic_v1")
    with pytest.raises(ValueError, match="exact formulation"):
        restore(capacity, cap_optimizer, ordinary, formulation=FORMULATION)
    with pytest.raises(ValueError, match="class"):
        checkpoint(native, optimizer, formulation=FORMULATION)
    saved = checkpoint(capacity, cap_optimizer, formulation=FORMULATION)
    before = state_digest(saved)
    with torch.no_grad():
        capacity.bypass[0].add_(.5)
    restore(capacity, cap_optimizer, saved, formulation=FORMULATION)
    assert state_digest(checkpoint(capacity, cap_optimizer, formulation=FORMULATION)) == before

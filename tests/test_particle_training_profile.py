"""Shared PR223 controls have an explicit profile and strict recovery boundary."""
from copy import deepcopy

import pytest
import torch

from supra.particle_pilot import checkpoint, restore, state_digest
from supra.particle_training import (
    LEGACY_ROUTED_PROFILE, SHARED_ROUTED_PROFILE, make_training_loop,
    resolve_training_profile, training_update,
)
from supra.runtime import TARGETS, model_module


@pytest.fixture
def loops():
    mod = model_module()
    with torch.random.fork_rng():
        torch.manual_seed(7)
        model = mod.SupraDiT(d_model=8, depth=1, n_heads=2, ctx_dim=768,
                            mlp_ratio=2, num_tokens=256, patch=2)
        mod.attach_supra_lora(model, rank=16, alpha=16, targets=TARGETS)
    context = torch.zeros(4, 4100)
    context[:, 4099] = 1
    guard = context.clone()
    guard[:, 0] = 1
    holds = context.clone()
    holds[:, 0] = 2
    data = dict(text_contexts=torch.zeros(2, 3, 768), text_masks=torch.ones(2, 3),
                coordinate_scale=torch.ones(16), fit=dict(context=context, targets=torch.zeros(4, 4, 32, 32)),
                guard=dict(context=guard), holds=dict(context=holds))
    return (make_training_loop(model, data, device="cpu"),
            make_training_loop(model, data, device="cpu", profile=SHARED_ROUTED_PROFILE))


def test_new_runs_enable_shared_profile_and_resumes_keep_saved_profile():
    assert resolve_training_profile() == SHARED_ROUTED_PROFILE
    assert resolve_training_profile({}) == LEGACY_ROUTED_PROFILE
    assert resolve_training_profile({"particle_profile": SHARED_ROUTED_PROFILE}) == SHARED_ROUTED_PROFILE
    with pytest.raises(ValueError, match="profile differs"):
        resolve_training_profile({}, SHARED_ROUTED_PROFILE)
    with pytest.raises(ValueError, match="unsupported"):
        resolve_training_profile({"particle_profile": "atlas_routed"})


def test_shared_profile_retains_game_and_routed_ownership(loops):
    legacy, shared = loops
    assert "particle_profile" not in legacy.config
    assert shared.config["particle_profile"] == SHARED_ROUTED_PROFILE
    assert legacy.policy.reopen_guard is None
    assert shared.policy.reopen_guard is not None
    assert shared.policy.recipe.birth_death_backend == "auto"
    assert shared.policy.recipe.reopen_guard == "settled"
    assert shared.policy.routed_control is not None
    assert shared.config["output_error_guard"] is False
    assert shared.config["max_feature_context_harm"] == 0
    assert torch.equal(legacy.policy.table, shared.policy.table)
    training_update(shared)
    selection = checkpoint(shared)["policy"]["backend_selection"]
    assert selection["actual_backend"] == "routed"
    assert selection["sampling_backend"] == "routed"
    assert selection["selection_reason"] == "routed_rows_owns_controls"
    assert selection["generator_noise_factor"] == 1.


def test_shared_native_recovery_is_exact_and_rejects_cross_profile(loops):
    legacy, shared = loops
    training_update(shared)
    saved = checkpoint(shared)
    row = training_update(shared)
    expected = state_digest(checkpoint(shared))
    restore(shared, saved)
    assert training_update(shared) == row
    assert state_digest(checkpoint(shared)) == expected
    with pytest.raises(ValueError, match="configuration mismatch"):
        restore(shared, checkpoint(legacy))
    malformed = deepcopy(saved)
    malformed["policy"]["reopen_guard"]["epoch_rebases"] = -1
    with pytest.raises(ValueError):
        restore(shared, malformed)
    assert state_digest(checkpoint(shared)) == expected

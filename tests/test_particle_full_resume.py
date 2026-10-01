"""Extending the external budget must preserve the native game configuration."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from train_e22_supra_full import IMMUTABLE_PROVENANCE, resolve_training_architecture, validate_resume
from supra.particle_adapter import LINEAR_MODULATED_V2, NONLINEAR_V1
from supra.particle_pilot import checkpoint, initial_digest, restore
from supra.particle_training import make_training_loop
from supra.runtime import TARGETS, model_module


@pytest.fixture
def continuation():
    previous = {key: "pinned" for key in IMMUTABLE_PROVENANCE}
    previous.update(fixed_updates=1600, original_baseline_sha256="matched-1600",
                    config=dict(branch_lr=5e-5, output_error_guard=False,
                                preservation_game_weight=.1, probe_interval=100,
                                dataset_digest="original-data"))
    declared = deepcopy(previous)
    declared.update(fixed_updates=6400, original_baseline_sha256="matched-6400",
                    comparison="original 6400-update recipe")
    state = dict(config=deepcopy(previous["config"]), policy=dict(completed_steps=1600))
    return previous, declared, state


def test_only_external_budget_and_evaluation_baseline_can_change(continuation):
    assert validate_resume(*continuation) == 1600


def test_native_recipe_tuples_match_json_receipt_arrays(continuation):
    previous, declared, state = continuation
    previous["config"]["recipe"] = {"betas": [0., .999]}
    declared["config"]["recipe"] = {"betas": (0., .999)}
    state["config"]["recipe"] = {"betas": (0., .999)}
    assert validate_resume(previous, declared, state) == 1600


@pytest.mark.parametrize("field,value", [
    ("branch_lr", 1e-4), ("output_error_guard", True),
    ("preservation_game_weight", 1.), ("dataset_digest", "different-data"),
])
def test_rejects_game_or_dataset_change(continuation, field, value):
    previous, declared, state = continuation
    declared["config"][field] = value
    with pytest.raises(ValueError, match="resume provenance changed: config"):
        validate_resume(previous, declared, state)


def test_rejects_optimizer_update_and_checkpoint_config_mismatch(continuation):
    previous, declared, state = continuation
    declared["particlegan_commit"] = "another-optimizer"
    with pytest.raises(ValueError, match="particlegan_commit"):
        validate_resume(previous, declared, state)
    declared["particlegan_commit"] = previous["particlegan_commit"]
    state["config"]["probe_interval"] = 50
    with pytest.raises(ValueError, match="checkpoint configuration"):
        validate_resume(previous, declared, state)


@pytest.mark.parametrize("step", [0, 6400, 6401])
def test_rejects_reset_or_completed_horizon(continuation, step):
    previous, declared, state = continuation
    state["policy"]["completed_steps"] = step
    with pytest.raises(ValueError, match="precede"):
        validate_resume(previous, declared, state)


def test_architecture_defaults_and_explicit_legacy_resume():
    assert resolve_training_architecture() == LINEAR_MODULATED_V2
    assert resolve_training_architecture(requested=NONLINEAR_V1) == NONLINEAR_V1
    # Native checkpoints written before versioned architecture metadata must
    # keep their original formula even though fresh training defaults to V2.
    assert resolve_training_architecture({}) == NONLINEAR_V1
    assert resolve_training_architecture({}, NONLINEAR_V1) == NONLINEAR_V1
    assert resolve_training_architecture({"architecture": LINEAR_MODULATED_V2}) == LINEAR_MODULATED_V2


@pytest.mark.parametrize("saved,requested", [
    ({}, LINEAR_MODULATED_V2),
    ({"architecture": LINEAR_MODULATED_V2}, NONLINEAR_V1),
])
def test_resume_rejects_architecture_override(saved, requested):
    with pytest.raises(ValueError, match="architecture differs"):
        resolve_training_architecture(saved, requested)


def test_resume_rejects_unknown_architecture():
    with pytest.raises(ValueError, match="unsupported particle architecture"):
        resolve_training_architecture({"architecture": "unknown"})


def test_resume_provenance_rejects_reinterpreting_legacy_weights(continuation):
    previous, declared, state = continuation
    declared["config"]["architecture"] = LINEAR_MODULATED_V2
    with pytest.raises(ValueError, match="resume provenance changed: config"):
        validate_resume(previous, declared, state)


def test_fresh_architectures_share_native_initial_state_and_preserve_legacy_config():
    module = model_module()
    model = module.SupraDiT(d_model=8, depth=1, n_heads=2, ctx_dim=768,
                            mlp_ratio=2, num_tokens=256, patch=2)
    module.attach_supra_lora(model, rank=16, alpha=16, targets=TARGETS)
    context = torch.zeros(4, 4100)
    context[:, 4099] = 1
    data = dict(text_contexts=torch.zeros(2, 3, 768), text_masks=torch.ones(2, 3),
                coordinate_scale=torch.ones(16), fit=dict(context=context, targets=torch.zeros(4, 4, 32, 32)),
                guard=dict(context=context), holds=dict(context=context))
    legacy = make_training_loop(model, data, device="cpu", architecture=NONLINEAR_V1)
    current = make_training_loop(model, data, device="cpu")
    assert legacy.policy.G.architecture == NONLINEAR_V1
    assert current.policy.G.architecture == LINEAR_MODULATED_V2
    assert "architecture" not in legacy.config
    assert {key: value for key, value in current.config.items() if key != "architecture"} == legacy.config
    assert initial_digest(current) == initial_digest(legacy)
    saved = checkpoint(legacy)
    with pytest.raises(ValueError, match="checkpoint configuration mismatch"):
        restore(current, saved)
    restore(legacy, saved)

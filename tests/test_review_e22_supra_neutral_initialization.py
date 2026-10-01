"""Cheap corruption checks for the CPU-only Supra artifact reviewer."""
from copy import deepcopy
import json
import math

import pytest
import torch
from safetensors.torch import save_file

from scripts.review_e22_supra_neutral_initialization import (
    ARMS, COUNTS, JUDGES, MODES, Reviewer, canonical, check_checkpoint_contract,
    check_evaluation, review_export,
)


@pytest.fixture
def checkpoint():
    sites = [f"site{index}" for index in range(71)]
    config = dict(particle_init=MODES[ARMS[0]], architecture="gated_particle_v3",
        particle_profile="pr223_shared_routed_v1", initialization="particlegan.init.initialize_",
        initialization_method="sample_distributions_v1", training_schedule="fresh_editing_only_v1",
        preservation_game_weight=0., output_error_guard=False, max_feature_context_harm=0.,
        dataset_digest="held-input", recipe=dict(birth_death_backend="auto", reopen_guard="settled"), sites=sites)
    g = {f"weight{index}": torch.tensor([index + 1.], dtype=torch.float32) for index in range(284)}
    router = {f"weight{index}": torch.tensor([index + 1.], dtype=torch.float32) for index in range(142)}
    router["log_mass"] = torch.zeros(128)
    policy = dict(completed_steps=5120, recipe=config["recipe"], requires_grad=dict(
        generator={name: True for name in g}, router={name: name != "log_mass" for name in router}),
        roles=[["generator", "router", "table"], ["critic"]], models=dict(generator=g, router=router),
        table=torch.ones(128, 4), table_requires_grad=True,
        routing=dict(config=dict(sites=sites, max_context_harm=0., output_error_guard=False)))
    return dict(policy=policy, config=config)


def test_checkpoint_contract_rejects_wrong_clock(checkpoint):
    initial = deepcopy(checkpoint)
    check_checkpoint_contract(Reviewer(), checkpoint, initial, canonical(checkpoint["config"]), ARMS[0], 5120, "held-input")
    checkpoint["policy"]["completed_steps"] = 5119
    with pytest.raises(ValueError, match="clock/config"):
        check_checkpoint_contract(Reviewer(), checkpoint, initial, canonical(initial["config"]), ARMS[0], 5120, "held-input")


@pytest.mark.parametrize("field,value", [("max_context_harm", 1e-4), ("output_error_guard", True)])
def test_checkpoint_contract_rejects_changed_structural_guard(checkpoint, field, value):
    initial = deepcopy(checkpoint)
    checkpoint["policy"]["routing"]["config"][field] = value
    with pytest.raises(ValueError, match="routed population/guard"):
        check_checkpoint_contract(Reviewer(), checkpoint, initial, canonical(initial["config"]), ARMS[0], 5120, "held-input")


def test_summary_reduction_rejects_changed_mean():
    data, result = {}, {}
    sources = (4, 5, 15, 16, 17, 18)
    for pool, count in COUNTS.items():
        context = torch.zeros(count, 4098)
        context[:, 4097] = torch.tensor([sources[index % 6] for index in range(count)])
        data[pool] = dict(context=context)
        records = [dict(index=index, source_caption_id=int(context[index, 4097]), time=0.,
                        mse_diagnostic=.001, D1856=.7, D6400=.8) for index in range(count)]
        result[pool] = dict(records=records, count=count, rmse_diagnostic=math.sqrt(.001),
                            **{judge: sum(row[judge] for row in records) / count for judge in JUDGES})
    reviewed = check_evaluation(Reviewer(), result, data, "fixture")
    assert len(reviewed["test"]["subjects"]) == 6
    result["test"]["D1856"] += .01
    with pytest.raises(ValueError, match="game reduction"):
        check_evaluation(Reviewer(), result, data, "fixture")


def export_fixture(path, checkpoint, mutate_tensor=False, mode=None):
    policy = checkpoint["policy"]
    tensors = {"generator." + name: tensor.clone() for name, tensor in policy["models"]["generator"].items()}
    tensors.update({"router." + name: tensor.clone() for name, tensor in policy["models"]["router"].items() if name != "log_mass"})
    tensors.update({"bank.table": policy["table"].clone(), "bank.log_mass": policy["models"]["router"]["log_mass"].clone()})
    if mutate_tensor:
        tensors["generator.weight0"].add_(1)
    config = dict(architecture="gated_particle_v3", rank=16, z_dim=4, num_particles=128,
        cfg=3, sampling="clean", routed_geometry="mass_atoms_v1", served_source="fast", completed_steps=5120,
        sites=checkpoint["config"]["sites"],
        extra=dict(particle_init=MODES[ARMS[0]] if mode is None else mode, training_schedule="fresh_editing_only_v1"))
    save_file(tensors, path, metadata=dict(format="supra_particlegan_clean_v3", config=json.dumps(config),
                                         native_pins=json.dumps(dict(backend_sha256="held-backend"))))


def test_export_rejects_tensor_that_does_not_match_checkpoint(tmp_path, checkpoint):
    path = tmp_path / "adapter.safetensors"
    export_fixture(path, checkpoint)
    assert review_export(Reviewer(), path, checkpoint, ARMS[0], 5120, "held-backend")["tensors"] == 428
    export_fixture(path, checkpoint, mutate_tensor=True)
    with pytest.raises(ValueError, match="exact FAST export tensor"):
        review_export(Reviewer(), path, checkpoint, ARMS[0], 5120, "held-backend")


def test_export_rejects_wrong_initialization_tag(tmp_path, checkpoint):
    path = tmp_path / "adapter.safetensors"
    export_fixture(path, checkpoint, mode=MODES[ARMS[1]])
    with pytest.raises(ValueError, match="explicit fresh initialization"):
        review_export(Reviewer(), path, checkpoint, ARMS[0], 5120, "held-backend")

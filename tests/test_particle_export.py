"""Lean public serving export preserves branch tensors, represented mass and CFG."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from particlegan import E22Policy, RoutedCandidate, RoutedRows, get_recipe, init
from supra.particle_adapter import (
    LINEAR_MODULATED_V2, NONLINEAR_V1, SupraParticleHost, SupraParticleRouter,
    adapter_input_dims, adapter_sites,
)
from supra.particle_export import (
    FORMAT, FORMAT_V2, _EncodedCondition, _native_forward, _unused_features, export_particle_adapter,
    export_served_adapter, load_particle_adapter,
)
from supra.runtime import TARGETS, model_module


def fixture(architecture=NONLINEAR_V1):
    module = model_module()
    model = module.SupraDiT(d_model=8, depth=2, n_heads=2, ctx_dim=12,
                            mlp_ratio=2, num_tokens=4, patch=2)
    module.attach_supra_lora(model, rank=2, alpha=2, targets=TARGETS)
    sites = adapter_sites(model)
    generator = SupraParticleHost(model, rank=2, sites=sites, architecture=architecture)
    router = SupraParticleRouter(num_particles=8, site_input_dims=adapter_input_dims(model))
    critic = nn.Linear(4, 1)
    init.deterministic_orthogonal_(generator, seed=0)
    init.deterministic_orthogonal_(router, seed=3)
    recipe = get_recipe("e22_routed", num_particles=8, z_dim=4, batch_size=2,
                        row_evidence_gate=False, particle_birth_death=False)
    table = init.deterministic_orthogonal_(recipe.make_prior()).z
    with torch.no_grad():
        router.log_mass.copy_(torch.linspace(-1, 1, 8))
        router.log_mass[0] = -torch.inf
        for branch in generator.particle_branches():
            branch.up.weight.fill_(.03)
    opt_g = recipe.make_generator_optimizer([
        dict(params=[p for p in generator.parameters() if p.requires_grad]),
        dict(params=list(router.parameters())), dict(params=[table])], latent_table=table, foreach=False)
    opt_d = recipe.make_critic_optimizer(critic, ema_critic=deepcopy(critic), foreach=False)
    policy = E22Policy(recipe, generator, critic, table=table, router=router,
                       generator_optimizer=opt_g, critic_optimizer=opt_d,
                       roles=[["generator", "router", "table"], ["critic"]],
                       routed_rows=RoutedRows(model_forward=_native_forward, features=_unused_features, sites=sites))
    z = torch.linspace(-.4, .7, 128).reshape(2, 4, 4, 4)
    ctx = torch.linspace(-.2, .8, 72).reshape(2, 3, 12)
    inputs = (z, torch.tensor([.2, .7]), ctx, torch.ones(2, 3), torch.zeros(1, 3, 12), torch.ones(1, 3))
    return model, policy, inputs


def public_clean(served, inputs):
    z, t, ctx, mask, uctx, umask = inputs
    condition = _EncodedCondition(z, ctx, mask, uctx, umask, 1., 3.)
    context = torch.cat((z.flatten(1), t[:, None]), dim=1)
    candidate = served.routing.candidate_for(served.models, served.table, averaged=served.source == "averaged")
    return served.routing.forward({**served.models, "conditioning": condition}, context, candidate)


@pytest.mark.parametrize("architecture", [NONLINEAR_V1, LINEAR_MODULATED_V2])
def test_public_served_export_reload_preserves_clean_velocity_mass_and_zero_strength(tmp_path, architecture):
    model, policy, inputs = fixture(architecture)
    served = policy.served_model()
    expected = public_clean(served, inputs)
    # A public snapshot owns independent tensors and must not export later
    # changes to the live training bank/router/generator.
    with torch.no_grad():
        policy.table.add_(.4)
        policy.router.log_mass[1:].add_(.8)
        policy.G.particle_branches()[0].up.weight.add_(.1)
    path = tmp_path / "particle.safetensors"
    receipt = export_served_adapter(served, path)
    adapter = load_particle_adapter(model, path)
    assert adapter.generator.architecture == architecture
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
    if architecture == NONLINEAR_V1:
        assert metadata["format"] == FORMAT
        assert "architecture" not in receipt["config"]
    else:
        assert metadata["format"] == FORMAT_V2
        assert receipt["config"]["architecture"] == architecture
    assert receipt["tensors"] == 6 * len(served.generator.sites) + 2
    assert torch.equal(adapter.table, served.table)
    assert torch.equal(adapter.router.log_mass, served.router.log_mass)
    assert torch.equal(adapter.velocity(*inputs), expected)
    assert not any(parameter.requires_grad for parameter in adapter.parameters())
    state = load_file(str(path))
    assert all(".base." not in key and "critic" not in key and "teacher" not in key for key in state)
    assert path.stat().st_size < 100_000
    for branch in model.modules():
        if hasattr(branch, "multiplier"):
            branch.multiplier = 0.
    z, t, ctx, mask, uctx, umask = inputs
    cond, uncond = model(torch.cat((z, z)), torch.cat((t, t)),
                         torch.cat((ctx, uctx.expand(2, -1, -1))),
                         torch.cat((mask, umask.expand(2, -1)))).chunk(2)
    assert torch.equal(adapter.velocity(*inputs, strength=0), uncond + 3 * (cond - uncond))
    # Arbitrary encoded prompts are supplied per inference call, not tied to
    # the captions or caption IDs stored in a training checkpoint.
    assert not torch.equal(adapter.velocity(*inputs), adapter.velocity(z, t, ctx + .2, mask, uctx, umask))


def test_loop_export_and_strict_rejection_of_wrong_shapes_and_native_pins(tmp_path):
    model, policy, _ = fixture()
    path = tmp_path / "particle.safetensors"
    export_particle_adapter(SimpleNamespace(policy=policy), path)
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
    state = load_file(str(path))
    state["bank.table"] = state["bank.table"][:, :3].contiguous()
    broken = tmp_path / "shape.safetensors"
    save_file(state, str(broken), metadata=metadata)
    with pytest.raises(ValueError, match="shape or dtype"):
        load_particle_adapter(model, broken)
    pins = json.loads(metadata["native_pins"])
    pins["model_revision"] = "wrong"
    metadata["native_pins"] = json.dumps(pins)
    save_file(load_file(str(path)), str(broken), metadata=metadata)
    with pytest.raises(ValueError, match="pins differ"):
        load_particle_adapter(model, broken)


@pytest.mark.parametrize("export_format,architecture,message", [
    (FORMAT, LINEAR_MODULATED_V2, "architecture and format version differ"),
    (FORMAT_V2, NONLINEAR_V1, "architecture and format version differ"),
    (FORMAT_V2, None, "architecture and format version differ"),
    (FORMAT, "unknown", "unsupported particle architecture"),
    (FORMAT_V2, "unknown", "unsupported particle architecture"),
    ("supra_particlegan_clean_v3", LINEAR_MODULATED_V2, "unsupported Supra particle export format"),
])
def test_rejects_ambiguous_or_mismatched_architecture_metadata(tmp_path, export_format, architecture, message):
    model, policy, _ = fixture(LINEAR_MODULATED_V2)
    path = tmp_path / "particle.safetensors"
    export_particle_adapter(SimpleNamespace(policy=policy), path)
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
    config = json.loads(metadata["config"])
    config.pop("architecture")
    if architecture is not None:
        config["architecture"] = architecture
    metadata.update(format=export_format, config=json.dumps(config))
    save_file(load_file(str(path)), str(path), metadata=metadata)
    with pytest.raises(ValueError, match=message):
        load_particle_adapter(model, path)


def test_explicit_legacy_architecture_retains_legacy_formula(tmp_path):
    model, policy, inputs = fixture()
    path = tmp_path / "legacy.safetensors"
    served = policy.served_model()
    export_served_adapter(served, path)
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
    config = json.loads(metadata["config"])
    config["architecture"] = NONLINEAR_V1
    metadata["config"] = json.dumps(config)
    save_file(load_file(str(path)), str(path), metadata=metadata)
    assert torch.equal(load_particle_adapter(model, path).velocity(*inputs), public_clean(served, inputs))


def test_export_rejects_mixed_branch_formulas(tmp_path):
    _, policy, _ = fixture(LINEAR_MODULATED_V2)
    served = policy.served_model()
    served.generator.particle_branches()[0].architecture = NONLINEAR_V1
    with pytest.raises(ValueError, match="branch architecture differs"):
        export_served_adapter(served, tmp_path / "mixed.safetensors")

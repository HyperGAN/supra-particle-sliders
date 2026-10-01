"""All native Supra attention/text sites share one bank and preserve CFG units."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from supra.particle_adapter import (
    GATED_PARTICLE_V3, LINEAR_MODULATED_V2, NONLINEAR_V1, SITES, SupraParticleHost, SupraParticleRouter,
    adapter_input_dims, adapter_sites, validate_particle_architecture,
)
from supra.runtime import TARGETS, model_module

pg_routing = pytest.importorskip("particlegan.routing")
from particlegan import get_recipe, init


class InspectExecution(pg_routing.RoutedExecution):
    def __init__(self, sites, candidate, batch_size, perturb_fn=None):
        super().__init__(sites, candidate, batch_size, perturb_fn)
        self.calls = []

    def mix(self, site_name, logits):
        self.calls.append((site_name, logits.detach().clone()))
        return super().mix(site_name, logits)


def tiny_native(depth=2):
    module = model_module()
    model = module.SupraDiT(d_model=8, depth=depth, n_heads=2, ctx_dim=12,
                            mlp_ratio=2, num_tokens=4, patch=2)
    module.attach_supra_lora(model, rank=2, alpha=2, targets=TARGETS)
    return model


def fixture(architecture=NONLINEAR_V1):
    model = tiny_native()
    sites = adapter_sites(model)
    host = SupraParticleHost(model, rank=2, sites=sites, architecture=architecture)
    router = SupraParticleRouter(num_particles=8, site_input_dims=adapter_input_dims(model))
    init.deterministic_orthogonal_(host, seed=0)
    init.deterministic_orthogonal_(router, seed=3)
    host.zero_particle_outputs()
    table = init.deterministic_orthogonal_(get_recipe("e22_routed", num_particles=8, z_dim=4).make_prior()).z
    candidate = pg_routing.RoutedCandidate(table, router.log_mass, {"log_mass": router.log_mass})
    z = torch.linspace(-.4, .7, 128).reshape(2, 4, 4, 4)
    ctx = torch.linspace(-.2, .8, 72).reshape(2, 3, 12)
    inputs = (z, torch.tensor([.2, .7]), ctx, torch.ones(2, 3), torch.zeros(1, 3, 12), torch.ones(1, 3))
    return model, host, router, candidate, inputs


def forward(host, router, candidate, inputs, *, strength=1, perturb=None, cfg=3):
    execution = InspectExecution(host.sites, candidate, len(inputs[0]), perturb)
    result = host.forward_routed(*inputs, candidate, execution, router, strength=strength, cfg=cfg)
    usage = execution.finish()
    execution.close()
    return result, usage, execution.calls


def base_cfg(model, inputs):
    for branch in model.modules():
        if hasattr(branch, "multiplier"):
            branch.multiplier = 0
    z, t, ctx, mask, uncond, umask = inputs
    conditional, unconditional = model(torch.cat((z, z)), torch.cat((t, t)),
                                      torch.cat((ctx, uncond.expand(2, -1, -1))),
                                      torch.cat((mask, umask.expand(2, -1)))).chunk(2)
    return unconditional + 3 * (conditional - unconditional)


def test_sites_follow_actual_native_execution_and_heterogeneous_token_lengths():
    _, host, router, candidate, inputs = fixture()
    expected = ("ctx_proj", *tuple(f"blocks.{block}.{projection}" for block in range(2)
                                  for projection in ("self_attn.qkv", "self_attn.proj", "cross_attn.q",
                                                     "cross_attn.kv", "cross_attn.proj")))
    assert host.sites == router.sites == expected
    assert router.site_input_dims == {site: 12 if site == "ctx_proj" else 8 for site in expected}
    output, usage, calls = forward(host, router, candidate, inputs)
    assert output.shape == inputs[0].shape
    assert tuple(site for site, _ in calls) == expected
    for site, logits in calls:
        text_site = site == "ctx_proj" or site.endswith("cross_attn.kv")
        assert logits.shape == (2, 2, 3 if text_site else 4, 8)
    assert usage.shape == (2, 8)
    assert torch.allclose(usage.sum(-1), torch.ones(2))
    assert all(branch.frame is None for branch in host.particle_branches())


@pytest.mark.parametrize("architecture", (NONLINEAR_V1, LINEAR_MODULATED_V2, GATED_PARTICLE_V3))
def test_all_branches_are_fresh_and_zero_start_and_strength_zero_match_base(architecture):
    model = tiny_native()
    for branch in model.modules():
        if hasattr(branch, "multiplier"):
            with torch.no_grad():
                branch.down.weight.fill_(123)
                branch.up.weight.fill_(321)
    original = deepcopy(model.state_dict())
    host = SupraParticleHost(model, rank=2, sites=adapter_sites(model), architecture=architecture)
    router = SupraParticleRouter(num_particles=8, site_input_dims=adapter_input_dims(model))
    init.deterministic_orthogonal_(host, seed=0)
    host.zero_particle_outputs()
    _, _, _, candidate, inputs = fixture()
    candidate = pg_routing.RoutedCandidate(candidate.table, router.log_mass, {"log_mass": router.log_mass})
    expected = base_cfg(model, inputs)
    assert torch.equal(forward(host, router, candidate, inputs)[0], expected)
    with torch.no_grad():
        for branch in host.particle_branches():
            branch.up.weight.fill_(.1)
    assert torch.equal(forward(host, router, candidate, inputs, strength=0)[0], expected)
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in original.items())
    assert not any(hasattr(module, "multiplier") for module in host.model.modules())
    assert all(branch.down.weight.abs().max() < 123 for branch in host.particle_branches())
    trainable = [name for name, parameter in host.named_parameters() if parameter.requires_grad]
    assert len(trainable) == 4 * len(host.sites)
    assert all(".base." not in name for name in trainable)


@pytest.mark.parametrize("architecture", (NONLINEAR_V1, LINEAR_MODULATED_V2, GATED_PARTICLE_V3))
def test_shared_bank_gradient_and_sequential_perturbations_reach_native_output(architecture):
    _, host, router, candidate, inputs = fixture(architecture)
    with torch.no_grad():
        for branch in host.particle_branches():
            branch.up.weight.fill_(.02)
    output, _, calls = forward(host, router, candidate, inputs)
    output.square().mean().backward()
    assert candidate.table.grad is not None
    assert (candidate.table.grad.norm(dim=-1) > 0).all()
    assert all(query.weight.grad is not None for query in router.site_queries.values())
    assert all(query.weight.grad.norm() > 0 for query in router.site_queries.values())
    shifted, _, changed_calls = forward(host, router, candidate, inputs, perturb=lambda codes: codes + .3)
    assert not torch.equal(output, shifted)
    # Later queries recompute from preceding particle edits rather than cache.
    assert not torch.equal(calls[3][1], changed_calls[3][1])


@pytest.mark.parametrize("architecture", (NONLINEAR_V1, LINEAR_MODULATED_V2, GATED_PARTICLE_V3))
def test_full_site_host_state_copy_owns_query_modules_and_remains_independent(architecture):
    _, host, router, candidate, inputs = fixture(architecture)
    cloned_host, cloned_router = deepcopy(host), deepcopy(router)
    assert tuple(cloned_router.site_input_dims) == host.sites
    assert cloned_host.architecture == architecture
    assert all(branch.architecture == architecture for branch in cloned_host.particle_branches())
    assert torch.equal(forward(host, router, candidate, inputs)[0],
                       forward(cloned_host, cloned_router, candidate, inputs)[0])
    for left, right in zip(router.parameters(), cloned_router.parameters()):
        assert left is not right
    with torch.no_grad():
        cloned_host.particle_branches()[0].up.weight.fill_(.03)
    assert not torch.equal(forward(host, router, candidate, inputs)[0],
                           forward(cloned_host, cloned_router, candidate, inputs)[0])


def test_legacy_defaults_keep_exact_checkpoint_keys():
    router = SupraParticleRouter(input_dim=8)
    assert router.sites == SITES
    assert set(router.state_dict()) == {"first_query.weight", "first_query.bias", "second_query.weight",
                                        "second_query.bias", "log_mass"}
    host = SupraParticleHost(tiny_native(depth=14), rank=2)
    assert host.sites == SITES
    assert len(host.particle_branches()) == 2
    assert host.architecture == NONLINEAR_V1
    assert all(branch.architecture == NONLINEAR_V1 for branch in host.particle_branches())


class FixedCodes:
    """Separate the projection Jacobian from the router's input dependence."""

    def __init__(self, codes):
        self.codes = codes

    def mix(self, site, logits):
        return self.codes


def projected_with_fixed_codes(branch, router, candidate, x, codes):
    branch.frame = SimpleNamespace(candidate=candidate, routing=FixedCodes(codes),
                                   router=router, batch_size=len(x), guided=False, strength=1.)
    try:
        return branch(x)
    finally:
        branch.frame = None


def test_legacy_projection_formula_is_exact_and_modes_keep_same_tensor_schema():
    _, host, router, candidate, _ = fixture()
    branch = host.particle_branches()[0]
    with torch.no_grad():
        branch.up.weight.fill_(.02)
    x = torch.linspace(-.7, .9, 36).reshape(1, 3, 12)
    codes = torch.linspace(-.4, .6, 12).reshape(1, 3, 4)
    hidden = branch.down(x)
    expected = branch.base(x) + branch.up(branch.bridge(torch.cat((hidden, codes), dim=-1)).tanh())
    assert torch.equal(projected_with_fixed_codes(branch, router, candidate, x, codes), expected)
    _, modern, _, _, _ = fixture(LINEAR_MODULATED_V2)
    assert {key: value.shape for key, value in host.state_dict().items()} == {
        key: value.shape for key, value in modern.state_dict().items()}


def test_saturated_particle_modulation_keeps_input_influence_and_gradient():
    _, old, router, candidate, _ = fixture()
    _, modern, _, _, _ = fixture(LINEAR_MODULATED_V2)
    old_branch, new_branch = old.particle_branches()[0], modern.particle_branches()[0]
    with torch.no_grad():
        old_branch.base.weight.zero_()
        if old_branch.base.bias is not None:
            old_branch.base.bias.zero_()
        old_branch.down.weight.zero_()
        old_branch.down.weight[:, :2].copy_(torch.eye(2))
        old_branch.bridge.weight.zero_()
        old_branch.bridge.bias.fill_(100.)
        old_branch.up.weight.zero_()
        old_branch.up.weight[:2].copy_(torch.eye(2))
        new_branch.load_state_dict(old_branch.state_dict())
    codes = torch.zeros(1, 3, 4)
    x = torch.linspace(-.7, .9, 36).reshape(1, 3, 12).requires_grad_()
    old_output = projected_with_fixed_codes(old_branch, router, candidate, x, codes)
    new_output = projected_with_fixed_codes(new_branch, router, candidate, x, codes)
    old_gradient, = torch.autograd.grad(old_output.sum(), x)
    new_gradient, = torch.autograd.grad(new_output.sum(), x)
    assert torch.equal(old_gradient, torch.zeros_like(x))
    expected_gradient = torch.zeros_like(x)
    expected_gradient[..., :2] = 1.
    assert torch.equal(new_gradient, expected_gradient)
    changed_input = x.detach() + .25
    assert torch.equal(projected_with_fixed_codes(old_branch, router, candidate, changed_input, codes), old_output)
    assert not torch.equal(projected_with_fixed_codes(new_branch, router, candidate, changed_input, codes), new_output)


def test_gated_particles_change_input_basis_with_identical_public_tensor_initialization():
    model, modern, router, candidate, _ = fixture(LINEAR_MODULATED_V2)
    gated = SupraParticleHost(model, rank=2, sites=modern.sites, architecture=GATED_PARTICLE_V3)
    init.deterministic_orthogonal_(gated, seed=0)
    gated.zero_particle_outputs()
    assert all(torch.equal(value, gated.state_dict()[name])
               for name, value in modern.state_dict().items())
    branch = gated.particle_branches()[0]
    with torch.no_grad():
        branch.base.weight.zero_()
        branch.base.bias.zero_()
        branch.down.weight.zero_()
        branch.down.weight[:, :2].copy_(torch.eye(2))
        branch.bridge.weight.zero_()
        branch.bridge.bias.zero_()
        branch.bridge.weight[:, 2:4].copy_(torch.eye(2))
        branch.up.weight.zero_()
        branch.up.weight[:2].copy_(torch.eye(2))
    x = torch.linspace(-.7, .9, 36).reshape(1, 3, 12).requires_grad_()
    codes = torch.zeros(1, 3, 4)
    changed_codes = codes + .5
    output = projected_with_fixed_codes(branch, router, candidate, x, codes)
    changed = projected_with_fixed_codes(branch, router, candidate, x, changed_codes)
    gain = 1 + torch.tanh(torch.tensor(.5))
    assert torch.equal(output[..., :2], x[..., :2])
    assert torch.allclose(changed[..., :2], gain * x[..., :2])
    native_jacobian = torch.autograd.grad(output.sum(), x)[0]
    changed_jacobian = torch.autograd.grad(changed.sum(), x)[0]
    assert torch.allclose(changed_jacobian, gain * native_jacobian)
    # Codes act on the input features instead of adding a constant output.
    assert torch.equal(projected_with_fixed_codes(branch, router, candidate,
                                                  torch.zeros_like(x), changed_codes),
                       torch.zeros_like(changed))


@pytest.mark.parametrize("architecture", (None, "linear", "", 2))
def test_reject_unsupported_architecture(architecture):
    with pytest.raises(ValueError, match="unsupported particle architecture"):
        validate_particle_architecture(architecture)
    with pytest.raises(ValueError, match="unsupported particle architecture"):
        SupraParticleHost(tiny_native(), architecture=architecture)


def test_reject_mismatched_site_order_and_invalid_dimensions():
    _, host, router, candidate, inputs = fixture()
    wrong = SupraParticleRouter(num_particles=8, site_input_dims=dict(reversed(tuple(router.site_input_dims.items()))))
    with pytest.raises(ValueError, match="same ordered"):
        forward(host, wrong, candidate, inputs)
    with pytest.raises(ValueError, match="positive integers"):
        SupraParticleRouter(site_input_dims={"ctx_proj": 0})
    with pytest.raises(ValueError, match="distinct"):
        SupraParticleHost(tiny_native(), sites=("ctx_proj", "ctx_proj"))

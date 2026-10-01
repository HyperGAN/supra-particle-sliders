"""Experimental key/value separation retains native sequential particle use."""
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
pytest.importorskip("particlegan.routing")
from experimental_e22_fixed_rms_keys import (
    FORMAT, ROUTING, FixedRMSKeyProjection, experimental_checkpoint,
    fixed_rms_keys, install_experimental_routing, restore_experimental,
)
from test_particle_full_adapter import fixture, forward
from supra.particle_adapter import LINEAR_MODULATED_V2
from supra.particle_pilot import state_digest


def install(host):
    loop = SimpleNamespace(policy=SimpleNamespace(G=host, ema_G=deepcopy(host)), config={})
    install_experimental_routing(loop)
    return loop


def test_native_mass_mix_uses_original_values_and_has_both_table_gradient_paths():
    import particlegan.routing as native
    _, host, router, candidate, inputs = fixture(LINEAR_MODULATED_V2)
    loop = install(host)
    with torch.no_grad():
        for branch in host.particle_branches():
            branch.up.weight.fill_(.02)
        candidate.table[0].mul_(7.)
        router.log_mass.copy_(torch.linspace(-1., 1., len(candidate.table)))
    # A native execution witnesses the first real site, including its CFG axes.
    captured = {}
    class Capture(native.RoutedExecution):
        def mix(self, site, logits):
            result = super().mix(site, logits)
            if site == host.sites[0]:
                captured.update(logits=logits, codes=result)
            return result
    execution = Capture(host.sites, candidate, len(inputs[0]), None)
    output = host.forward_routed(*inputs, candidate, execution, router, cfg=3)
    execution.finish(); execution.close()
    expected = (captured["logits"] + candidate.log_mass).softmax(-1) @ candidate.table
    assert torch.equal(captured["codes"], expected)
    keys = fixed_rms_keys(candidate.table)
    assert torch.allclose(keys.norm(dim=-1), torch.full((len(keys),), 2.))
    assert not torch.equal(expected, (captured["logits"] + candidate.log_mass).softmax(-1) @ keys)
    key_gradient = torch.autograd.grad(captured["logits"].square().mean(), candidate.table, retain_graph=True)[0]
    assert key_gradient.norm() > 0
    # Normalized selection has no radial bank derivative. Physical value
    # decoding retains radial sensitivity and thus cannot be normalized away.
    assert torch.allclose((key_gradient * candidate.table).sum(-1),
                          torch.zeros(len(keys)), atol=1e-7, rtol=0)
    code_gradient = torch.autograd.grad(expected.square().mean(), candidate.table, retain_graph=True)[0]
    assert (code_gradient * candidate.table).sum(-1).abs().max() > 1e-5
    output.square().mean().backward()
    assert (candidate.table.grad.norm(dim=1) > 0).all()
    assert all(query.weight.grad is not None and query.weight.grad.norm() > 0
               for query in router.site_queries.values())
    assert all(type(branch) is FixedRMSKeyProjection for branch in loop.policy.ema_G.particle_branches())


def test_particle_perturbation_changes_final_output_and_later_queries_and_clones_keep_law():
    _, host, router, candidate, inputs = fixture(LINEAR_MODULATED_V2)
    install(host)
    with torch.no_grad():
        for branch in host.particle_branches():
            branch.up.weight.fill_(.02)
    clean, _, calls = forward(host, router, candidate, inputs)
    perturbed, _, changed = forward(host, router, candidate, inputs, perturb=lambda codes: codes + .3)
    assert not torch.equal(clean, perturbed)
    assert not torch.equal(calls[3][1], changed[3][1])
    assert torch.equal(clean, forward(deepcopy(host), deepcopy(router), candidate, inputs)[0])
    assert all(branch.frame is None for branch in host.particle_branches())


def test_zero_and_tiny_keys_are_finite_and_preserve_zero_strength_native_parity():
    values = torch.tensor([[0., 0., 0., 0.], [1e-20, 0., 0., 0.], [2., 3., -1., 4.]], requires_grad=True)
    keys = fixed_rms_keys(values)
    assert torch.isfinite(keys).all()
    keys.sum().backward()
    assert torch.isfinite(values.grad).all()
    _, host, router, candidate, inputs = fixture(LINEAR_MODULATED_V2)
    native_zero = forward(host, router, candidate, inputs, strength=0)[0]
    install(host)
    assert torch.equal(native_zero, forward(host, router, candidate, inputs, strength=0)[0])


def test_dedicated_checkpoint_rejects_missing_or_misdescribed_law():
    _, host, _, _, _ = fixture(LINEAR_MODULATED_V2)
    loop = install(host)
    with pytest.raises(ValueError, match="not a supported"):
        restore_experimental(loop, {"policy": {}, "config": {}})
    with pytest.raises(ValueError, match="does not describe"):
        restore_experimental(loop, dict(format=FORMAT, experimental_routing=ROUTING,
                                       native={"config": {}}))
    before = state_digest(host.state_dict())
    host.particle_branches()[0].__class__ = type(fixture(LINEAR_MODULATED_V2)[1].particle_branches()[0])
    with pytest.raises(ValueError, match="FAST and averaged"):
        experimental_checkpoint(loop)
    assert state_digest(host.state_dict()) == before

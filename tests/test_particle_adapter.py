"""Whole-model routing ownership, CFG correlation and base preservation."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from supra.particle_adapter import SITES, SupraParticleHost, SupraParticleRouter

pg_routing = pytest.importorskip("particlegan.routing")


class _LoRA(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Linear(4, 4)
        self.down = nn.Linear(4, 2, bias=False)
        self.up = nn.Linear(2, 4, bias=False)
        self.multiplier = .7
        with torch.no_grad():
            self.base.weight.copy_(torch.eye(4) * .95)
            self.base.bias.fill_(.01)
            self.down.weight.fill_(.02)
            self.up.weight.fill_(.03)

    def forward(self, x):
        return self.base(x) + self.multiplier * self.up(self.down(x))


class _TinySupra(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Module() for _ in range(14)])
        for block in self.blocks:
            block.cross_attn = nn.Module()
            block.cross_attn.proj = _LoRA()
        self.fail = False

    def forward(self, z, t, ctx, mask):
        hidden = z.flatten(2).transpose(1, 2) + .1 * ctx + .01 * t[:, None, None]
        for index, block in enumerate(self.blocks):
            hidden = block.cross_attn.proj(hidden)
            if self.fail and index == 12:
                raise RuntimeError("intentional host failure")
        return hidden.transpose(1, 2).reshape_as(z)


class _InspectExecution(pg_routing.RoutedExecution):
    def __init__(self, candidate, batch_size, perturb_fn=None):
        super().__init__(SITES, candidate, batch_size, perturb_fn)
        self.calls = []

    def mix(self, site_name, logits):
        self.calls.append((site_name, logits.detach().clone()))
        return super().mix(site_name, logits)


def _fixture():
    original = _TinySupra()
    host = SupraParticleHost(original, rank=2, z_dim=4)
    router = SupraParticleRouter(num_particles=8, input_dim=4)
    with torch.no_grad():
        for query in (router.first_query, router.second_query):
            query.weight.fill_(.025)
            query.bias.fill_(.1)
        for branch in host.particle_branches():
            branch.down.weight.fill_(.08)
            branch.bridge.weight.fill_(.05)
            branch.bridge.bias.zero_()
            branch.up.weight.fill_(.02)
    candidate = pg_routing.RoutedCandidate(
        torch.linspace(-1, 1, 32).reshape(8, 4), router.log_mass,
        {"log_mass": router.log_mass})
    z = torch.linspace(-.5, .7, 16).reshape(2, 4, 1, 2)
    ctx = torch.linspace(-.2, .8, 16).reshape(2, 2, 4)
    uncond = torch.zeros(1, 2, 4)
    inputs = (z, torch.tensor([.2, .7]), ctx, torch.ones(2, 2), uncond, torch.ones(1, 2))
    return SimpleNamespace(original=original, host=host, router=router, candidate=candidate, inputs=inputs)


def _forward(fixture, candidate=None, strength=1, perturb=None, cfg=None):
    candidate = fixture.candidate if candidate is None else candidate
    execution = _InspectExecution(candidate, 2, perturb)
    output = fixture.host.forward_routed(*fixture.inputs, candidate, execution, fixture.router,
                                          strength=strength, cfg=cfg)
    usage = execution.finish()
    execution.close()
    return output, usage, execution.calls


def test_frozen_published_host_and_strength_zero_equality():
    fixture = _fixture()
    original_state = deepcopy(fixture.original.state_dict())
    trainable = [name for name, value in fixture.host.named_parameters() if value.requires_grad]
    assert len(trainable) == 8
    assert all(any(site in name for site in SITES) for name in trainable)
    assert all(".base." not in name for name in trainable)
    for module in fixture.original.modules():
        if isinstance(module, _LoRA):
            module.multiplier = 0
    z, t, ctx, mask, uncond, umask = fixture.inputs
    expected = fixture.original(torch.cat((z, z)), torch.cat((t, t)),
                                torch.cat((ctx, uncond.expand(2, -1, -1))),
                                torch.cat((mask, umask.expand(2, -1))))
    cond, unconditional = expected.chunk(2)
    expected = unconditional + 3 * (cond - unconditional)
    output, _, _ = _forward(fixture, strength=0)
    assert torch.equal(output, expected)
    assert all(torch.equal(value, fixture.original.state_dict()[key]) for key, value in original_state.items())
    assert all(branch.frame is None for branch in fixture.host.particle_branches())


def test_shared_candidate_and_actual_perturbed_codes_change_complete_cfg_forward():
    fixture = _fixture()
    output, usage, calls = _forward(fixture)
    assert [name for name, _ in calls] == list(SITES)
    assert all(logits.shape == (2, 2, 2, 8) for _, logits in calls)
    assert usage.shape == (2, 8)
    assert torch.allclose(usage.sum(-1), torch.ones(2))
    changed = pg_routing.RoutedCandidate(fixture.candidate.table + 1, fixture.router.log_mass,
                                         {"log_mass": fixture.router.log_mass})
    shifted, _, changed_calls = _forward(fixture, candidate=changed)
    assert not torch.equal(output, shifted)
    assert not torch.equal(calls[1][1], changed_calls[1][1])
    perturbed, _, perturbed_calls = _forward(fixture, perturb=lambda codes: codes + .5)
    assert not torch.equal(output, perturbed)
    assert not torch.equal(calls[1][1], perturbed_calls[1][1])
    assert all(branch.frame is None for branch in fixture.host.particle_branches())


def test_conditional_only_routing_retains_original_context_batch():
    fixture = _fixture()
    output, usage, calls = _forward(fixture, cfg=1)
    assert output.shape == fixture.inputs[0].shape
    assert all(logits.shape == (2, 2, 8) for _, logits in calls)
    assert usage.shape == (2, 8)


def test_failed_host_forward_clears_frame_and_restores_published_scales():
    fixture = _fixture()
    fixture.host.model.fail = True
    ordinary = [module for module in fixture.host.model.modules() if isinstance(module, _LoRA)]
    before = [module.multiplier for module in ordinary]
    with pytest.raises(RuntimeError, match="intentional host failure"):
        _forward(fixture)
    assert all(branch.frame is None for branch in fixture.host.particle_branches())
    assert [module.multiplier for module in ordinary] == before
    assert fixture.host._in_forward is False
    fixture.host.model.fail = False
    _forward(fixture)

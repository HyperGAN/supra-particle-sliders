"""Selective moving teachers preserve native optimizer and recovery state."""
from copy import deepcopy

import pytest
import torch
from torch import nn

from particlegan import RoutedBatch
from supra.particle_adapter import SITES
from supra.particle_pilot import (FrozenTextContexts, checkpoint, make_loop,
                                  restore, state_digest, switch_teacher)


class _LoRA(nn.Module):
    def __init__(self, index):
        super().__init__()
        self.base = nn.Linear(4, 4)
        self.down = nn.Linear(4, 2, bias=False)
        self.up = nn.Linear(2, 4, bias=False)
        self.weight = nn.Parameter(torch.tensor((index + 1) / 8))
        self.multiplier = .3 + index / 10


class _Teacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.ctx_proj = _LoRA(14)
        self.blocks = nn.ModuleList([nn.Module() for _ in range(14)])
        for index, block in enumerate(self.blocks):
            block.cross_attn = nn.Module()
            block.cross_attn.proj = _LoRA(index)
        self.fail = False
        self.observed = None

    def forward(self, z, t, ctx, mask):
        branches = [(name, module) for name, module in self.named_modules()
                    if hasattr(module, "multiplier")]
        self.observed = {name: module.multiplier for name, module in branches}
        if self.fail:
            raise RuntimeError("intentional teacher failure")
        delta = sum(module.weight * module.multiplier for _, module in branches)
        return z + delta + t[:, None, None, None]


def _data():
    text = torch.arange(3 * 2 * 768, dtype=torch.float32).reshape(3, 2, 768) / 1000
    masks = torch.ones(3, 2)
    context = torch.cat((torch.linspace(-.1, .1, 8192).reshape(2, 4096),
                         torch.tensor([[.25, 1., 1.], [.5, 2., 1.]])), dim=1)
    split = dict(context=context, targets=torch.ones(2, 4, 32, 32),
                 partial=torch.zeros(2, 4, 32, 32), base=torch.zeros(2, 4, 32, 32))
    data = dict(text_contexts=text, text_masks=masks,
                fit=deepcopy(split), guard=deepcopy(split), test=deepcopy(split))
    data["guard"]["context"][:, 0] += 1
    return data


def test_teacher_switch_scales_only_selected_published_branches_and_restores():
    data = _data()
    owner = FrozenTextContexts(data["text_contexts"], data["text_masks"], _Teacher())
    context = data["fit"]["context"]
    weights = state_digest(dict(owner.teacher.named_parameters()))
    original = {name: module.multiplier for name, module in owner.teacher.named_modules()
                if hasattr(module, "multiplier")}
    initial = owner.teacher_velocity(context)
    owner.teacher_site_scale.fill_(-1)
    switched = owner.teacher_velocity(context)
    expected_delta = -2 * sum(owner.teacher.get_submodule(site).weight for site in SITES)
    torch.testing.assert_close(switched - initial, expected_delta.expand_as(initial), rtol=0, atol=1e-5)
    assert owner.teacher.observed == {name: (-1. if name in SITES else 1.) for name in original}
    assert original == {name: module.multiplier for name, module in owner.teacher.named_modules()
                        if hasattr(module, "multiplier")}
    assert state_digest(dict(owner.teacher.named_parameters())) == weights
    assert all(not parameter.requires_grad for parameter in owner.teacher.parameters())
    zero = context.clone()
    zero[:, 4098] = 0
    switched_zero = owner.teacher_velocity(zero)
    owner.teacher_site_scale.fill_(1)
    assert torch.equal(owner.teacher_velocity(zero), switched_zero)
    owner.teacher.fail = True
    with pytest.raises(RuntimeError, match="intentional teacher failure"):
        owner.teacher_velocity(context)
    assert original == {name: module.multiplier for name, module in owner.teacher.named_modules()
                        if hasattr(module, "multiplier")}


def test_switch_checkpoint_owns_both_teacher_scales_without_resetting_native_state():
    loop = make_loop(_Teacher(), _data(), mode="full", device="cpu")
    loop.policy.routed_control.begin(RoutedBatch(loop.fit_context, loop.fit_targets,
                                                loop.guard_context, loop.guard_targets))
    # Include nonempty Adam memory: switching the task must preserve it.
    parameter = loop.policy.opt_g.param_groups[0]["params"][0]
    loop.policy.opt_g.state[parameter] = {
        "step": torch.tensor(7.), "exp_avg": torch.full_like(parameter, .1),
        "exp_avg_sq": torch.full_like(parameter, .2),
        "max_exp_avg_sq": torch.full_like(parameter, .3)}
    before = checkpoint(loop)
    expected = deepcopy(before)
    for family in ("models", "averages"):
        expected["policy"][family]["encoder"]["teacher_site_scale"].fill_(-1)
    switch_teacher(loop, -1)
    assert loop.policy.encoder.teacher_site_scale.item() == -1
    assert loop.policy.ema_encoder.teacher_site_scale.item() == -1
    assert loop.policy.encoder.teacher_site_scale.data_ptr() != loop.policy.ema_encoder.teacher_site_scale.data_ptr()
    assert state_digest(checkpoint(loop)) == state_digest(expected)
    assert loop.policy.served_model().encoder.teacher_site_scale.item() == -1
    restore(loop, before)
    assert loop.policy.encoder.teacher_site_scale.item() == 1
    assert loop.policy.ema_encoder.teacher_site_scale.item() == 1
    assert state_digest(checkpoint(loop)) == state_digest(before)
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="must be finite"):
            switch_teacher(loop, bad)


def test_switch_releases_temporary_served_swap_without_losing_fast_weights():
    loop = make_loop(_Teacher(), _data(), mode="movable", device="cpu")
    loop.policy._table_tester().last_decisive = -1
    fast = loop.policy.G.particle_branches()[0].up.weight
    average = loop.policy.ema_G.particle_branches()[0].up.weight
    with torch.no_grad():
        fast.fill_(1)
        average.fill_(2)
    loop.policy._serve_apply()
    assert loop.policy._fast is not None
    assert bool((fast == 2).all())
    switch_teacher(loop, -1)
    assert loop.policy._fast is None
    assert bool((fast == 1).all())
    assert bool((average == 2).all())
    assert loop.policy.served_model().encoder.teacher_site_scale.item() == -1


def test_legacy_static_checkpoint_restores_full_published_teacher_without_mutating_input():
    loop = make_loop(_Teacher(), _data(), mode="full", device="cpu")
    expected = checkpoint(loop)
    legacy = deepcopy(expected)
    for family in ("models", "averages"):
        del legacy["policy"][family]["encoder"]["teacher_site_scale"]
    legacy_digest = state_digest(legacy)
    switch_teacher(loop, -1)
    with torch.no_grad():
        loop.policy.G.particle_branches()[0].up.weight.add_(.25)
    torch.rand(3, generator=loop.data_rng)
    restore(loop, legacy)
    assert loop.policy.encoder.teacher_site_scale.item() == 1
    assert loop.policy.ema_encoder.teacher_site_scale.item() == 1
    assert state_digest(checkpoint(loop)) == state_digest(expected)
    assert state_digest(legacy) == legacy_digest
    assert all("teacher_site_scale" not in legacy["policy"][family]["encoder"]
               for family in ("models", "averages"))


@pytest.mark.parametrize("moving,missing", [(True, ("models", "averages")),
                                           (False, ("models",)), (False, ("averages",))])
def test_restore_rejects_missing_moving_or_one_sided_teacher_scale(moving, missing):
    loop = make_loop(_Teacher(), _data(), mode="movable", device="cpu")
    if moving:
        loop.config["teacher_schedule"] = {"switch_step": 900, "site_scale": -1}
    before = checkpoint(loop)
    invalid = deepcopy(before)
    for family in missing:
        del invalid["policy"][family]["encoder"]["teacher_site_scale"]
    with pytest.raises(ValueError, match="incompatible policy (models|averages)/encoder"):
        restore(loop, invalid)
    assert state_digest(checkpoint(loop)) == state_digest(before)


@pytest.mark.parametrize("mode", ["fixed", "movable", "full"])
def test_recipe_r1_override_preserves_mode_controls(mode):
    loop = make_loop(_Teacher(), _data(), mode=mode, device="cpu",
                     recipe_overrides={"reopen_signal": "none", "reopen_anchor": "hold"})
    assert loop.policy.recipe.reopen_signal == "none"
    assert loop.policy.recipe.reopen_anchor == "hold"
    assert loop.policy.recipe.row_evidence_gate == (mode == "full")
    assert loop.policy.recipe.particle_birth_death == (mode == "full")
    assert loop.policy.table.requires_grad == (mode != "fixed")
    assert loop.config["recipe"] == loop.policy.recipe.to_dict()


def test_recipe_override_accepts_new_default_and_rejects_non_r1_changes():
    loop = make_loop(_Teacher(), _data(), mode="movable", device="cpu")
    assert loop.policy.recipe.reopen_signal == "optimizer"
    assert loop.policy.recipe.reopen_anchor == "release"
    assert loop.policy.surprise is not None
    for field in ("batch_size", "particle_birth_death", "critic_r1_real", "critic_payoff_damping"):
        with pytest.raises(ValueError, match="limited to R1 controls"):
            make_loop(None, {}, mode="movable", recipe_overrides={field: False})

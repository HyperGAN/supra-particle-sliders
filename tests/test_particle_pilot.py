"""Frozen matched teachers, residual serving and recovery dataset identity."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from particlegan import RoutedCandidate, RoutedRows
from supra.particle_adapter import SITES
from supra.particle_pilot import FrozenTextContexts, checkpoint, model_forward, restore, state_digest


class _TeacherBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(.12))
        self.multiplier = .7


class _Teacher(nn.Module):
    """A batch-sensitive host represents native BF16 batch-shape dependence."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(.8))
        self.branch = _TeacherBranch()
        self.fail = False

    def forward(self, z, t, ctx, mask):
        if self.fail:
            raise RuntimeError("intentional teacher failure")
        text = (ctx * mask.unsqueeze(-1)).sum((1, 2))
        correction = (text * self.weight + t + self.branch.multiplier * self.branch.weight
                      + .03 * len(z))
        return z + correction[:, None, None, None]


def _guided(host, z, t, ctx, mask, uctx, umask, strength):
    original = host.branch.multiplier
    try:
        host.branch.multiplier = strength
        batch = len(z)
        output = host(torch.cat((z, z)), torch.cat((t, t)),
                      torch.cat((ctx, uctx.expand(batch, -1, -1))),
                      torch.cat((mask, umask.expand(batch, -1))))
        conditional, unconditional = output.chunk(2)
        return unconditional + 3 * (conditional - unconditional)
    finally:
        host.branch.multiplier = original


class _Student(nn.Module):
    def __init__(self, teacher):
        super().__init__()
        self.host = deepcopy(teacher).requires_grad_(False)
        self.offset = nn.Parameter(torch.tensor(.15))

    def forward_routed(self, z, t, ctx, mask, uctx, umask, candidate, routing, router, strength):
        base = _guided(self.host, z, t, ctx, mask, uctx, umask, strength)
        logits = z.flatten(1).mean(1)[:, None] * candidate.table[:, 0][None]
        first = routing.mix(SITES[0], logits)
        second = routing.mix(SITES[1], logits + first[:, :1] * candidate.table[:, 1][None])
        if strength == 0:
            return base
        delta = self.offset * second.mean(-1)
        return base + strength * delta[:, None, None, None]


def _inputs():
    contexts = torch.arange(24, dtype=torch.float32).reshape(3, 2, 4) / 20
    masks = torch.tensor([[1., 1.], [1., 0.], [1., 1.]])
    packed = torch.cat((torch.linspace(-.2, .3, 8192).reshape(2, 4096),
                        torch.tensor([[.2, 1., 1.], [.7, 2., 1.]])), dim=1)
    return contexts, masks, packed


def test_encoder_owns_frozen_teacher_and_detached_text_masks():
    contexts, masks, packed = _inputs()
    contexts.requires_grad_(True)
    teacher = _Teacher()
    owner = FrozenTextContexts(contexts, masks, teacher)
    before = deepcopy(owner.state_dict())
    expected_condition = owner.condition(packed).clone()
    expected_velocity = owner.teacher_velocity(packed).clone()
    assert owner.contexts.data_ptr() != contexts.data_ptr()
    assert owner.masks.data_ptr() != masks.data_ptr()
    assert not owner.contexts.requires_grad
    assert all(not parameter.requires_grad for parameter in owner.teacher.parameters())
    with torch.no_grad():
        contexts.add_(100)
        masks.zero_()
        teacher.weight.add_(50)
    assert torch.equal(owner.condition(packed), expected_condition)
    assert torch.equal(owner.teacher_velocity(packed), expected_velocity)
    assert all(torch.equal(before[name], value) for name, value in owner.state_dict().items())
    assert owner.teacher.branch.multiplier == .7
    assert not expected_velocity.requires_grad


def test_live_teacher_pairs_current_cfg_batch_shape_and_restores_strength():
    contexts, masks, packed = _inputs()
    owner = FrozenTextContexts(contexts, masks, _Teacher())
    z, t, ctx, mask, uctx, umask, strength = owner.unpack(packed)
    expected = _guided(owner.teacher, z, t, ctx, mask, uctx, umask, strength)
    actual = owner.teacher_velocity(packed)
    assert torch.equal(actual, expected)
    cached_single = torch.cat([owner.teacher_velocity(row.unsqueeze(0)) for row in packed])
    assert not torch.equal(actual, cached_single)
    owner.teacher.fail = True
    with pytest.raises(RuntimeError, match="intentional teacher failure"):
        owner.teacher_velocity(packed)
    assert owner.teacher.branch.multiplier == .7


def test_residual_callback_uses_snapshot_owners_and_restores_native_velocity():
    contexts, masks, packed = _inputs()
    teacher = _Teacher()
    models = dict(generator=_Student(teacher), encoder=FrozenTextContexts(contexts, masks, teacher),
                  router=nn.Identity())
    snapshot = deepcopy(models)
    table = torch.linspace(-.5, .8, 32).reshape(8, 4).requires_grad_(True)
    mass = torch.zeros(8)
    candidate = RoutedCandidate(table, mass, {"log_mass": mass})
    rows = RoutedRows(model_forward=model_forward, features=lambda *args: None, sites=SITES)
    residual = rows.forward(snapshot, packed, candidate)
    actual = residual + snapshot["encoder"].teacher_velocity(packed)
    z, t, ctx, mask, uctx, umask, strength = snapshot["encoder"].unpack(packed)
    expected_base = _guided(snapshot["generator"].host, z, t, ctx, mask, uctx, umask, strength)
    assert not torch.equal(actual, expected_base)
    torch.testing.assert_close(actual - expected_base, residual, rtol=1e-5, atol=1e-6)
    with torch.no_grad():
        models["encoder"].contexts.add_(100)
        models["encoder"].teacher.weight.add_(100)
        models["generator"].offset.add_(100)
    assert torch.equal(rows.forward(snapshot, packed, candidate), residual)
    residual.sum().backward()
    assert snapshot["generator"].offset.grad.abs() > 0
    assert table.grad.norm() > 0
    assert all(parameter.grad is None for parameter in snapshot["encoder"].teacher.parameters())
    zero = packed.clone()
    zero[:, 4098] = 0
    assert torch.equal(rows.forward(snapshot, zero, candidate), torch.zeros_like(actual))
    z, t, ctx, mask, uctx, umask, strength = snapshot["encoder"].unpack(zero)
    expected_zero = _guided(snapshot["generator"].host, z, t, ctx, mask, uctx, umask, strength)
    assert torch.equal(rows.forward(snapshot, zero, candidate) + snapshot["encoder"].teacher_velocity(zero),
                       expected_zero)


def test_scalar_digest_is_portable_and_recovery_rejects_changed_source_contexts():
    _, _, packed = _inputs()
    scalar = torch.tensor(.125)
    assert state_digest(dict(noise=scalar, context=packed)) == state_digest(dict(context=packed.clone(), noise=scalar.clone()))
    assert state_digest(scalar) != state_digest(scalar.reshape(1))
    assert state_digest(scalar) != state_digest(scalar + .01)

    class Policy:
        def state_dict(self):
            return {"noise": scalar.clone()}

        def load_state_dict(self, state):
            raise AssertionError("changed dataset must be rejected before loading model state")

    loop = SimpleNamespace(policy=Policy(), config={"dataset_digest": state_digest(packed)},
                           data_rng=torch.Generator().manual_seed(42),
                           paired_noise_rng=torch.Generator().manual_seed(43))
    saved = checkpoint(loop)
    packed[0, 0] += .1
    loop.config["dataset_digest"] = state_digest(packed)
    assert saved["config"] != loop.config
    with pytest.raises(ValueError, match="configuration mismatch"):
        restore(loop, saved)

"""Original pool identity, training-only guards and caption-owned live pairs."""
from copy import deepcopy

import pytest
import torch
from torch import nn

from supra.particle_training_data import (FrozenSliderContexts, SOURCE_COLUMN,
                                          STRENGTH_COLUMN, TARGET_COLUMN,
                                          build_slider_data)


class _Branch(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(.4))
        self.multiplier = .7


class _Teacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(.8))
        self.branch = _Branch()
        self.fail = False

    def forward(self, z, t, ctx, mask):
        if self.fail:
            raise RuntimeError("intentional base failure")
        text = (ctx * mask.unsqueeze(-1)).sum((1, 2))
        correction = self.weight * text + t + self.branch.weight * self.branch.multiplier + .03 * len(z)
        return z + correction[:, None, None, None]


def _record(prompt, z, t, *, edit=True, trajectory="neutral"):
    row = dict(prompt=prompt, z=torch.full((1, 4, 32, 32), float(z)), t=t,
               neutral=torch.full((1, 4, 32, 32), z+.1),
               target=torch.full((1, 4, 32, 32), z+(.3 if edit else .1)))
    if edit:
        row.update(row="knight", seed=7, context=trajectory, hold=False)
    return row


def _inputs():
    prompts = ["", "neutral", "positive", "lake", "bridge", "bridge positive", "fruit"]
    texts = torch.arange(len(prompts)*8, dtype=torch.float32).reshape(len(prompts), 2, 4)/20
    masks = torch.ones(len(prompts), 2)
    config = dict(rows=[dict(name="knight", neutral="neutral", positive="positive")],
                  preservation=["lake"], verification=[dict(prompt="bridge", teacher="bridge positive"),
                                                       dict(prompt="fruit")])
    train = dict(records=[_record("neutral", 1, 0), _record("neutral", 2, .2),
                          _record("neutral", 3, 0, trajectory="positive"),
                          _record("neutral", 4, .2, trajectory="positive")],
                 holds=[_record("lake", 5, 0, edit=False), _record("lake", 6, .2, edit=False)])
    validation = dict(records=[_record("neutral", 10, 0), _record("neutral", 11, .2)],
                      holds=[_record("lake", 12, 0, edit=False), _record("lake", 13, .2, edit=False)],
                      endpoints=[dict(prompt="neutral", seed=39, name="knight",
                                      neutral=torch.zeros(1, 4, 32, 32), target=torch.ones(1, 4, 32, 32))])
    return train, validation, config, texts, masks, prompts


def test_original_pool_order_captions_and_identity_preservation_are_retained():
    train, validation, config, texts, masks, prompts = _inputs()
    data = build_slider_data(train, validation, config, texts, masks, prompts)
    assert data["fit"]["context"].shape == (4, 4100)
    assert torch.equal(data["fit"]["source_indices"], torch.arange(4))
    assert torch.equal(data["fit"]["context"][:, :4096], torch.cat([row["z"] for row in train["records"]]).flatten(1))
    assert torch.equal(data["fit"]["targets"], torch.cat([row["target"] for row in train["records"]]))
    assert torch.equal(data["fit"]["base"], torch.cat([row["neutral"] for row in train["records"]]))
    assert bool(data["fit"]["context"][:, SOURCE_COLUMN].eq(1).all())
    assert bool(data["fit"]["context"][:, TARGET_COLUMN].eq(2).all())
    assert bool(data["holds"]["context"][:, SOURCE_COLUMN].eq(data["holds"]["context"][:, TARGET_COLUMN]).all())
    assert bool(data["holds"]["hold"].all())
    assert torch.equal(data["coordinate_scale"], torch.full((16,), .04))
    assert data["test"]["context"].shape == (2, 4100)
    assert data["preservation"]["context"].shape == (2, 4100)
    validation["endpoints"][0]["target"].zero_()
    assert bool(data["endpoints"][0]["target"].eq(1).all())


def test_guards_are_live_training_midpoints_and_never_read_validation_velocities():
    train, validation, config, texts, masks, prompts = _inputs()
    data = build_slider_data(train, validation, config, texts, masks, prompts)
    guard = data["guard"]
    assert guard["context"].shape == (3, 4100)
    assert bool(guard["targets"].eq(0).all())
    assert bool(guard["context"][:, STRENGTH_COLUMN].eq(1).all())
    expected = torch.stack([(data["fit"]["context"][0]+data["fit"]["context"][1])*.5,
                            (data["fit"]["context"][2]+data["fit"]["context"][3])*.5,
                            data["holds"]["context"].mean(0)])
    assert torch.equal(guard["context"], expected)
    for row in validation["records"]+validation["holds"]:
        row["target"].add_(100)
        row["neutral"].add_(100)
    changed = build_slider_data(train, validation, config, texts, masks, prompts)
    assert torch.equal(changed["guard"]["context"], guard["context"])
    assert torch.equal(changed["coordinate_scale"], data["coordinate_scale"])
    assert not torch.equal(changed["test"]["targets"], data["test"]["targets"])
    data_without_validation = build_slider_data(train, None, config, texts, masks, prompts)
    assert data_without_validation["test"] is None
    assert data_without_validation["preservation"] is None
    assert torch.equal(data_without_validation["guard"]["context"], guard["context"])


def test_frozen_target_caption_teacher_is_live_batch_shaped_and_restores_lora():
    inputs = _inputs()
    data = build_slider_data(*inputs)
    texts, masks = inputs[3], inputs[4]
    owner = FrozenSliderContexts(texts, masks, _Teacher())
    context = data["fit"]["context"][:2]
    z, t, source_text, _, _, _, _ = owner.unpack(context)
    assert torch.equal(source_text, texts[[1, 1]])
    target = owner.teacher_velocity(context)
    target_text = texts[[2, 2]]
    # Pure base disables the nonzero ordinary branch. CFG ordering is
    # conditional sources first, unconditional contexts second.
    text_term = 3*(target_text.sum((1, 2))*.8)-2*(texts[[0, 0]].sum((1, 2))*.8)
    expected = z+(text_term+t+.03*4)[:, None, None, None]
    torch.testing.assert_close(target, expected)
    singles = torch.cat([owner.teacher_velocity(row.unsqueeze(0)) for row in context])
    assert not torch.equal(singles, target)
    assert owner.teacher.branch.multiplier == .7
    assert not target.requires_grad
    assert all(not parameter.requires_grad for parameter in owner.teacher.parameters())
    zero_strength = context.clone()
    zero_strength[:, STRENGTH_COLUMN] = 0
    assert torch.equal(owner.teacher_velocity(zero_strength), target)
    owner.teacher.fail = True
    with pytest.raises(RuntimeError, match="intentional base failure"):
        owner.teacher_velocity(context)
    assert owner.teacher.branch.multiplier == .7


def test_checkpoint_copy_owns_both_caption_buffers_and_frozen_base():
    inputs = _inputs()
    data = build_slider_data(*inputs)
    teacher = _Teacher()
    owner = FrozenSliderContexts(inputs[3], inputs[4], teacher)
    snapshot = deepcopy(owner)
    packed = data["fit"]["context"][:2]
    before = snapshot.teacher_velocity(packed)
    before_condition = snapshot.condition(packed)
    with torch.no_grad():
        inputs[3].add_(100)
        inputs[4].zero_()
        teacher.weight.add_(100)
        owner.contexts.add_(10)
        owner.teacher.weight.add_(10)
    assert torch.equal(snapshot.teacher_velocity(packed), before)
    assert torch.equal(snapshot.condition(packed), before_condition)
    restored = FrozenSliderContexts(data["text_contexts"], data["text_masks"], _Teacher())
    restored.load_state_dict(snapshot.state_dict())
    assert torch.equal(restored.teacher_velocity(packed), before)


def test_invalid_caption_ids_and_validation_leak_are_rejected():
    inputs = _inputs()
    data = build_slider_data(*inputs)
    owner = FrozenSliderContexts(inputs[3], inputs[4], _Teacher())
    for column, value in ((SOURCE_COLUMN, .5), (TARGET_COLUMN, -1), (TARGET_COLUMN, 100)):
        context = data["fit"]["context"][:1].clone()
        context[:, column] = value
        with pytest.raises(ValueError, match="invalid source or target"):
            owner.teacher_velocity(context)
    train, validation, config, texts, masks, prompts = _inputs()
    validation["records"][0] = deepcopy(train["records"][0])
    with pytest.raises(ValueError, match="validation contexts overlap"):
        build_slider_data(train, validation, config, texts, masks, prompts)

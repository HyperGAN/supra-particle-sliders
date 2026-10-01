"""Original Supra caption-transfer records and live frozen-base pairing.

Packed contexts are [latent 4096, time, source caption ID, target caption ID,
slider strength]. Cached velocities are retained for reporting and fixed
training-coordinate scales. The adversarial pair always uses a live teacher
at the current complete batch shape; no reconstruction objective is supplied.
"""
from copy import deepcopy
import math

import torch
from torch import nn

from .particle_game import patchify


CONTEXT_WIDTH = 4100
TIME_COLUMN, SOURCE_COLUMN, TARGET_COLUMN, STRENGTH_COLUMN = range(4096, 4100)


class FrozenSliderContexts(nn.Module):
    """Checkpoint-owned source/target text and a frozen pure-base teacher."""

    def __init__(self, contexts, masks, pure_base_model, cfg=3.):
        super().__init__()
        contexts, masks = torch.as_tensor(contexts), torch.as_tensor(masks)
        if (contexts.ndim != 3 or not len(contexts) or masks.shape != contexts.shape[:2]
                or not contexts.is_floating_point() or not masks.is_floating_point()
                or not bool(torch.isfinite(contexts).all()) or not bool(torch.isfinite(masks).all())):
            raise ValueError("text contexts and masks must be finite floating [captions,tokens,width] and [captions,tokens]")
        if not isinstance(pure_base_model, nn.Module) or not math.isfinite(cfg) or cfg <= 0:
            raise ValueError("a frozen base module and finite positive CFG are required")
        self.register_buffer("contexts", contexts.detach().clone())
        self.register_buffer("masks", masks.detach().clone())
        self.teacher = deepcopy(pure_base_model).eval().requires_grad_(False)
        self.cfg = float(cfg)

    def _ids(self, context, column):
        values = context[:, column]
        ids = values.long()
        if not torch.equal(ids.to(values.dtype), values) or bool((ids < 0).any()) or bool((ids >= len(self.contexts)).any()):
            raise ValueError("invalid source or target caption IDs")
        return ids

    def _validate(self, context):
        if (not isinstance(context, torch.Tensor) or context.ndim != 2 or not len(context)
                or context.shape[1] != CONTEXT_WIDTH or not context.is_floating_point()
                or context.device != self.contexts.device or not bool(torch.isfinite(context).all())):
            raise ValueError("packed slider contexts must be finite floating [B,4100] on the text-buffer device")
        self._ids(context, SOURCE_COLUMN)
        self._ids(context, TARGET_COLUMN)
        if not bool((context[:, STRENGTH_COLUMN] == context[0, STRENGTH_COLUMN]).all()):
            raise ValueError("a routed batch must have one slider strength")

    def unpack(self, context, *, target=False):
        """Return the existing seven host inputs, using source captions by default."""
        self._validate(context)
        ids = self._ids(context, TARGET_COLUMN if target else SOURCE_COLUMN)
        return (context[:, :4096].reshape(-1, 4, 32, 32), context[:, TIME_COLUMN],
                self.contexts[ids], self.masks[ids], self.contexts[:1], self.masks[:1],
                float(context[0, STRENGTH_COLUMN]))

    def condition(self, context):
        """Fixed source-text and flow-time critic conditioning, outside errors."""
        self._validate(context)
        ids = self._ids(context, SOURCE_COLUMN)
        text, mask = self.contexts[ids], self.masks[ids]
        pooled = (text * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        return torch.cat((pooled, context[:, TIME_COLUMN:TIME_COLUMN+1]), dim=-1)

    @torch.no_grad()
    def teacher_velocity(self, context):
        """Pure-base target-caption CFG on the exact current paired batch.

        Targets do not depend on the slider strength. Serving should call the
        raw student velocity when checking exact strength-zero behavior.
        """
        self.teacher.eval()
        z, t, ctx, mask, uctx, umask, _ = self.unpack(context, target=True)
        if self.cfg > 1:
            batch = len(z)
            z, t = torch.cat((z, z)), torch.cat((t, t))
            ctx = torch.cat((ctx, uctx.expand(batch, -1, -1)))
            mask = torch.cat((mask, umask.expand(batch, -1)))
        ordinary = [module for module in self.teacher.modules() if hasattr(module, "multiplier")]
        previous = [module.multiplier for module in ordinary]
        try:
            for module in ordinary:
                module.multiplier = 0.
            with torch.autocast(device_type=z.device.type, dtype=torch.bfloat16, enabled=z.device.type == "cuda"):
                velocity = self.teacher(z, t, ctx, mask).float()
            if self.cfg > 1:
                conditional, unconditional = velocity.chunk(2)
                velocity = unconditional + self.cfg * (conditional - unconditional)
            return velocity
        finally:
            for module, value in zip(ordinary, previous):
                module.multiplier = value


def _pack_records(records, positive_by_prompt, prompt_ids, *, hold=False):
    if not records:
        raise ValueError("each original training pool must contain records")
    contexts, targets, bases, source_prompts, target_prompts = [], [], [], [], []
    for row in records:
        source = row["prompt"]
        target = source if hold else positive_by_prompt[source]
        if source not in prompt_ids or target not in prompt_ids:
            raise ValueError("all source and target captions require frozen text buffers")
        z, base, teacher = (torch.as_tensor(row[name]).detach().cpu().float()
                            for name in ("z", "neutral", "target"))
        if (any(value.shape != (1, 4, 32, 32) or not bool(torch.isfinite(value).all())
                for value in (z, base, teacher)) or not math.isfinite(float(row["t"]))):
            raise ValueError("original cache records must contain finite singleton native latents and velocities")
        tail = torch.tensor([[float(row["t"]), prompt_ids[source], prompt_ids[target], 1.]])
        contexts.append(torch.cat((z.flatten(1), tail), dim=1))
        targets.append(teacher)
        bases.append(base)
        source_prompts.append(source)
        target_prompts.append(target)
    return dict(context=torch.cat(contexts), targets=torch.cat(targets), base=torch.cat(bases),
                source_indices=torch.arange(len(records)), prompts=source_prompts, target_prompts=target_prompts,
                hold=torch.full((len(records),), hold, dtype=torch.bool))


def _training_guards(train_cache, fit, holds, count):
    """Interpolate training trajectories; validation rows never enter guards."""
    groups = {}
    for pool_name, rows, packed in (("fit", train_cache["records"], fit["context"]),
                                   ("holds", train_cache["holds"], holds["context"])):
        for index, row in enumerate(rows):
            key = (pool_name, row["prompt"], row.get("row"), row.get("seed"), row.get("context"))
            groups.setdefault(key, []).append(packed[index])
    candidates = []
    for trajectory in groups.values():
        trajectory = sorted(trajectory, key=lambda row: float(row[TIME_COLUMN]))
        for left, right in zip(trajectory, trajectory[1:]):
            if right[TIME_COLUMN] > left[TIME_COLUMN]:
                candidates.append((left + right) * .5)
    if not candidates:
        raise ValueError("training-only guards require adjacent trajectory times")
    candidates = torch.stack(candidates)
    # Keep the first occurrence in trajectory order before even selection.
    _, inverse = torch.unique(candidates, dim=0, return_inverse=True)
    seen, indices = set(), []
    for index, group in enumerate(inverse.tolist()):
        if group not in seen:
            seen.add(group)
            indices.append(index)
    candidates = candidates[indices]
    indices = torch.linspace(0, len(candidates)-1, min(count, len(candidates))).round().long()
    contexts = candidates[indices]
    original = torch.unique(torch.cat((fit["context"], holds["context"])), dim=0)
    if len(torch.unique(torch.cat((original, contexts)), dim=0)) != len(original) + len(contexts):
        raise ValueError("training midpoints overlap original fit contexts")
    return dict(context=contexts, targets=torch.zeros(len(contexts), 4, 32, 32),
                target_semantics="zero paired-residual labels; teacher velocity computed live",
                hold=contexts[:, SOURCE_COLUMN].eq(contexts[:, TARGET_COLUMN]))


def build_slider_data(train_cache, validation_cache, prompt_config, contexts, masks, prompts, *, guard_count=64):
    """Preserve original cache pool order; reserve all validation for reporting.

    ``prompts[0]`` must be the unconditional empty caption. ``contexts`` and
    ``masks`` contain its frozen T5 values followed by the named captions.
    ``validation_cache=None`` supports training before evaluation preparation.
    """
    prompts = list(prompts)
    contexts, masks = torch.as_tensor(contexts), torch.as_tensor(masks)
    if (not prompts or prompts[0] != "" or len(set(prompts)) != len(prompts)
            or contexts.ndim != 3 or len(contexts) != len(prompts) or masks.shape != contexts.shape[:2]):
        raise ValueError("unique named text buffers must begin with the unconditional empty caption")
    if type(guard_count) is not int or guard_count < 1:
        raise ValueError("guard_count must be a positive integer")
    prompt_ids = {prompt: index for index, prompt in enumerate(prompts)}
    positive_by_prompt = {row["neutral"]: row["positive"] for row in prompt_config["rows"]}
    for row in prompt_config.get("verification", ()):
        if "teacher" in row:
            positive_by_prompt[row["prompt"]] = row["teacher"]
    fit = _pack_records(train_cache["records"], positive_by_prompt, prompt_ids)
    holds = _pack_records(train_cache["holds"], positive_by_prompt, prompt_ids, hold=True)
    guards = _training_guards(train_cache, fit, holds, guard_count)
    test = preservation = None
    endpoints = []
    if validation_cache is not None:
        test = _pack_records(validation_cache["records"], positive_by_prompt, prompt_ids)
        preservation = _pack_records(validation_cache["holds"], positive_by_prompt, prompt_ids, hold=True)
        endpoints = deepcopy(validation_cache.get("endpoints", []))
        original = torch.unique(torch.cat((fit["context"], holds["context"], guards["context"])), dim=0)
        evaluation = torch.unique(torch.cat((test["context"], preservation["context"])), dim=0)
        if len(torch.unique(torch.cat((original, evaluation)), dim=0)) != len(original) + len(evaluation):
            raise ValueError("validation contexts overlap training or structural guard contexts")
    scale = patchify(fit["targets"]-fit["base"]).std(dim=(0, 1)).clamp_min(.04)
    return dict(fit=fit, holds=holds, guard=guards, test=test, preservation=preservation,
                endpoints=endpoints, prompts=prompts, text_contexts=contexts.detach().cpu().clone(),
                text_masks=masks.detach().cpu().clone(), coordinate_scale=scale,
                metadata=dict(context_layout=["latent_4096", "time", "source_caption_id", "target_caption_id", "strength"],
                              fit_records=len(fit["context"]), hold_records=len(holds["context"]),
                              guard_records=len(guards["context"]), validation_records=0 if test is None else len(test["context"]),
                              preservation_records=0 if preservation is None else len(preservation["context"]),
                              original_pool_order_preserved=True,
                              guard_source="midpoints of original training trajectories only; live frozen teacher",
                              coordinate_scale="training edit target-minus-neutral patch std, minimum .04",
                              cached_velocities="reporting and fixed coordinate units only; live batch-shaped target for training"))

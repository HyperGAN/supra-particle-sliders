"""Shared full-Supra asset loading and direct native editing updates.

Qualification and continuation use the same public construction/restoration
and update path. These helpers contain no evaluation-based training decisions.
"""
import gc
import hashlib
import math
from pathlib import Path

import torch

from .runtime import TARGETS, MODEL_SOURCE, model_module
from .particle_game import ConditionalTokenCritic, update
from .particle_pilot import state_digest
from .particle_final_precision import (make_precision_training_loop,
                                      bootstrap_precision, precision_rng_scope)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_task(card):
    """Authenticate the original task and build its pure live teacher/judges."""
    for item in card["inputs"].values():
        if sha(item["path"]) != item["sha256"]:
            raise ValueError("fixed task input changed: " + item["path"])
    if sha(MODEL_SOURCE) != card["backend_sha256"]:
        raise ValueError("fixed backend source differs")
    device = torch.device("cuda:0")
    data = torch.load(card["inputs"]["data"]["path"], map_location="cpu",
                      weights_only=False, mmap=True)
    if (state_digest(data) != card["data_digest"]
            or data["test"]["context"].shape != (240, 4100)
            or len(data["fit"]["context"]) != 240):
        raise ValueError("full task data identity/geometry differs")
    parent = torch.load(card["inputs"]["neutral_checkpoint"]["path"],
                        map_location="cpu", weights_only=False, mmap=True)
    if (state_digest(parent) != card["neutral_native_digest"]
            or parent["policy"]["completed_steps"] != 12800
            or parent["config"]["training_schedule"] != "fresh_editing_only_v1"
            or parent["config"]["preservation_game_weight"] != 0.):
        raise ValueError("qualified editing-only original12800 state differs")
    teacher = {k.removeprefix("teacher."): v
               for k, v in parent["policy"]["models"]["encoder"].items()
               if k.startswith("teacher.")}
    with torch.random.fork_rng(devices=[0]), torch.device("meta"):
        backend = model_module()
        base = backend.SupraDiT()
        backend.attach_supra_lora(base, rank=16, alpha=16, targets=TARGETS)
    base.load_state_dict(teacher, strict=True, assign=True)
    base.to(device).eval().requires_grad_(False)
    if len(base.blocks) != 14 or any(bool(v.count_nonzero()) for k, v in teacher.items()
                                   if k.endswith(".up.weight")):
        raise ValueError("pure fourteen-block live teacher differs")
    del teacher
    judges = {}
    for name in ("D1856", "D6400"):
        saved = torch.load(card["inputs"][name]["path"], map_location="cpu",
                           weights_only=False, mmap=True)
        with torch.random.fork_rng(devices=[0]):
            judge = ConditionalTokenCritic(data["coordinate_scale"].to(device)).to(device)
        judge.load_state_dict(saved["policy"]["models"]["critic"], strict=True)
        judge.eval().requires_grad_(False)
        if state_digest(judge.state_dict()) != card["critic_tensor_digests"][name]:
            raise ValueError("fixed learned judge differs: " + name)
        judges[name] = judge
        del saved
    gc.collect()
    return data, parent, base, judges


def fresh_restored(base, data, parent, precision, expected_native_digest):
    """Fresh public owners, then original trained restore; no post-restore init."""
    with torch.random.fork_rng(devices=[0]):
        loop = make_precision_training_loop(
            base, data, precision=precision, device="cuda:0",
            probe_interval=parent["config"]["probe_interval"],
            branch_lr=parent["config"]["branch_lr"])
        proof = bootstrap_precision(loop, parent, expected_native_digest=expected_native_digest)
    if (len(loop.policy.G.sites) != 71 or loop.policy.G.rank != 16
            or tuple(loop.policy.table.shape) != (128, 4)):
        raise ValueError("original shared-Up particle geometry differs")
    return loop, proof


def editing_update(loop):
    """One unmodified native game update, with FIT-only CPU7 sampling."""
    with precision_rng_scope(loop), torch.autograd.set_multithreading_enabled(False):
        indices = torch.randint(len(loop.fit_context), (4,), generator=loop.data_rng)
        context = loop.fit_context[indices.to(loop.fit_context.device)]
        row = update(loop, context=context,
                     target=torch.zeros(4, 4, 32, 32, device=loop.policy.device),
                     batch_indices=indices, game_weight=1.)
    row.update(hold=False, game_weight=1.)
    if any(not math.isfinite(row[name]) for name in (
            "loss_g", "loss_d", "loss_d_game", "penalty", "bank_grad_norm", "output_sigma")):
        raise FloatingPointError("nonfinite native game update")
    return row

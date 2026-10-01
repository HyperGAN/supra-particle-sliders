#!/usr/bin/env python3
"""Verify exact continuation of a real Supra checkpoint after R1 recovery."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import struct
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from scripts.verify_e22_supra import checked_update, emit, optimizer_provenance
from supra.particle_pilot import checkpoint, frozen_digest, make_loop, restore, state_digest
from supra.runtime import SupraRuntime


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_leaves(reference, replay, path="checkpoint", counts=None):
    """Compare every nested tensor and scalar, including all random streams."""
    if counts is None:
        counts = dict(tensors=0, scalars=0)
    if isinstance(reference, torch.Tensor):
        if (not isinstance(replay, torch.Tensor) or reference.dtype != replay.dtype
                or reference.shape != replay.shape
                or not torch.equal(reference.detach().contiguous().reshape(-1).view(torch.uint8),
                                   replay.detach().contiguous().reshape(-1).view(torch.uint8))):
            raise AssertionError(f"tensor mismatch at {path}")
        counts["tensors"] += 1
    elif isinstance(reference, dict):
        if not isinstance(replay, dict) or reference.keys() != replay.keys():
            raise AssertionError(f"mapping mismatch at {path}")
        for key in reference:
            exact_leaves(reference[key], replay[key], f"{path}.{key}", counts)
    elif isinstance(reference, (tuple, list)):
        if type(reference) is not type(replay) or len(reference) != len(replay):
            raise AssertionError(f"sequence mismatch at {path}")
        for index, (left, right) in enumerate(zip(reference, replay)):
            exact_leaves(left, right, f"{path}[{index}]", counts)
    else:
        matches = (type(reference) is type(replay)
                   and (struct.pack("!d", reference) == struct.pack("!d", replay)
                        if isinstance(reference, float) else reference == replay))
        if not matches:
            raise AssertionError(f"scalar mismatch at {path}")
        counts["scalars"] += 1
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("outputs/e22-pilot/data.pt"))
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("outputs/e22-moving-f459cb6d/stronger/new/final.pt"))
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/e22-moving-f459cb6d/stronger/resume.json"))
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("resume evidence already exists; choose a new output path")
    torch.set_num_threads(8)
    started = time.perf_counter()
    data = torch.load(args.data, map_location="cpu", weights_only=False)
    teacher_sha = file_digest(args.teacher)
    if teacher_sha != data["metadata"]["adapter_sha256"]:
        raise ValueError("teacher weights differ from the paired dataset")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    runtime = SupraRuntime(device="cuda:0", rank=16, allow_hub=False)
    runtime.load(args.teacher)
    runtime.model.requires_grad_(False)
    runtime.text_encoder.to("cpu")
    loop = make_loop(runtime.model, data, mode="full", probe_interval=100)
    loop.config["teacher_schedule"] = deepcopy(saved["config"]["teacher_schedule"])
    restore(loop, saved)
    start = loop.policy.completed_steps
    scales = dict(fast=float(loop.policy.encoder.teacher_site_scale),
                  averaged=float(loop.policy.ema_encoder.teacher_site_scale))
    fires = loop.policy.surprise.fires
    if start != 1500 or scales != {"fast": 2., "averaged": 2.} or fires != 1:
        raise AssertionError(f"unexpected recovery start: step={start}, scales={scales}, fires={fires}")
    frozen = frozen_digest(loop)
    emit(dict(event="restored", step=start, teacher_scales=scales, optimizer_surprise_fires=fires,
              **optimizer_provenance()))
    reference_trace = [checked_update(loop), checked_update(loop)]
    reference = checkpoint(loop)
    reference_digest = state_digest(reference)
    if frozen != frozen_digest(loop):
        raise AssertionError("a frozen parameter changed in the reference continuation")
    emit(dict(event="reference_complete", step=loop.policy.completed_steps,
              checkpoint_digest=reference_digest))
    restore(loop, saved)
    replay_trace = [checked_update(loop), checked_update(loop)]
    replay = checkpoint(loop)
    checkpoint_leaves = exact_leaves(reference, replay)
    trace_leaves = exact_leaves(reference_trace, replay_trace, path="trace")
    replay_digest = state_digest(replay)
    if reference_digest != replay_digest or frozen != frozen_digest(loop):
        raise AssertionError("checkpoint digest or frozen parameters differed after replay")
    result = dict(exact=True, start_step=start, end_step=loop.policy.completed_steps,
                  optimizer_surprise_fires_before=fires, optimizer_surprise_fires_after=loop.policy.surprise.fires,
                  anchor_release_events=loop.policy.surprise.anchor_events,
                  ladder_reopens=sum(tester.counts.get("reopens", 0) for row in loop.policy.lr_settle.testers
                                     for tester in row if tester is not None),
                  teacher_scales=scales, frozen_parameters_unchanged=True,
                  reference_checkpoint_digest=reference_digest, replay_checkpoint_digest=replay_digest,
                  checkpoint_leaves_compared=checkpoint_leaves, trace_leaves_compared=trace_leaves,
                  traces=reference_trace, checkpoint_path=str(args.checkpoint.resolve()),
                  data_sha256=file_digest(args.data), teacher_sha256=teacher_sha,
                  elapsed_seconds=time.perf_counter()-started, **optimizer_provenance())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=str)+"\n")
    emit(dict(event="complete", exact=True, start_step=start, end_step=result["end_step"],
              checkpoint_leaves_compared=checkpoint_leaves, path=str(args.output)))


if __name__ == "__main__":
    main()

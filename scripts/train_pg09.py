#!/usr/bin/env python3
"""Train the shared PG0.9 GAN on owner-admitted native Supra assets."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-root", type=Path, help="Optional reviewed source checkout; default is the pinned installed package")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/supra/prompts-final-boss.yaml")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--cache-sha256", required=True)
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--contexts-sha256", required=True)
    parser.add_argument("--frozen-base-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--steps", type=int, default=28800, help="Fixed maximum recipe budget, including resumed steps")
    parser.add_argument("--stop-at", type=int, help="Checkpoint and pause at this absolute step without changing the recipe")
    parser.add_argument("--save-every", type=int, default=80)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if not 1 <= args.steps <= 28800 or args.save_every < 1:
        parser.error("Require a bounded positive budget and checkpoint interval")
    stop = args.steps if args.stop_at is None else args.stop_at
    if not 1 <= stop <= args.steps:
        parser.error("--stop-at must lie inside the unchanged maximum recipe budget")

    if args.shared_root:
        os.environ["PARTICLE_SLIDERS_ROOT"] = str(args.shared_root.resolve())
    from supra.shared import shared_module, file_sha
    bridge = shared_module("particle_sliders.pg09_remaining_campaigns")
    import yaml
    config = yaml.safe_load(args.config.read_text())
    from supra.runtime import SupraRuntime, MODEL_REV, T5_REV
    # Reject budgets, sources, config and disk asset checksums before native load.
    bridge.validate_live_supra_context_preparation(config,
        dict(model_revision=MODEL_REV, text_revision=T5_REV))
    bridge.load_supra_contexts(args.contexts, args.contexts_sha256, config)
    if file_sha(args.cache) != args.cache_sha256:
        raise ValueError("Native teacher cache checksum differs")
    if args.out.exists() and any(args.out.iterdir()) and not args.resume:
        parser.error("Use an empty output directory or a full GAN --resume checkpoint")

    import torch
    from supra.pg09 import build_run, frozen_digest, save_checkpoint, load_checkpoint, verify_teacher_cache
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    runtime = SupraRuntime(args.device, rank=16, allow_hub=False)
    run = build_run(runtime, config, args.cache, args.contexts,
        cache_sha256=args.cache_sha256, contexts_sha256=args.contexts_sha256,
        expected_frozen_sha256=args.frozen_base_sha256, total_steps=args.steps, seed=args.seed)
    # The bridge consumes only frozen contexts; no hidden encode while training.
    runtime.text_encoder.to("cpu")
    del runtime.text_encoder, runtime.tokenizer
    def no_encode(prompt):
        if prompt not in runtime.cache:
            raise ValueError("Prompt is outside frozen admitted contexts")
        return runtime.cache[prompt]
    runtime.encode = no_encode
    if args.resume:
        load_checkpoint(run, args.resume)
    if stop <= run.game.completed_steps:
        raise ValueError("Pause step must be beyond the actual restored step")
    parity = verify_teacher_cache(run)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "teacher-parity.json").write_text(json.dumps(parity, indent=2) + "\n")
    with (args.out / "train.jsonl").open("a") as log:
        while run.game.completed_steps < stop:
            metrics = run.step()
            log.write(json.dumps(metrics, allow_nan=False) + "\n")
            log.flush()
            if run.game.completed_steps % args.save_every == 0 or run.game.completed_steps == stop:
                if frozen_digest(runtime) != args.frozen_base_sha256:
                    raise RuntimeError("Frozen base changed during shared GAN training")
                saved = save_checkpoint(run, args.out / f"step-{run.game.completed_steps:06d}")
                print(saved, flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Render an arbitrary prompt with a lean, clean Supra ParticleGAN adapter."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from PIL import Image

from supra.particle_export import load_particle_adapter
from supra.runtime import SupraRuntime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scale", type=float, default=1.)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg", type=float, default=3.)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-hub", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.steps <= 0 or not math.isfinite(args.scale) or not math.isfinite(args.cfg):
        parser.error("steps must be positive and scale/CFG must be finite")
    started = time.perf_counter()
    runtime = SupraRuntime(device=args.device, rank=16, allow_hub=args.allow_hub)
    adapter = load_particle_adapter(runtime.model, args.adapter, device=runtime.device)
    context, mask = runtime.encode(args.prompt)
    uncond, uncond_mask = runtime.encode("")
    # This stream selects initial image noise only. Clean particle routing does
    # not draw from training, controller or inference perturbation streams.
    z = runtime.noise(args.seed)
    with torch.no_grad():
        for step in range(args.steps):
            velocity = adapter.velocity(z, step / args.steps, context, mask, uncond, uncond_mask,
                                        strength=args.scale, cfg=args.cfg)
            z = z + velocity / args.steps
        with torch.autocast(device_type=runtime.device.type, dtype=torch.bfloat16):
            decoded = runtime.vae.decode(z / .18215).sample
        pixels = ((decoded.float().clamp(-1, 1) + 1) * 127.5).round().byte()[0]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels.permute(1, 2, 0).cpu().numpy()).save(args.out)
    report = dict(adapter=str(args.adapter.resolve()),
                  adapter_sha256=hashlib.sha256(args.adapter.read_bytes()).hexdigest(),
                  prompt=args.prompt, seed=args.seed, scale=args.scale, cfg=args.cfg,
                  steps=args.steps, sampling="clean", output=str(args.out.resolve()),
                  served_source=adapter.metadata["served_source"],
                  training_completed_steps=adapter.metadata["completed_steps"],
                  elapsed_seconds=time.perf_counter()-started)
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

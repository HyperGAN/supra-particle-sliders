#!/usr/bin/env python3
"""Render the base Supra model or a saved native Supra LoRA."""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from supra.runtime import SupraRuntime


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompt", required=True)
    p.add_argument("--adapter", type=Path)
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--cfg", type=float, default=3)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--allow-hub", action="store_true")
    p.add_argument("--shared-root", type=Path, help="Reviewed external particle-sliders checkout (or PARTICLE_SLIDERS_ROOT)")
    args = p.parse_args()
    if args.shared_root:
        os.environ["PARTICLE_SLIDERS_ROOT"] = str(args.shared_root.resolve())
    import torch
    torch.set_num_threads(8)
    rank = 16
    if args.adapter:
        import json
        from safetensors import safe_open
        with safe_open(str(args.adapter), framework="pt") as handle:
            rank = json.loads(handle.metadata()["rank"])
    rt = SupraRuntime(args.device, rank, args.allow_hub)
    if args.adapter:
        rt.load(args.adapter)
    image = rt.render(args.prompt, args.seed, args.scale if args.adapter else 0, args.steps, args.cfg)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.out)
    print(args.out)


if __name__ == "__main__":
    main()

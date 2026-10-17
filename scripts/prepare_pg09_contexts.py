#!/usr/bin/env python3
"""Owner-admitted, bounded native context export. Requires separate GPU admission."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/supra/prompts-final-boss.yaml")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    os.environ["PARTICLE_SLIDERS_ROOT"] = str(args.shared_root.resolve())
    from supra.shared import shared_module
    bridge = shared_module("conceptmod.textsliders.pg09_remaining_campaigns")
    from supra.runtime import SupraRuntime, MODEL_REV, T5_REV
    import yaml
    config = yaml.safe_load(args.config.read_text())
    identity = dict(model_revision=MODEL_REV, text_revision=T5_REV)
    bridge.validate_live_supra_context_preparation(config, identity)
    if args.out.exists() or args.out.with_suffix(".json").exists():
        raise FileExistsError(args.out)
    runtime = SupraRuntime(args.device, rank=16, allow_hub=False)
    if not runtime.stored_unconditional:
        raise ValueError("Reviewed native stored unconditional context required")
    bridge.export_live_supra_contexts(runtime, config, args.out, runtime_identity=identity)


if __name__ == "__main__":
    main()

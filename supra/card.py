"""Native Supra sampling constants; all training recipe controls are shared."""
from __future__ import annotations

import json
from pathlib import Path

# Base checkpoint. Inference and the ordinary-LoRA release pin the same revision.
BASE_MODEL_ID = "SupraLabs/Supra2-IMG"
BASE_MODEL_REVISION = "10dec6e4b4b5d1c44fd1d7d3fe5e50137333da5b"
T5_ID = "google/flan-t5-base"
T5_REVISION = "7bcac572ce56db69c1ea7c8af255c5d7c9672fc2"
VAE_ID = "stabilityai/sd-vae-ft-mse"
VAE_REVISION = "31f26fdeee1355a5c34592e401dd41e45d25a493"

# Concept-slider product repo on the Hub. The measured ordinary-LoRA release
# stays on ntc-ai/supra-particle-sliders; this id is the slider product.
PRODUCT_HUB_ID = "ntc-ai/supra-concept-sliders"
RELEASE_HUB_ID = "ntc-ai/supra-particle-sliders"

# Hub Euler card. t = i/K, z <- z + (1/K) * v. Not Anima's FlowMatch schedule.
RESOLUTION = 256
SAMPLE_STEPS = 50
CFG = 3.0
EULER_SCHEDULE = "t=i/K, z<-z+(1/K)*v"

# The dependency identity follows the single reviewed package lock.
CORE_COMMIT = json.loads((Path(__file__).resolve().parents[1] / "shared.lock.json").read_text())["commit"]
CORE_REQUIREMENT = (
    "particle-sliders-core @ git+https://github.com/HyperGAN/particle-sliders.git@"
    f"{CORE_COMMIT}#subdirectory=packages/particle-sliders-core"
)

# ComfyUI is not in this product yet. The node, when it exists, belongs here.
COMFY_CLASS = None

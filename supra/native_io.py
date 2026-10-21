"""Supra-owned native loading/observation/sampling; no training dependencies."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from particle_sliders.lifecycle import native_cleanup

from .shared import file_sha, shared_root, source_cohort

LOCK = Path(__file__).with_name("core081-receipt.json")


def verify_sources():
    with source_cohort(LOCK):
        shared_root()


def model_identity(runtime):
    from .runtime import VAE_REV
    return dict(model_revision=runtime.MODEL_REV, text_revision=runtime.T5_REV,
        vae_revision=VAE_REV, frozen_parameter_sha256=frozen_digest(runtime),
        core_receipt_sha256=file_sha(LOCK),
        product_sources={name: file_sha(Path(__file__).with_name(name)) for name in
            ("runtime.py", "shared.py", "native_model.py", "native_io.py", "thin-product-provenance.json")})


def frozen_digest(runtime):
    """Same product frozen-parameter digest, without importing training IO."""
    import torch
    h = hashlib.sha256()
    for name, value in list(runtime.model.named_parameters()) + list(runtime.model.named_buffers()):
        if value.requires_grad:
            continue
        value = value.detach().cpu().contiguous()
        h.update(json.dumps([name, list(value.shape), str(value.dtype)]).encode())
        h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def load_native(options):
    from . import native_model
    from .runtime import SupraRuntime
    verify_sources()
    if options["device"] == "cpu":
        raise ValueError("Native Supra requires CUDA; CPU qualification uses explicitly staged fixtures")
    return SupraRuntime(options["device"], rank=options.get("rank", 16),
                        allow_hub=False, native_model=native_model)


def snapshot_native(runtime):
    """Restore the specific borrowed Supra attributes mutated during binding."""
    cleanups = [native_cleanup(root) for root in
        (runtime.model, getattr(runtime, "text_encoder", None), getattr(runtime, "vae", None))
        if root is not None]
    cache, branch = runtime.cache, runtime.cfg_adapter_branch
    had_encode, encode = "encode" in vars(runtime), vars(runtime).get("encode")
    tokens = getattr(runtime, "token_counts", None)
    token_values = deepcopy(tokens)
    multipliers = [(module, module.multiplier) for module in runtime.loras]

    def restore():
        for cleanup in reversed(cleanups):
            cleanup()
        runtime.cache, runtime.cfg_adapter_branch = cache, branch
        if had_encode:
            runtime.encode = encode
        else:
            vars(runtime).pop("encode", None)
        if tokens is not None:
            tokens.clear()
            tokens.update(token_values)
            runtime.token_counts = tokens
        for module, multiplier in multipliers:
            module.multiplier = multiplier
    return restore


@contextmanager
def observation(runtime, encode):
    restore = snapshot_native(runtime)
    runtime.cache = dict(runtime.cache)
    runtime.encode = encode
    try:
        with source_cohort(LOCK):
            yield
    finally:
        restore()


def validate_render(native, *, partial=False):
    allowed = {"prompt", "steps", "cfg", "output_type", "images"}
    if set(native) - allowed:
        raise ValueError("Unsupported Supra native render argument")
    if not partial or "prompt" in native:
        if not isinstance(native.get("prompt"), str) or not native["prompt"]:
            raise ValueError("Supra rendering requires a nonempty prompt")
    steps, cfg = native.get("steps", 50), native.get("cfg", 3.)
    if (native.get("images") or native.get("output_type", "pil") != "pil"
            or type(steps) is not int or not 1 <= steps <= 1000
            or isinstance(cfg, bool) or not isinstance(cfg, (int, float)) or not math.isfinite(cfg) or cfg < 0):
        raise ValueError("Supra needs no references, PIL output, bounded steps and finite nonnegative CFG")


def render(backend, runtime, *, prompt, strength, seed, images=None, steps=50, cfg=3., output_type="pil"):
    """Keep native seed/noise, rising flow time, positive Euler and VAE decode."""
    import torch
    from PIL import Image
    from particle_sliders.pg09_backends import FlowSchedule, ResponseRow
    validate_render(dict(prompt=prompt, images=images, steps=steps, cfg=cfg, output_type=output_type))
    if type(seed) is not int or not 0 <= seed < 2**63 - 100 or not math.isfinite(strength):
        raise ValueError("Finite strength and bounded integer seed required")
    row = ResponseRow(prompt, prompt, "", runtime.noise(seed),
                      schedule=FlowSchedule.supra(steps), guidance=cfg)
    image = backend.render(row, scale=strength)
    if not torch.isfinite(image).all():
        raise FloatingPointError("Nonfinite native Supra VAE output")
    pixels = ((image.float().clamp(-1, 1) + 1) * 127.5).round().byte()[0]
    return Image.fromarray(pixels.permute(1, 2, 0).cpu().numpy())

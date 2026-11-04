"""Supra-owned native loading/observation/sampling; no training dependencies."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys

from particle_sliders.lifecycle import native_cleanup

from .shared import file_sha, source_cohort
from .shared import verify_imported_sources

LOCK = Path(__file__).with_name("core012-ddb4142-receipt.json")


def read_context_receipt(contexts):
    try:
        receipt = json.loads(Path(contexts).with_suffix(".json").read_text())
    except (OSError, ValueError) as error:
        raise ValueError("Readable JSON frozen context receipt required") from error
    if not isinstance(receipt, dict):
        raise ValueError("Frozen context receipt must be a JSON object")
    return receipt


def verify_sources():
    """Admit the exact current installed core and official ParticleGAN release."""
    from importlib.metadata import version
    receipt = verify_imported_sources(LOCK)
    if (receipt.get("schema") != "supra-recursive-core-closure-v1"
            or receipt.get("source_commit") != "ddb41423bcefc1fe76cd94d0f9b367353b3ea137"
            or receipt.get("namespaces") != ["concept_slider_core", "particle_sliders"]):
        raise ValueError("Current core closure identity contradicts the source receipt")
    import particle_sliders
    base = Path(particle_sliders.__file__).resolve().parent.parent
    actual = {str(path.relative_to(base)): file_sha(path)
              for namespace in receipt["namespaces"] for path in (base / namespace).rglob("*")
              if path.is_file() and "__pycache__" not in path.parts and path.suffix not in (".pyc", ".pyo")}
    if actual != receipt["source_sha256"]:
        raise RuntimeError("Imported shared recursive source/data closure differs from receipt")
    for package, field in (("particle-sliders-core", "version"), ("particlegan", "pg_version")):
        if version(package) != receipt[field]:
            raise ValueError("Installed " + package + " version contradicts the source receipt")
    from particle_sliders.pg09 import _released_sources
    _released_sources()


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
        primary = sys.exception()
        errors = []

        def attempt(operation):
            try:
                operation()
            except BaseException as error:
                errors.append(error)

        for cleanup in reversed(cleanups):
            attempt(cleanup)
        attempt(lambda: setattr(runtime, "cache", cache))
        attempt(lambda: setattr(runtime, "cfg_adapter_branch", branch))
        if had_encode:
            attempt(lambda: setattr(runtime, "encode", encode))
        else:
            attempt(lambda: vars(runtime).pop("encode", None))
        if tokens is not None:
            attempt(tokens.clear)
            attempt(lambda: tokens.update(token_values))
            attempt(lambda: setattr(runtime, "token_counts", tokens))
        for module, multiplier in multipliers:
            attempt(lambda module=module, multiplier=multiplier: setattr(module, "multiplier", multiplier))
        if errors:
            if primary is not None:
                for error in errors:
                    primary.add_note("Supra native cleanup failed: " + str(error))
            else:
                for error in errors[1:]:
                    errors[0].add_note("Additional Supra cleanup failure: " + str(error))
                raise errors[0]
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

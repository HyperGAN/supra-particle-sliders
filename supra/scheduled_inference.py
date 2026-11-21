"""Per-step strength on an already opened scoped Supra RAW inference owner."""
from math import isfinite
from sys import float_info

from PIL import Image
import torch
from particle_sliders.backends import FlowSchedule, ResponseRow

from .native_io import validate_render
from .step_strength import scheduled_latent


@torch.no_grad()
def render(owner, *, prompt, strength, seed, step_gains, steps=50, cfg=3.0):
    """Scope the request to the original conditional-only Euler/VAE path."""
    from supra_scoped.training import observed

    owner.require_open()
    validate_render(dict(prompt=prompt, steps=steps, cfg=cfg))
    if type(seed) is not int or not 0 <= seed < 2**63 - 100 or not isfinite(strength):
        raise ValueError("Finite strength and bounded integer seed required")
    if not isinstance(step_gains, (tuple, list)) or len(step_gains) != steps:
        raise ValueError("Supra requires one step gain per native Euler step")
    if any(type(gain) not in (int, float) or not 0 <= gain <= float_info.max
           for gain in step_gains):
        raise ValueError("Supra step gains must be finite and nonnegative")
    strengths = [strength * gain for gain in step_gains]
    if any(not isfinite(value) for value in strengths):
        raise ValueError("Effective Supra strengths must be finite")

    schedule = FlowSchedule.increasing_linear(steps)
    backend = owner.backend
    with observed(owner, seed):
        latent = owner.runtime.noise(seed)
        row = ResponseRow(prompt, prompt, "", latent, schedule=schedule, guidance=cfg)
        latent = scheduled_latent(backend, row, strengths)
        image = backend.decoder(latent)
        if not torch.isfinite(image).all():
            raise FloatingPointError("Nonfinite native Supra VAE output")
        pixels = ((image.float().clamp(-1, 1) + 1) * 127.5).round().byte()[0]
        output = Image.fromarray(pixels.permute(1, 2, 0).cpu().numpy())
    return output, dict(timesteps=list(schedule.timesteps), deltas=list(schedule.deltas),
                        effective_strengths=strengths)

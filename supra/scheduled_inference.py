"""Per-interval fixed or activation-dependent strength on scoped Supra RAW."""
from math import isfinite
from sys import float_info

from PIL import Image
import torch
from particle_sliders.backends import FlowSchedule, ResponseRow

from .native_io import validate_render
from .step_strength import scheduled_latent


def _resolve_fixed_strengths(strength, steps, step_gains):
    if not isinstance(step_gains, (tuple, list)) or len(step_gains) != steps:
        raise ValueError("Supra requires one step gain per native Euler step")
    if any(type(gain) not in (int, float) or not 0 <= gain <= float_info.max
           for gain in step_gains):
        raise ValueError("Supra step gains must be finite and nonnegative")
    strengths = [strength * gain for gain in step_gains]
    if any(not isfinite(value) for value in strengths):
        raise ValueError("Effective Supra strengths must be finite")
    return strengths


def _adaptive_latent(backend, row, strength, gate, alignment):
    latent = row.latent
    schedule = row.schedule
    decisions = []
    for index, (time, delta) in enumerate(zip(schedule.timesteps, schedule.deltas, strict=True)):
        timestep = latent.new_tensor(time)
        score = None if alignment is None else float(alignment(latent.detach(), timestep.detach()))
        if score is not None and not isfinite(score):
            raise ValueError("Supra alignment must return one finite scalar")
        decision = gate.step(score)
        value = strength if decision.fired else 0.0
        velocity = backend.cfg(row, latent, timestep, scale=value)
        latent = backend._advance(latent, velocity, delta)
        decisions.append(dict(interval=index, time=time, delta=delta, alignment=score,
                              probability=decision.probability, fired=decision.fired,
                              effective_strength=value))
    return latent, decisions


@torch.no_grad()
def render(owner, *, prompt, strength, seed, step_gains=None, gate=None, alignment=None,
           steps=50, cfg=3.0):
    """Render through the original conditional-only Euler/VAE path.

    A fresh shared ActivationGate and read-only alignment(latent, timestep)
    callback use the live pre-edit solver latent and native rising flow time.
    The matched random gate needs no alignment callback.
    """
    from supra_scoped.training import observed

    owner.require_open()
    validate_render(dict(prompt=prompt, steps=steps, cfg=cfg))
    if type(seed) is not int or not 0 <= seed < 2**63 - 100 or not isfinite(strength):
        raise ValueError("Finite strength and bounded integer seed required")
    if gate is None:
        if alignment is not None:
            raise ValueError("Supra activation alignment requires a gate")
        strengths = _resolve_fixed_strengths(strength, steps, step_gains)
    else:
        from particle_sliders.adaptive_steering import ActivationGate

        if step_gains is not None:
            raise ValueError("Choose either Supra step gains or an activation gate")
        if not isinstance(gate, ActivationGate) or gate.usage.eligible:
            raise ValueError("Supra adaptive rendering requires a fresh shared gate per request")
        if strength == 0:
            raise ValueError("Supra adaptive rendering requires nonzero nominal strength")
        if alignment is not None and not callable(alignment):
            raise TypeError("Supra alignment must be a callable")

    schedule = FlowSchedule.increasing_linear(steps)
    receipt = dict(timesteps=list(schedule.timesteps), deltas=list(schedule.deltas))
    backend = owner.backend
    with observed(owner, seed):
        latent = owner.runtime.noise(seed)
        row = ResponseRow(prompt, prompt, "", latent, schedule=schedule, guidance=cfg)
        if gate is None:
            latent = scheduled_latent(backend, row, strengths)
        else:
            latent, decisions = _adaptive_latent(backend, row, strength, gate, alignment)
            usage = gate.usage
            strengths = [decision["effective_strength"] for decision in decisions]
            receipt.update(decisions=decisions, fired=usage.fired, eligible=usage.eligible,
                           gate_rate=usage.rate,
                           interval_weighted_dose=sum(abs(delta * value) for delta, value in
                                                      zip(schedule.deltas, strengths, strict=True)))
        image = backend.decoder(latent)
        if not torch.isfinite(image).all():
            raise FloatingPointError("Nonfinite native Supra VAE output")
        pixels = ((image.float().clamp(-1, 1) + 1) * 127.5).round().byte()[0]
        output = Image.fromarray(pixels.permute(1, 2, 0).cpu().numpy())
    return output, dict(receipt, effective_strengths=strengths)

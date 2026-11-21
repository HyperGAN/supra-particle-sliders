from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from particle_sliders.backends import FrozenResponseAdapter

from supra import native_io
from supra.scheduled_inference import render
from supra_scoped import training


def test_schedule_uses_original_euler_interval_and_observation_scope(monkeypatch):
    calls = []
    latent = torch.zeros(1, 3, 1, 1)

    @contextmanager
    def observed(owner, seed):
        calls.append(("enter", seed))
        try:
            yield
        finally:
            calls.append(("exit", seed))

    def cfg(row, current, timestep, *, scale):
        calls.append((float(timestep), scale, row.guidance))
        return current.new_full(current.shape, scale)

    monkeypatch.setattr(training, "observed", observed)
    runtime = SimpleNamespace(noise=lambda seed: latent)
    backend = SimpleNamespace(cfg=cfg, _advance=lambda z, v, d: z + v * d, decoder=lambda z: z)
    owner = SimpleNamespace(runtime=runtime, backend=backend, require_open=lambda: None)

    image, receipt = render(owner, prompt="room", strength=0.5, seed=7, cfg=3.0,
                            steps=2, step_gains=[1.5, 0.5])

    assert calls == [("enter", 7), (0.0, 0.75, 3.0), (0.5, 0.25, 3.0), ("exit", 7)]
    assert receipt == dict(timesteps=[0.0, 0.5], deltas=[0.5, 0.5], effective_strengths=[0.75, 0.25])
    assert image.getpixel((0, 0)) == (191, 191, 191)
    assert latent.count_nonzero() == 0


def test_bad_schedule_refused_before_native_noise():
    owner = SimpleNamespace(require_open=lambda: None)
    with pytest.raises(ValueError, match="one step gain"):
        render(owner, prompt="room", strength=1.0, seed=7, steps=2, step_gains=[1.5])


def test_constant_gains_match_native_image_and_leave_negative_branch_off(monkeypatch):
    from contextlib import nullcontext

    active = 0.25
    scales = []

    class Network(torch.nn.Module):
        @contextmanager
        def scaled(self, value):
            nonlocal active
            saved = active
            active = value
            try:
                yield
            finally:
                active = saved

    class Backend(FrozenResponseAdapter):
        def _velocity(self, condition, latent, timestep):
            scales.append((condition, active))
            return latent * timestep + active

    backend = Backend(Network(), decoder=lambda z: z, trajectory_direction="increasing")
    runtime = SimpleNamespace(noise=lambda seed: torch.zeros(1, 3, 1, 1))
    owner = SimpleNamespace(backend=backend, runtime=runtime, require_open=lambda: None)
    monkeypatch.setattr(training, "observed", lambda *_: nullcontext())
    request = dict(prompt="room", strength=0.25, seed=7, steps=2, cfg=3.0)
    original = native_io.render(backend, runtime, **request)
    expected_scales = [("room", 0.25), ("", 0.0)] * 2
    assert scales == expected_scales

    scales.clear()
    scheduled, _ = render(owner, **request, step_gains=[1.0, 1.0])
    assert scheduled.tobytes() == original.tobytes()
    assert scales == expected_scales and active == 0.25

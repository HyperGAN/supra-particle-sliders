from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

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

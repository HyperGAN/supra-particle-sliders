from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from particle_sliders.adaptive_steering import ActivationGate, activation_projection
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


@pytest.fixture
def adaptive_owner(monkeypatch):
    events = []

    class Network(torch.nn.Module):
        strength = 0.125

        @contextmanager
        def scaled(self, value):
            saved, self.strength = self.strength, value
            try:
                yield
            finally:
                self.strength = saved

    class Backend(FrozenResponseAdapter):
        def _velocity(self, condition, latent, timestep):
            events.append((condition, self.network.strength))
            return torch.full_like(latent, self.network.strength)

    @contextmanager
    def observed(owner, seed):
        events.append(("enter", seed))
        try:
            yield
        finally:
            events.append(("exit", seed))

    monkeypatch.setattr(training, "observed", observed)
    runtime = SimpleNamespace(noise=lambda seed: torch.zeros(1, 3, 1, 1))
    backend = Backend(Network(), decoder=lambda latent: latent, trajectory_direction="increasing")
    owner = SimpleNamespace(backend=backend, runtime=runtime, require_open=lambda: None)
    return owner, events


def test_adaptive_reads_live_pre_edit_latent_once_per_interval(adaptive_owner):
    owner, events = adaptive_owner
    request = dict(prompt="room", strength=0.25, seed=7, steps=2, cfg=3.0)
    direction = torch.tensor([1.0, 0.0, 0.0])
    gate = ActivationGate("cfp", 0.1)

    def alignment(latent, timestep):
        score = activation_projection(latent[0].mean(dim=(-2, -1)), direction,
                                      strength=request["strength"]).item()
        events.append(("alignment", float(timestep), score))
        return score

    adaptive, receipt = render(owner, **request, gate=gate, alignment=alignment)
    assert events == [
        ("enter", 7), ("alignment", 0.0, 0.0), ("room", 0.25), ("", 0.0),
        ("alignment", 0.5, 0.375), ("room", 0.0), ("", 0.0), ("exit", 7),
    ]
    assert receipt == dict(
        timesteps=[0.0, 0.5], deltas=[0.5, 0.5], effective_strengths=[0.25, 0.0],
        fired=1, eligible=2, gate_rate=0.5, interval_weighted_dose=0.125,
        decisions=[
            dict(interval=0, time=0.0, delta=0.5, alignment=0.0, probability=1.0,
                 fired=True, effective_strength=0.25),
            dict(interval=1, time=0.5, delta=0.5, alignment=0.375, probability=0.0,
                 fired=False, effective_strength=0.0),
        ],
    )
    fixed, _ = render(owner, **request, step_gains=[1.0, 0.0])
    assert adaptive.tobytes() == fixed.tobytes()
    assert owner.backend.network.strength == 0.125
    events.clear()
    with pytest.raises(ValueError, match="fresh shared gate"):
        render(owner, **request, gate=gate, alignment=alignment)
    assert not events


@pytest.mark.parametrize("rate", [0.0, 1.0])
def test_random_endpoints_match_native_off_and_constant(adaptive_owner, rate):
    owner, _ = adaptive_owner
    request = dict(prompt="room", strength=-0.25, seed=7, steps=2, cfg=3.0)
    adaptive, receipt = render(owner, **request, gate=ActivationGate("random", rate, seed=11))
    direct = native_io.render(owner.backend, owner.runtime, **dict(request, strength=-0.25 * rate))
    assert adaptive.tobytes() == direct.tobytes()
    assert receipt["effective_strengths"] == [-0.25 * rate] * 2
    assert receipt["interval_weighted_dose"] == 0.25 * rate
    assert owner.backend.network.strength == 0.125


def test_failed_alignment_leaves_owner_and_native_multiplier_restored(adaptive_owner):
    owner, events = adaptive_owner
    gate = ActivationGate("cfp", 0.0)

    def alignment(latent, timestep):
        return 0.0 if timestep == 0 else float("nan")

    with pytest.raises(ValueError, match="finite scalar"):
        render(owner, prompt="room", strength=0.25, seed=7, steps=2,
               gate=gate, alignment=alignment)
    assert events == [("enter", 7), ("room", 0.25), ("", 0.0), ("exit", 7)]
    assert gate.usage.eligible == 1
    assert owner.backend.network.strength == 0.125

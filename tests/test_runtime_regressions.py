"""CPU regressions for artifact loading and the actual native/shared CFG paths."""
from contextlib import nullcontext
import json

import pytest
import torch
from safetensors.torch import save_file

from supra.runtime import (
    MODEL_ID, MODEL_REV, T5_REV, TARGETS, VAE_REV, SupraRuntime,
    adapter_state, load_adapter_state, model_module,
)
from supra.shared import LOCK, file_sha, shared_module


@pytest.fixture(autouse=True)
def no_cuda_initialization():
    assert not torch.cuda.is_initialized()
    yield
    assert not torch.cuda.is_initialized()


@pytest.fixture
def runtime(monkeypatch):
    """Use the shared LoRA itself, with an analytically checkable native model."""
    lora_type = model_module().LoRALinear

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = lora_type(torch.nn.Linear(1, 1, bias=False), rank=1, alpha=1)
            with torch.no_grad():
                self.projection.base.weight.fill_(2)
                self.projection.down.weight.fill_(1)
                self.projection.up.weight.fill_(0.5)
            self.calls = []

        def forward(self, latent, time, context, mask):
            self.calls.append(dict(scale=self.projection.multiplier,
                                   times=time.detach().tolist(),
                                   contexts=context[:, 0, 0].detach().tolist()))
            value = self.projection(context[:, 0])[:, :, None, None]
            return latent + value + time[:, None, None, None]

    # __init__ is deliberately bypassed: no pretrained loader or CUDA admission.
    rt = SupraRuntime.__new__(SupraRuntime)
    rt.device, rt.rank = torch.device("cpu"), 1
    rt.model = TinyModel()
    rt.loras = [rt.model.projection]
    rt.cfg_adapter_branch = "both"
    rt.cache = {prompt: (torch.full((1, 2, 1), value), torch.ones(1, 2))
                for prompt, value in (("neutral", 2.0), ("", 1.0))}
    rt.stored_unconditional = True
    # Legacy native forwards historically enter CUDA autocast unconditionally.
    # Keep that branch CPU-only while checking its actual batching/CFG logic.
    monkeypatch.setattr(torch, "autocast", lambda *args, **kwargs: nullcontext())
    return rt


def snapshot(model):
    return {name: value.detach().clone() for name, value in model.named_parameters()}


def assert_unchanged(model, expected):
    assert all(torch.equal(value, expected[name]) for name, value in model.named_parameters())


@pytest.mark.parametrize("corruption", ["shape", "nan", "inf", "dtype", "non_tensor"])
def test_corrupt_last_adapter_tensor_never_partially_loads(runtime, corruption):
    before = snapshot(runtime.model)
    state = {name: value.clone() for name, value in adapter_state(runtime.model).items()}
    first, last = list(state)[0], list(state)[-1]
    state[first].fill_(123)  # A valid earlier leaf must not copy before rejection.
    if corruption == "shape":
        state[last] = torch.zeros(2)
    elif corruption in ("nan", "inf"):
        state[last].fill_(float(corruption))
    elif corruption == "dtype":
        state[last] = state[last].double()
    else:
        state[last] = None
    with pytest.raises(ValueError, match="schema|finite|tensor"):
        load_adapter_state(runtime.model, state)
    assert_unchanged(runtime.model, before)


def write_adapter(runtime, path, extra=None):
    metadata = dict(format="supra-native-lora-v1", model_id=MODEL_ID,
                    model_revision=MODEL_REV, rank=runtime.rank, alpha=runtime.rank,
                    targets=list(TARGETS), text_encoder_revision=T5_REV,
                    vae_revision=VAE_REV)
    metadata.update(extra or {})
    save_file({name: value.clone() for name, value in adapter_state(runtime.model).items()},
              str(path), metadata={key: json.dumps(value) for key, value in metadata.items()})


def test_old_adapter_without_branch_metadata_retains_legacy_both_policy(runtime, tmp_path):
    path = tmp_path / "legacy.safetensors"
    write_adapter(runtime, path)
    runtime.cfg_adapter_branch = "conditional_only"
    runtime.load(path)
    assert runtime.cfg_adapter_branch == "both"


def test_pg_adapter_load_selects_shared_conditional_only_policy(runtime, tmp_path):
    path = tmp_path / "pg.safetensors"
    write_adapter(runtime, path, dict(cfg_adapter_branch="conditional_only",
        training_objective="pg09_gan_only", shared_lock_sha256=file_sha(LOCK)))
    expected = snapshot(runtime.model)
    with torch.no_grad():
        for parameter in runtime.parameters():
            parameter.zero_()
    runtime.load(path)
    assert runtime.cfg_adapter_branch == "conditional_only"
    assert_unchanged(runtime.model, expected)


@pytest.mark.parametrize("extra", [
    dict(cfg_adapter_branch="unknown"),
    dict(cfg_adapter_branch="conditional_only", training_objective="mse", shared_lock_sha256=file_sha(LOCK)),
    dict(cfg_adapter_branch="conditional_only", training_objective="pg09_gan_only", shared_lock_sha256="invalid"),
])
def test_invalid_adapter_policy_is_rejected_before_weight_or_policy_mutation(runtime, tmp_path, extra):
    path = tmp_path / "invalid.safetensors"
    write_adapter(runtime, path, extra)
    with torch.no_grad():
        for parameter in runtime.parameters():
            parameter.add_(10)
    before = snapshot(runtime.model)
    with pytest.raises(ValueError, match="branch|objective|binding"):
        runtime.load(path)
    assert runtime.cfg_adapter_branch == "both"
    assert_unchanged(runtime.model, before)


@pytest.mark.parametrize("guidance", [0.0, 0.5, 1.0, 3.0])
def test_legacy_cfg_honors_guidance_below_one(runtime, guidance):
    actual = runtime.velocity("neutral", torch.zeros(1, 4, 2, 2), 0.25, scale=0, cfg=guidance)
    # Frozen conditional=4+t and unconditional=2+t, on this native fixture.
    torch.testing.assert_close(actual, torch.full_like(actual, 2.25 + guidance * 2), rtol=0, atol=0)


@pytest.mark.parametrize("scale", [-1.0, 0.0, 0.5, 1.0])
@pytest.mark.parametrize("guidance", [0.0, 0.5, 1.0, 3.0])
def test_pg_velocity_executes_shared_cfg_with_frozen_negative_branch_and_native_time(runtime, scale, guidance):
    runtime.cfg_adapter_branch = "conditional_only"
    z = torch.zeros(1, 4, 2, 2)
    actual = runtime.velocity("neutral", z, 0.25, scale=scale, cfg=guidance)
    # The signed edit is scale*1 on the conditional branch; negative stays 2+t.
    expected = 2.25 + guidance * (2 + scale)
    torch.testing.assert_close(actual, torch.full_like(actual, expected), rtol=0, atol=0)
    calls = runtime.model.calls
    assert calls[0] == dict(scale=scale, times=[0.25], contexts=[2.0])
    if guidance == 1:
        assert len(calls) == 1
    else:
        assert calls[1] == dict(scale=0.0, times=[0.25], contexts=[1.0])
    assert runtime.loras[0].multiplier == 1.0


def test_native_trajectory_and_shared_schedule_use_positive_euler_flow_time(runtime):
    calls = []
    runtime.noise = lambda seed: torch.full((1, 4, 2, 2), 5.0)

    def constant_velocity(prompt, z, time, *, scale, cfg):
        calls.append((time, scale, cfg))
        return torch.full_like(z, 3.0)

    runtime.velocity = constant_velocity
    end, states = runtime.trajectory("neutral", 99, steps=4, scale=-0.5, cfg=3, keep=True)
    assert calls == [(time, -0.5, 3) for time in (0.0, 0.25, 0.5, 0.75)]
    assert [float(state.flatten()[0]) for state in states] == [5.0, 5.75, 6.5, 7.25]
    assert torch.equal(end, torch.full_like(end, 8.0))
    schedule = shared_module("conceptmod.textsliders.pg09_backends").FlowSchedule.supra(4)
    assert schedule.timesteps == (0.0, 0.25, 0.5, 0.75)
    assert schedule.deltas == (0.25, 0.25, 0.25, 0.25)

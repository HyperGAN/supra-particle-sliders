"""Execute the product seam on a small native CPU DiT, never pretrained weights."""
from contextlib import nullcontext
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch
import yaml

from supra.pg09 import build_run, frozen_digest, load_checkpoint, save_checkpoint
from supra.runtime import SupraRuntime, TARGETS, model_module
from supra.shared import file_sha, shared_module

ROOT = Path(__file__).resolve().parents[1]


class FixtureRuntime(SupraRuntime):
    def __init__(self, config):
        torch.manual_seed(71)
        mod = model_module()
        self.device, self.rank, self.cfg_adapter_branch = torch.device("cpu"), 2, "both"
        self.model = mod.SupraDiT(latent_ch=4, d_model=16, depth=1, n_heads=2, ctx_dim=8,
                                  mlp_ratio=2., num_tokens=256, patch=2)
        self.loras = mod.attach_supra_lora(self.model, rank=2, alpha=2, targets=TARGETS)
        bridge = shared_module("conceptmod.textsliders.pg09_remaining_campaigns")
        self.cache = {p: (torch.randn(1, 3, 8), torch.ones(1, 3))
                      for p in bridge.required_supra_prompts(config)}
        self.stored_unconditional = True

    def encode(self, prompt):
        assert prompt in self.cache, "No implicit text encoding in CPU qualification"
        return self.cache[prompt]


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


@pytest.fixture
def admitted(tmp_path, monkeypatch):
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(1)
    monkeypatch.setattr(torch, "autocast", lambda *args, **kwargs: nullcontext())
    config = yaml.safe_load((ROOT / "configs/supra/prompts-final-boss.yaml").read_text())
    runtime = FixtureRuntime(config)
    bridge = shared_module("conceptmod.textsliders.pg09_remaining_campaigns")
    contexts = tmp_path / "contexts.safetensors"
    bridge.export_live_supra_contexts(runtime, config, contexts,
        runtime_identity=dict(model_revision=runtime.MODEL_REV, text_revision=runtime.T5_REV))
    records, holds = [], []
    with torch.no_grad():
        for row in config["rows"]:
            for role in ("neutral", "positive"):
                for time in (0., .5):
                    z = torch.randn(1, 4, 32, 32)
                    base = runtime.velocity(row["neutral"], z, time)
                    target = runtime.velocity(row["positive"], z, time)
                    records.append(dict(prompt=row["neutral"], row=row["name"], context=role,
                        seed=71, t=time, z=z, neutral=base, target=target))
        for prompt in config["preservation"]:
            z = torch.randn(1, 4, 32, 32)
            base = runtime.velocity(prompt, z, .5)
            holds.append(dict(prompt=prompt, t=.5, z=z, neutral=base, target=base.clone()))
    cache = tmp_path / "cache.pt"
    torch.save(dict(records=records, holds=holds), cache)
    # Synthetic values are never represented as the real reviewed cache scale.
    rows = bridge.SupraCampaignRows(cache, cache_sha256=file_sha(cache), config=config, contexts=runtime.cache)
    monkeypatch.setattr(bridge, "SUPRA_SCALE", rows.response_scale())
    yield runtime, config, cache, contexts
    assert not torch.cuda.is_initialized()


def run_for(admitted, runtime=None):
    original, config, cache, contexts = admitted
    runtime = original if runtime is None else runtime
    return build_run(runtime, config, cache, contexts,
        cache_sha256=file_sha(cache), contexts_sha256=file_sha(contexts),
        expected_frozen_sha256=frozen_digest(runtime), total_steps=4, critic_width=8)


def test_product_executes_shared_gan_and_fresh_checkpoint_replay(admitted, tmp_path, monkeypatch):
    from particle_sliders.pg09 import ConditionalPG09
    runtime, config, _, _ = admitted
    run = run_for(admitted)
    assert type(run.game) is ConditionalPG09
    assert run.game.sampling == "balanced" and run.game.batch_size == 4
    assert runtime.cfg_adapter_branch == "conditional_only"
    frozen = frozen_digest(runtime)
    adapter = deepcopy(run.backend.network.state_dict())
    critic = deepcopy(run.game.critic.state_dict())
    def forbidden(*args, **kwargs):
        raise AssertionError("MSE cannot be optimized by this GAN path")
    monkeypatch.setattr(torch.nn.functional, "mse_loss", forbidden)
    run.step()
    assert not equal(adapter, run.backend.network.state_dict())
    assert not equal(critic, run.game.critic.state_dict())
    assert frozen_digest(runtime) == frozen
    checkpoint = save_checkpoint(run, tmp_path / "step-000001")
    expected_metrics = run.step()
    expected_state = deepcopy(run.game.state_dict())
    restored = run_for(admitted, FixtureRuntime(config))
    load_checkpoint(restored, checkpoint)
    assert restored.game.completed_steps == 1
    assert restored.step() == expected_metrics
    assert equal(restored.game.state_dict(), expected_state)
    assert frozen_digest(restored.runtime) == frozen


def test_checkpoint_corruption_and_binding_mismatch_reject_without_mutation(admitted, tmp_path):
    run = run_for(admitted)
    directory = save_checkpoint(run, tmp_path / "step-0")
    before = deepcopy(run.game.state_dict())
    receipt = json.loads((directory / "receipt.json").read_text())
    (directory / "checkpoint.pt").write_bytes(b"broken")
    with pytest.raises(ValueError, match="checksum"):
        load_checkpoint(run, directory)
    assert equal(run.game.state_dict(), before)
    receipt["binding"]["config"]["rows"][0]["positive"] += " changed"
    (directory / "receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="binding"):
        load_checkpoint(run, directory)
    assert equal(run.game.state_dict(), before)
    with pytest.raises(FileExistsError):
        save_checkpoint(run, directory)


@pytest.mark.parametrize("mutation", ["over_budget", "same_count_wrong_config"])
def test_context_preparation_rejects_before_runtime_load_or_output(tmp_path, monkeypatch, mutation):
    config = yaml.safe_load((ROOT / "configs/supra/prompts-final-boss.yaml").read_text())
    if mutation == "over_budget":
        config["preservation"].append("A twentieth context.")
    else:
        config["rows"][0]["positive"] += " altered"
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid preparation must not load a native runtime")
    monkeypatch.setattr(SupraRuntime, "__init__", forbidden)
    spec = importlib.util.spec_from_file_location("prepare", ROOT / "scripts/prepare_pg09_contexts.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    import os
    monkeypatch.setattr(sys, "argv", ["prepare", "--shared-root", os.environ["PARTICLE_SLIDERS_ROOT"],
        "--config", str(path), "--out", str(tmp_path / "absent.safetensors")])
    with pytest.raises(ValueError, match="19|reviewed"):
        script.main()
    assert not (tmp_path / "absent.safetensors").exists()
    assert not torch.cuda.is_initialized()

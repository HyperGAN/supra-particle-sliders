"""Execute the separate HNDL lifecycle with the actual shared game on CPU."""
from copy import deepcopy
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path

import pytest

if version("particle-sliders-core") != "0.12.0":
    pytest.skip("requires the distinct core0.12 HyperGAN/HNDL environment", allow_module_level=True)

import torch

from particle_sliders.lifecycle import CheckpointError, SessionConfig, UnsupportedOperation
from particle_sliders.pg09 import _hash
from supra import cache, hndl, hndl_lifecycle, native_io
from supra.shared import file_sha

from .fixtures import TinyRuntime, assets, equal  # noqa: F401

@pytest.fixture
def configured(assets):
    runtime, config, path, contexts = assets
    hndl.bind_native_loras(runtime)
    rows = cache.SupraCampaignRows(path, cache_sha256=file_sha(path), config=config, contexts=runtime.cache)
    options = dict(native_config=config, cache=str(path), cache_sha256=file_sha(path),
        contexts=str(contexts), contexts_sha256=file_sha(contexts),
        frozen_base_sha256=native_io.frozen_digest(runtime), device="cpu", rank=2,
        steps=4, batch=4, seed=7, critic_width=8, ema_decay=.9)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(cache, "SUPRA_SCALE", rows.response_scale())
        yield runtime, SessionConfig("supra_hndl", "synthetic-supra-hndl-v1", **options)


def fresh_runtime(config):
    runtime = TinyRuntime(config.options["native_config"])
    hndl.bind_native_loras(runtime)
    return runtime


def test_real_hndl_training_owns_native_leaves_and_resumes_exactly(configured):
    from particle_sliders.fibo_hndl import HNDLFieldCritic
    from particle_sliders.pg09 import ConditionalPG09
    runtime, config = configured
    with hndl_lifecycle.BACKEND.open(config, runtime=runtime) as first:
        game = first.game
        assert type(game) is ConditionalPG09 and type(game.critic) is HNDLFieldCritic
        native = [parameter for parameter in runtime.model.parameters() if parameter.requires_grad]
        generator = [parameter for group in game.opt_g.param_groups for parameter in group["params"]]
        critic = [parameter for group in game.opt_d.param_groups for parameter in group["params"]]
        assert len(generator) == len({id(parameter) for parameter in generator})
        assert {id(parameter) for parameter in generator} == {id(parameter) for parameter in native}
        assert not {id(parameter) for parameter in generator} & {id(parameter) for parameter in critic}
        before = [parameter.detach().clone() for parameter in native]
        frozen = native_io.frozen_digest(runtime)
        row = game.rows[0].row
        with torch.no_grad():
            off = game.backend._response(row, scale=0.)
        first.step()
        assert any(not torch.equal(old, parameter) for old, parameter in zip(before, native))
        assert native_io.frozen_digest(runtime) == frozen
        assert all(parameter.grad is None for parameter in runtime.model.parameters()
                   if not parameter.requires_grad)
        with torch.no_grad():
            assert torch.equal(game.backend._response(row, scale=0.), off)
        packet = first.checkpoint()
        expected_metrics = first.step()
        expected_game, expected_ema = deepcopy(game.state_dict()), first.ema.state_dict()
    with hndl_lifecycle.BACKEND.open(config, runtime=fresh_runtime(config)) as second:
        second.restore(packet)
        assert second.step() == expected_metrics
        assert equal(second.game.state_dict(), expected_game)
        assert equal(second.ema.state_dict(), expected_ema)
        assert native_io.frozen_digest(second.game.backend.runtime) == frozen
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("corruption", ["nonfinite", "cohort"])
def test_checkpoint_rejection_is_atomic(configured, corruption):
    runtime, config = configured
    with hndl_lifecycle.BACKEND.open(config, runtime=runtime) as session:
        session.step()
        packet = session.checkpoint()
        before = deepcopy(session.game.state_dict())
        if corruption == "nonfinite":
            next(iter(packet["payload"]["game"]["network"].values())).fill_(float("nan"))
        else:
            packet["cohort"] = "historical-core081"
        packet["sha256"] = _hash({key: value for key, value in packet.items() if key != "sha256"})
        with pytest.raises(CheckpointError):
            session.restore(packet)
        assert equal(before, session.game.state_dict())
        runtime.cfg_adapter_branch = "both"
        with pytest.raises(UnsupportedOperation, match="admission"):
            session.step()
        assert equal(before, session.game.state_dict())


def test_observation_and_failed_prepare_restore_borrowed_native_state(configured, monkeypatch):
    runtime, config = configured
    cache_ref, branch = runtime.cache, runtime.cfg_adapter_branch
    flags = [(parameter, parameter.requires_grad) for parameter in runtime.model.parameters()]
    with hndl_lifecycle.BACKEND.open(config, runtime=runtime) as session:
        admitted_cache = runtime.cache
        tokens_ref, tokens = runtime.token_counts, dict(runtime.token_counts)
        state = torch.get_rng_state().clone()
        image = session.render(prompt="new HNDL fixture prompt", strength=-.5, seed=13, steps=1)
        assert image.size == (32, 32)
        assert runtime.cache is admitted_cache and runtime.token_counts is tokens_ref
        assert runtime.token_counts == tokens and torch.equal(torch.get_rng_state(), state)
    assert runtime.cache is cache_ref and runtime.cfg_adapter_branch == branch
    assert "encode" not in vars(runtime)
    assert all(parameter.requires_grad == flag for parameter, flag in flags)
    fresh = fresh_runtime(config)
    original_cache, original_branch = fresh.cache, fresh.cfg_adapter_branch

    def fail(*args, **kwargs):
        raise RuntimeError("teacher parity probe")

    monkeypatch.setattr(hndl_lifecycle.campaign, "teacher_parity", fail)
    with pytest.raises(RuntimeError, match="parity probe"):
        hndl_lifecycle.BACKEND.open(config, runtime=fresh)
    assert fresh.cache is original_cache and fresh.cfg_adapter_branch == original_branch
    assert "encode" not in vars(fresh)


def test_config_and_source_rejection_precede_native_allocation(configured, monkeypatch):
    runtime, config = configured

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid config reached native allocation")

    monkeypatch.setattr(hndl, "load_native", forbidden)
    invalid = dict(config.options, base_precision="bfloat16", device="cuda:1")
    with pytest.raises(UnsupportedOperation, match="precision"):
        hndl_lifecycle.BACKEND.open(SessionConfig("supra_hndl", "invalid", **invalid))
    import json
    lock = json.loads(hndl.LOCK.read_text())
    lock["source_sha256"]["particle_sliders/pg09.py"] = "0" * 64
    path = Path(config.options["cache"]).with_name("corrupt-core.lock.json")
    path.write_text(json.dumps(lock))
    monkeypatch.setattr(hndl, "LOCK", path)
    with pytest.raises(RuntimeError, match="source differs"):
        hndl_lifecycle.BACKEND.open(config, runtime=runtime)


def test_changed_frozen_model_rejects_without_adapter_mutation(configured):
    runtime, config = configured
    with torch.no_grad():
        next(parameter for parameter in runtime.model.parameters() if not parameter.requires_grad).add_(1)
    before = deepcopy(runtime.model.state_dict())
    with pytest.raises(ValueError, match="base/context"):
        hndl_lifecycle.BACKEND.open(config, runtime=runtime)
    assert equal(runtime.model.state_dict(), before)


@pytest.mark.parametrize("native_request", [{"cfg": -1}, {"steps": 0}, {"output_type": "latent"}, {"width": 512}])
def test_invalid_native_request_rejects_in_preflight(configured, native_request):
    runtime, config = configured

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid request reached factory construction")

    backend = replace(hndl_lifecycle.BACKEND, prepare=forbidden)
    with pytest.raises((ValueError, UnsupportedOperation)):
        backend.open(config, runtime=runtime, operation="render", native_request=native_request)


@pytest.mark.parametrize("selection", ["raw", "ema"])
def test_native_only_inference_in_fresh_process_without_training_assets(configured, tmp_path, selection):
    import json
    import os
    import subprocess
    import sys
    from .inference_worker import backend
    from supra.hndl_inference import PROTOCOL
    runtime, config = configured
    with hndl_lifecycle.BACKEND.open(config, runtime=runtime) as session:
        session.step()
        session.binding.inference = backend().specification(session.game.backend,
            model_identity=hndl.model_identity(runtime), protocol=PROTOCOL,
            configuration=dict(rank=runtime.rank))
        packet = session.export_inference(selection=selection)
    for filename in (config.options["cache"], config.options["contexts"],
                     str(Path(config.options["contexts"]).with_suffix(".json"))):
        Path(filename).unlink()
    path = tmp_path / "native-only.pt"
    torch.save(packet, path)
    result = subprocess.run([sys.executable, "-I", str(Path(__file__).with_name("inference_worker.py")), str(path)],
        cwd=tmp_path, env=dict(os.environ, PYTHONPATH="", CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1"),
        capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == dict(size=[32, 32], selection=selection, cuda_initialized=False)

"""Prepared conformance checks for a future isolated, offline CPU slot."""
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from particle_sliders.inference import InferenceBackend, open_inference
from particle_sliders.lifecycle import CheckpointError, SessionConfig, UnsupportedOperation
from particle_sliders.pg09 import _hash
from supra import cache, campaign, lifecycle, native_inference, native_io
from supra.shared import file_sha

from .test_thin_relocation import TinyRuntime, assets, equal  # noqa: F401 (fixture)


@pytest.fixture
def configured(assets, monkeypatch):
    runtime, config, path, contexts = assets
    rows = cache.SupraCampaignRows(path, cache_sha256=file_sha(path), config=config, contexts=runtime.cache)
    # This is explicitly a synthetic fixture scale, never the admitted native cache.
    monkeypatch.setattr(cache, "SUPRA_SCALE", rows.response_scale())
    options = dict(native_config=config, cache=str(path), cache_sha256=file_sha(path),
        contexts=str(contexts), contexts_sha256=file_sha(contexts),
        frozen_base_sha256=native_io.frozen_digest(runtime), device="cpu", rank=2,
        steps=4, batch=4, seed=7, critic_width=8, ema_decay=.9)
    return runtime, SessionConfig("supra", "synthetic-clean081-test-cohort", **options)


def test_exact_split_resume_with_metrics_ema_rng_and_coverage(configured):
    runtime, config = configured
    with lifecycle.BACKEND.open(config, runtime=runtime) as first:
        assert first.game.prior is None and first.game.batch_size == 4
        first.step()
        packet = first.checkpoint()
        expected_metrics = first.step()
        expected_game = deepcopy(first.game.state_dict())
        expected_ema = first.ema.state_dict()
        frozen = native_io.frozen_digest(runtime)
    fresh = TinyRuntime(config.options["native_config"])
    with lifecycle.BACKEND.open(config, runtime=fresh) as second:
        second.restore(packet)
        assert second.step() == expected_metrics
        assert equal(second.game.state_dict(), expected_game)
        assert equal(second.ema.state_dict(), expected_ema)
        assert native_io.frozen_digest(fresh) == frozen


def test_native_identity_digest_keeps_the_legacy_product_convention(configured):
    from supra.pg09 import frozen_digest
    runtime, _ = configured
    assert native_io.frozen_digest(runtime) == frozen_digest(runtime)


def test_corrupt_restore_and_native_admission_reject_without_update(configured):
    runtime, config = configured
    with lifecycle.BACKEND.open(config, runtime=runtime) as session:
        session.step()
        before = deepcopy(session.game.state_dict())
        packet = session.checkpoint()
        key = next(iter(packet["payload"]["game"]["network"]))
        packet["payload"]["game"]["network"][key].fill_(float("nan"))
        packet["sha256"] = _hash({key: value for key, value in packet.items() if key != "sha256"})
        with pytest.raises(CheckpointError):
            session.restore(packet)
        assert equal(before, session.game.state_dict())
        runtime.cfg_adapter_branch = "both"
        with pytest.raises(UnsupportedOperation, match="admission"):
            session.step()
        assert equal(before, session.game.state_dict())
        runtime.cfg_adapter_branch = "conditional_only"
        assert not session.closed


def test_observation_preserves_native_cache_modes_scales_rng_on_success_and_failure(configured, monkeypatch):
    runtime, config = configured
    with lifecycle.BACKEND.open(config, runtime=runtime) as session:
        cache_ref, tokens_ref = runtime.cache, runtime.token_counts
        tokens = dict(tokens_ref)
        modes = [(module, module.training) for module in runtime.model.modules()]
        multipliers = [module.multiplier for module in runtime.loras]
        state = torch.get_rng_state().clone()
        session.render(prompt="a separately encoded new fixture prompt", strength=-.5, seed=13, steps=1)
        assert runtime.cache is cache_ref and runtime.token_counts is tokens_ref
        assert dict(tokens_ref) == tokens
        assert [module.multiplier for module in runtime.loras] == multipliers
        assert all(module.training == mode for module, mode in modes)
        assert torch.equal(state, torch.get_rng_state())
        def fail(*args, **kwargs):
            runtime.encode("another new fixture prompt before failure")
            raise RuntimeError("native decode probe")
        monkeypatch.setattr(runtime.vae, "decode", fail)
        with pytest.raises(RuntimeError, match="decode probe"):
            session.render(prompt="failure prompt", strength=.25, seed=13, steps=1)
        assert runtime.cache is cache_ref and runtime.token_counts is tokens_ref and dict(tokens_ref) == tokens
        assert torch.equal(state, torch.get_rng_state())


def test_failed_preparation_restores_borrowed_runtime(configured, monkeypatch):
    runtime, config = configured
    cache_ref, branch = runtime.cache, runtime.cfg_adapter_branch
    modes = [(module, module.training) for module in runtime.model.modules()]
    flags = [(parameter, parameter.requires_grad) for parameter in runtime.model.parameters()]
    def fail(*args, **kwargs):
        raise RuntimeError("teacher parity probe")
    monkeypatch.setattr(campaign, "teacher_parity", fail)
    with pytest.raises(RuntimeError, match="parity probe"):
        lifecycle.BACKEND.open(config, runtime=runtime)
    assert runtime.cache is cache_ref and runtime.cfg_adapter_branch == branch
    assert "encode" not in vars(runtime)
    assert all(module.training == mode for module, mode in modes)
    assert all(parameter.requires_grad == grad for parameter, grad in flags)


@pytest.mark.parametrize("native", [{"cfg": -1}, {"steps": 0}, {"output_type": "latent"}, {"width": 512}])
def test_native_preflight_rejects_before_factory_allocation(configured, monkeypatch, native):
    runtime, config = configured
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid native request reached native construction")
    monkeypatch.setattr(native_io, "load_native", forbidden)
    with pytest.raises((ValueError, UnsupportedOperation)):
        lifecycle.BACKEND.open(config, runtime=runtime, operation="render", native_request=native)


def fixture_validate(options, spec):
    """An explicit test-only CPU factory profile; production CUDA gates stay intact."""
    assert options["device"] == "cpu"
    assert options["frozen_base_sha256"] == spec.identity["model"]["frozen_parameter_sha256"]
    native_io.verify_sources()


@pytest.mark.parametrize("selection", ["raw", "ema"])
def test_native_only_export_load_without_training_files_or_constructors(configured, monkeypatch, selection):
    runtime, config = configured
    fixture_backend = InferenceBackend("supra", native_inference.CAPABILITIES,
        native_inference.prepare, fixture_validate, response=False, preflight=native_inference.preflight)
    with lifecycle.BACKEND.open(config, runtime=runtime) as session:
        session.step()
        session.binding.inference = fixture_backend.specification(session.game.backend,
            model_identity=native_io.model_identity(runtime), protocol=native_inference.PROTOCOL,
            configuration=dict(rank=2))
        packet = session.export_inference(selection=selection)
    for file in (config.options["cache"], config.options["contexts"],
                 str(Path(config.options["contexts"]).with_suffix(".json"))):
        Path(file).unlink()
    from particle_sliders.pg09 import ConditionalPG09
    def forbidden(*args, **kwargs):
        raise AssertionError("native-only inference touched training construction or data")
    monkeypatch.setattr(ConditionalPG09, "__init__", forbidden)
    monkeypatch.setattr(campaign, "build_supra_campaign", forbidden)
    monkeypatch.setattr(cache, "load_supra_contexts", forbidden)
    monkeypatch.setattr(lifecycle, "prepare", forbidden)
    monkeypatch.setattr(native_io, "load_native", lambda options: TinyRuntime())
    options = dict(device="cpu", frozen_base_sha256=config.options["frozen_base_sha256"])
    with open_inference(packet, backend=fixture_backend, options=options) as inference:
        assert inference.selection == selection
        assert equal(inference.binding.network.state_dict(), packet["network"])
        assert inference.render(prompt="new native-only fixture prompt", strength=.5, seed=9, steps=1).size == (32, 32)
        assert not hasattr(inference, "game")
        with pytest.raises(UnsupportedOperation):
            inference.step()

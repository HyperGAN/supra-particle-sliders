"""Final immutable core pin: archived-cohort refusal and installed lifecycle."""
from copy import deepcopy
from importlib.metadata import version
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

if version("particle-sliders-core") != "0.12.0":
    pytest.skip("requires the modern core environment", allow_module_level=True)

import torch
from particle_sliders.inference import open_inference
from particle_sliders.lifecycle import CheckpointError, UnsupportedOperation
from supra import hndl, hndl_lifecycle
from .fixtures import assets, equal  # noqa: F401
from .test_lifecycle import configured, fresh_runtime  # noqa: F401
from .inference_worker import backend, load_fixture
from supra.hndl_inference import PROTOCOL


@pytest.fixture
def old_previews():
    value = os.environ.get("SUPRA_OLD_PREVIEW_DIR")
    if not value:
        pytest.skip("supply the archived genuine dev1 preview packets")
    root = Path(value)
    assert all((root / name).is_file() for name in (
        "old-preview-training.pt", "old-preview-raw.pt", "old-preview-ema.pt"))
    return root


def test_old_training_preview_is_refused_without_state_mutation(configured, old_previews):
    runtime, config = configured
    packet = torch.load(old_previews / "old-preview-training.pt", map_location="cpu", weights_only=True)
    with hndl_lifecycle.BACKEND.open(config, runtime=runtime) as session:
        before = deepcopy(session.game.state_dict()), session.ema.state_dict(), torch.get_rng_state().clone()
        with pytest.raises(CheckpointError, match="source|cohort|rebinding"):
            session.restore(packet)
        assert equal(before[0], session.game.state_dict())
        assert equal(before[1], session.ema.state_dict())
        assert torch.equal(before[2], torch.get_rng_state()) and not session.closed


@pytest.mark.parametrize("selection", ["raw", "ema"])
def test_old_native_preview_is_refused(old_previews, monkeypatch, selection):
    packet = torch.load(old_previews / ("old-preview-" + selection + ".pt"), map_location="cpu", weights_only=True)
    monkeypatch.setattr(hndl, "load_native", load_fixture)
    rng = torch.get_rng_state().clone()
    with pytest.raises((CheckpointError, ValueError), match="source|identity|cohort|factory|differs"):
        open_inference(packet, backend=backend(), options=dict(device="cpu",
            frozen_base_sha256=packet["specification"]["identity"]["model"]["frozen_parameter_sha256"]))
    assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize("selection", ["raw", "ema"])
def test_off_raw_and_ema_inference(configured, monkeypatch, selection):
    runtime, config = configured
    with hndl_lifecycle.BACKEND.open(config, runtime=runtime) as session:
        off = session.render(prompt="final pin fixture", strength=0., seed=9, steps=1)
        session.step()
        session.binding.inference = backend().specification(session.game.backend,
            model_identity=hndl.model_identity(runtime), protocol=PROTOCOL, configuration=dict(rank=runtime.rank))
        packet = session.export_inference(selection=selection)
    monkeypatch.setattr(hndl, "load_native", load_fixture)
    with open_inference(packet, backend=backend(), options=dict(device="cpu",
            frozen_base_sha256=config.options["frozen_base_sha256"])) as inference:
        assert equal(inference.binding.network.state_dict(), packet["network"])
        assert inference.render(prompt="final pin fixture", strength=0., seed=9, steps=1).tobytes() == off.tobytes()
        for strength in (.25, 1.):
            assert inference.render(prompt="final pin fixture", strength=strength, seed=9, steps=1).size == (32, 32)


def test_admission_refusal_can_retry_exactly(configured):
    runtime, config = configured
    with hndl_lifecycle.BACKEND.open(config, runtime=runtime) as session:
        before = deepcopy(session.game.state_dict())
        runtime.cfg_adapter_branch = "both"
        with pytest.raises(UnsupportedOperation, match="admission"):
            session.step()
        assert equal(before, session.game.state_dict()) and not session.closed
        runtime.cfg_adapter_branch = "conditional_only"
        actual = session.step()
        state = deepcopy(session.game.state_dict())
    with hndl_lifecycle.BACKEND.open(config, runtime=fresh_runtime(config)) as expected:
        assert expected.step() == actual
        assert equal(expected.game.state_dict(), state)


def test_failed_step_closes_and_can_reopen_from_checkpoint(configured, monkeypatch):
    runtime, config = configured
    with hndl_lifecycle.BACKEND.open(config, runtime=runtime) as reference:
        reference.step()
        packet = reference.checkpoint()
        metrics = reference.step()
        state, ema = deepcopy(reference.game.state_dict()), reference.ema.state_dict()
    failed_runtime = fresh_runtime(config)
    failed = hndl_lifecycle.BACKEND.open(config, runtime=failed_runtime)
    failed.restore(packet)
    original_flags = [(parameter, parameter.requires_grad) for parameter in failed_runtime.model.parameters()]

    def fail(*args, **kwargs):
        raise RuntimeError("final pin critic failure probe")

    monkeypatch.setattr(failed.game.critic, "forward", fail)
    with pytest.raises(RuntimeError, match="critic failure probe"):
        failed.step()
    assert failed.closed and all(parameter.requires_grad == flag for parameter, flag in original_flags)
    with hndl_lifecycle.BACKEND.open(config, runtime=fresh_runtime(config)) as retry:
        retry.restore(packet)
        assert retry.step() == metrics
        assert equal(retry.game.state_dict(), state) and equal(retry.ema.state_dict(), ema)


@pytest.mark.parametrize("change", ["license", "extra_data"])
def test_recursive_data_closure_is_enforced(tmp_path, change):
    import particle_sliders
    import concept_slider_core
    base = Path(particle_sliders.__file__).parent.parent
    for name in ("particle_sliders", "concept_slider_core"):
        shutil.copytree(base / name, tmp_path / name, ignore=shutil.ignore_patterns("__pycache__"))
    path = tmp_path / "particle_sliders" / ("LICENSE" if change == "license" else "probe/nested-data.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(path.read_text() + "\nchanged fixture\n" if path.is_file() else "{}")
    program = "import sys;sys.path.insert(0,sys.argv[1]);from supra import hndl;hndl.verify_sources()"
    result = subprocess.run([sys.executable, "-I", "-c", program, str(tmp_path)],
        capture_output=True, text=True, timeout=60)
    assert result.returncode != 0 and "source" in result.stderr and ("receipt" in result.stderr or "differs" in result.stderr)

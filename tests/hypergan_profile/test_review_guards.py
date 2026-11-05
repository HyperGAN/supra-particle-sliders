"""Exercise the shipped candidate validation and transactional ownership guards."""
from copy import deepcopy
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest
import importlib.util
if (importlib.util.find_spec("supra.hndl") is None
        or version("supra-particle-sliders") == "0.3.1"):
    pytest.skip("separate HyperGAN/HNDL wheel required", allow_module_level=True)

if version("particle-sliders-core") != "0.13.0":
    pytest.skip("requires the distinct core0.13 HyperGAN/HNDL environment", allow_module_level=True)

import torch

from supra import hndl, hndl_inference, hndl_lifecycle, native_model
from .test_lifecycle import configured, fresh_runtime  # noqa: F401
from .fixtures import assets, equal  # noqa: F401


def test_actual_inference_validator_accepts_cuda_contract_without_allocating():
    spec = SimpleNamespace(configuration={"rank": 2}, identity={"model": {"frozen_parameter_sha256": "a" * 64}})
    hndl_inference.validate(dict(device="cuda:1", frozen_base_sha256="a" * 64), spec)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("device", ["", "cpu", "cuda", "cuda:-1", "cuda:bad", 1])
def test_shipped_inference_validator_rejects_invalid_devices(device):
    spec = SimpleNamespace(configuration={"rank": 2}, identity={"model": {"frozen_parameter_sha256": "a" * 64}})
    with pytest.raises(ValueError, match="cuda:N"):
        hndl_inference.validate(dict(device=device, frozen_base_sha256="a" * 64), spec)


@pytest.mark.parametrize("rank", [0, -1, 257, True, 2.5, "2"])
def test_shipped_inference_validator_rejects_invalid_ranks(rank):
    spec = SimpleNamespace(configuration={"rank": rank}, identity={"model": {"frozen_parameter_sha256": "a" * 64}})
    with pytest.raises(ValueError, match="rank"):
        hndl_inference.validate(dict(device="cuda:1", frozen_base_sha256="a" * 64), spec)


@pytest.mark.parametrize("options", [{"device": "cuda:1", "base_precision": "bfloat16", "frozen_base_sha256": "a" * 64},
                                    {"device": "cuda:1", "frozen_base_sha256": "b" * 64},
                                    {"device": "cuda:1", "cache": "unused", "frozen_base_sha256": "a" * 64}])
def test_shipped_inference_validator_rejects_precision_base_and_training_assets(options):
    spec = SimpleNamespace(configuration={"rank": 2}, identity={"model": {"frozen_parameter_sha256": "a" * 64}})
    with pytest.raises(ValueError):
        hndl_inference.validate(options, spec)


@pytest.mark.parametrize("contents", [None, "{broken", "[]"])
def test_context_receipt_errors_are_value_errors_before_allocation(configured, contents):
    runtime, config = configured
    path = Path(config.options["contexts"]).with_suffix(".json")
    if contents is None:
        path.unlink()
    else:
        path.write_text(contents)
    with pytest.raises(ValueError, match="context receipt"):
        hndl_lifecycle.validate(config.options)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("decay", [-.1, 1., 1.1, float("nan"), float("inf"), True, False, "0.9"])
def test_ema_decay_rejects_invalid_values(configured, decay):
    _, config = configured
    with pytest.raises(ValueError, match="EMA decay"):
        hndl_lifecycle.validate(dict(config.options, ema_decay=decay))


@pytest.mark.parametrize("decay", [None, 0., .9, .999999])
def test_ema_decay_accepts_supported_bounds(configured, decay):
    _, config = configured
    hndl_lifecycle.validate(dict(config.options, ema_decay=decay))


def test_native_lora_modes_flags_weights_and_scales_are_preserved():
    original = native_model.LoRALinear(torch.nn.Linear(3, 4), rank=2, alpha=3)
    original.eval()
    original.up.train()
    original.down.weight.requires_grad_(False)
    original.multiplier = -.5
    bound = hndl.HNDLLoRALinear(original)
    assert bound.base is original.base
    assert bound.training == original.training
    assert bound.down.training == original.down.training and bound.up.training == original.up.training
    assert bound.down.weight.requires_grad == original.down.weight.requires_grad
    assert bound.up.weight.requires_grad == original.up.weight.requires_grad
    assert bound.scale == original.scale and bound.multiplier == original.multiplier
    assert equal(bound.state_dict(), original.state_dict())


def test_partial_native_attachment_rolls_back_model_and_list_identity():
    class FailSecond(torch.nn.Sequential):
        armed = False

        def __setitem__(self, index, module):
            super().__setitem__(index, module)
            if self.armed and index == 1:
                raise RuntimeError("second attachment failed after mutation")

    model = FailSecond(*(native_model.LoRALinear(torch.nn.Linear(3, 3), rank=2, alpha=2) for _ in range(2)))
    originals = list(model)
    runtime = SimpleNamespace(model=model, loras=originals)
    state = deepcopy(model.state_dict())
    model.armed = True
    with pytest.raises(RuntimeError, match="second attachment"):
        hndl.bind_native_loras(runtime)
    assert runtime.loras is originals and list(model) == originals
    assert equal(model.state_dict(), state)


@pytest.mark.parametrize("field", ["version", "pg_version", "hndl_version", "hypergan_commit", "schema", "source_commit", "namespaces"])
def test_changed_receipt_runtime_versions_are_enforced(tmp_path, monkeypatch, field):
    receipt = json.loads(hndl.LOCK.read_text())
    receipt[field] = "invalid"
    path = tmp_path / "wrong-version.json"
    path.write_text(json.dumps(receipt))
    monkeypatch.setattr(hndl, "LOCK", path)
    with pytest.raises(ValueError, match="receipt"):
        hndl.verify_sources()


def test_good_checkout_override_cannot_hide_tampered_imported_core(tmp_path):
    import particle_sliders
    import concept_slider_core
    root = Path(particle_sliders.__file__).parent
    shutil.copytree(root, tmp_path / "particle_sliders", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(Path(concept_slider_core.__file__).parent, tmp_path / "concept_slider_core",
                    ignore=shutil.ignore_patterns("__pycache__"))
    path = tmp_path / "particle_sliders/pg09.py"
    path.write_text(path.read_text() + "\n# deliberate source tamper fixture\n")
    program = 'import sys\nfor name in tuple(sys.modules):\n    if name == "particle_sliders" or name.startswith("particle_sliders.") or name == "concept_slider_core":\n        del sys.modules[name]\nsys.path.insert(0, sys.argv[1])\nfrom supra import hndl; hndl.verify_sources()'
    result = subprocess.run([sys.executable, "-I", "-c", program, str(tmp_path)],
        env=dict(os.environ, PARTICLE_SLIDERS_ROOT=str(root), CUDA_VISIBLE_DEVICES=""),
        capture_output=True, text=True, timeout=60)
    assert result.returncode != 0 and "Imported shared particle-sliders source differs" in result.stderr

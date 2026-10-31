"""Explicit CPU inference test profile, also executable in a fresh process."""
import importlib.util
import json
from pathlib import Path
import sys

from particle_sliders.inference import InferenceBackend
from supra import hndl, hndl_inference


def fixture_validate(options, spec):
    assert options["device"] == "cpu"
    assert options["frozen_base_sha256"] == spec.identity["model"]["frozen_parameter_sha256"]
    hndl.verify_sources()


def backend():
    return InferenceBackend("supra_hndl", hndl_inference.CAPABILITIES,
        hndl_inference.prepare, fixture_validate, response=False, preflight=hndl_inference.preflight)


def load_fixture(options):
    name = "supra_hndl_native_fixture"
    if name not in sys.modules:
        source = Path(__file__).resolve().parents[1] / "core081/test_thin_relocation.py"
        spec = importlib.util.spec_from_file_location(name, source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    runtime = sys.modules[name].TinyRuntime()
    hndl.bind_native_loras(runtime)
    return runtime


if __name__ == "__main__":
    import torch
    from particle_sliders.inference import open_inference
    from particle_sliders.pg09 import ConditionalPG09
    from supra import cache, campaign, hndl_lifecycle

    def forbidden(*args, **kwargs):
        raise AssertionError("native-only inference constructed training state or read training assets")

    packet = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
    ConditionalPG09.__init__ = forbidden
    torch.optim.Optimizer.__init__ = forbidden
    campaign.build_supra_campaign = cache.load_supra_contexts = hndl_lifecycle.prepare = forbidden
    hndl.load_native = load_fixture
    options = dict(device="cpu", frozen_base_sha256=packet["specification"]["identity"]["model"]["frozen_parameter_sha256"])
    with open_inference(packet, backend=backend(), options=options) as inference:
        image = inference.render(prompt="native-only new fixture prompt", strength=.5, seed=9, steps=1)
        assert not hasattr(inference, "game")
        print(json.dumps(dict(size=image.size, selection=inference.selection,
                              cuda_initialized=torch.cuda.is_initialized())))

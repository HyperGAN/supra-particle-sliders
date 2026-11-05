"""Real HyperGAN/HNDL CPU graphs; no pretrained loader, CUDA or mock builder."""
from copy import deepcopy
from importlib.metadata import version

import pytest
import importlib.util
if (importlib.util.find_spec("supra.hndl") is None
        or version("supra-particle-sliders") == "0.3.1"):
    pytest.skip("separate HyperGAN/HNDL wheel required", allow_module_level=True)

if version("particle-sliders-core") != "0.13.0":
    pytest.skip("requires the distinct core0.13 HyperGAN/HNDL environment", allow_module_level=True)

import torch

from supra import hndl, native_model

@pytest.mark.parametrize("strength", [0., -.5, .25, 1.])
def test_real_hndl_lora_matches_native_forward_gradients_and_state(strength):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(8)
        native = native_model.LoRALinear(torch.nn.Linear(8, 5), rank=2, alpha=3)
        torch.nn.init.normal_(native.up.weight, std=.1)
        state = torch.get_rng_state().clone()
        bound = hndl.HNDLLoRALinear(deepcopy(native))
        assert torch.equal(torch.get_rng_state(), state)
        value = torch.randn(2, 3, 8)
    native.multiplier = bound.multiplier = strength
    left = value.detach().requires_grad_()
    right = value.detach().requires_grad_()
    expected, actual = native(left), bound(right)
    assert torch.equal(actual, expected)
    assert native.state_dict().keys() == bound.state_dict().keys()
    actual.square().mean().backward()
    expected.square().mean().backward()
    assert torch.equal(left.grad, right.grad)
    for (name, parameter), (other, current) in zip(native.named_parameters(), bound.named_parameters()):
        assert name == other and parameter.requires_grad == current.requires_grad
        assert parameter.grad is None and current.grad is None or torch.equal(parameter.grad, current.grad)
    assert "5f368a3770cf5bdae0448995c9f76e29af8049f7" in bound.extra_repr()
    assert not torch.cuda.is_initialized()


def test_hndl_spatial_critic_matches_native_values_and_response_gradients():
    from particle_sliders.fibo_hndl import HNDLFieldCritic
    from particle_sliders.field_critic import FieldCritic
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2)
        native = FieldCritic(4, 7, 8)
        bound = HNDLFieldCritic(4, 7, (32, 32), 8)
        for name, layer in (("field0", native.field[0]), ("field2", native.field[2]),
                            ("condition", native.condition), ("score0", native.score[0]),
                            ("score2", native.score[2])):
            bound.graph[name].load_state_dict(layer.state_dict())
        response, condition = torch.randn(2, 4, 32, 32), torch.randn(2, 7)
    left, right = response.requires_grad_(), response.detach().clone().requires_grad_()
    expected, actual = native(left, condition), bound(right, condition)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    left_grad = torch.autograd.grad(expected.sum(), left, create_graph=True)[0]
    right_grad = torch.autograd.grad(actual.sum(), right, create_graph=True)[0]
    torch.testing.assert_close(left_grad, right_grad, rtol=1e-5, atol=1e-8)
    # The shared gradient penalty differentiates through this response gradient.
    # Compare the full input and critic-parameter VJPs without inventing a loss.
    native_parameters = [parameter for layer in (native.field[0], native.field[2],
        native.condition, native.score[0], native.score[2]) for parameter in layer.parameters()]
    bound_parameters = [parameter for name in ("field0", "field2", "condition", "score0", "score2")
                        for parameter in bound.graph[name].parameters()]
    a = torch.autograd.grad(left_grad.square().sum(), [left, *native_parameters], allow_unused=True)
    b = torch.autograd.grad(right_grad.square().sum(), [right, *bound_parameters], allow_unused=True)
    for expected, actual in zip(a, b):
        assert (expected is None) == (actual is None)
        if expected is not None:
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-8)


def test_tampered_hypergan_builder_is_rejected_before_native_attachment(tmp_path, monkeypatch):
    import inspect
    import json
    from pathlib import Path
    from types import SimpleNamespace
    from hypergan.hndl_networks import build_network
    from particle_sliders import fibo_hndl
    source = Path(inspect.getsourcefile(build_network)).parent
    manifest = json.loads((fibo_hndl.ROOT / "fibo_hndl_010_sources.json").read_text())
    for name in (*manifest["hypergan"], "_build_provenance.json"):
        (tmp_path / name).write_bytes((source / name).read_bytes())
    target = tmp_path / "hndl_networks.py"
    target.write_text(target.read_text() + "\n# deliberate tamper fixture\n")
    monkeypatch.setattr(fibo_hndl.inspect, "getsourcefile", lambda function: str(target))
    model = torch.nn.Sequential(native_model.LoRALinear(torch.nn.Linear(8, 5), rank=2, alpha=2))
    runtime = SimpleNamespace(model=model, loras=[model[0]])
    original = model[0]
    with pytest.raises(ValueError, match="builder closure"):
        hndl.bind_native_loras(runtime)
    assert model[0] is original and runtime.loras == [original]

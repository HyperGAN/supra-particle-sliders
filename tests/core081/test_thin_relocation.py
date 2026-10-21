"""Prepared, unexecuted equivalence checks; run only in an admitted CPU slot.

Set SUPRA_REFERENCE_CORE to the preserved fd89e1c core package source directory.
That path is read as reference data; it is never added to sys.path.
"""
import ast
from contextlib import nullcontext
from copy import deepcopy
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import types

import pytest
import torch

from supra import cache, campaign, native_model
from supra.runtime import SupraRuntime, TARGETS
from supra.shared import file_sha


@pytest.fixture
def reference():
    value = os.environ.get("SUPRA_REFERENCE_CORE")
    if not value:
        pytest.skip("required equivalence reference not supplied; this skip does not qualify adoption")
    root = Path(value)
    assert (root / "supra_model.py").is_file()
    provenance = json.loads(Path(native_model.__file__).with_name("thin-product-provenance.json").read_text())
    assert file_sha(root / "supra_model.py") == provenance["native_model"]["sha256"]
    assert file_sha(root / "pg09_remaining_campaigns.py") == provenance["selected_source_sha256"]
    return root, provenance


def selected(text, names):
    tree = ast.parse(text)
    return {node.name: node for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names}


def test_native_file_and_selected_definitions_are_unchanged(reference):
    root, provenance = reference
    assert Path(native_model.__file__).read_bytes() == (root / "supra_model.py").read_bytes()
    original = selected((root / "pg09_remaining_campaigns.py").read_text(), provenance["definitions"])
    for name, item in provenance["definitions"].items():
        current = selected(Path(cache.__file__).with_name(item["destination"]).read_text(), [name])[name]
        old = deepcopy(original[name])
        if name == "build_supra_campaign":
            for node in ast.walk(old):
                if isinstance(node, ast.ImportFrom) and node.module == "pg09" and node.level == 1:
                    node.module, node.level = "particle_sliders.pg09", 0
        assert ast.dump(old, include_attributes=False) == ast.dump(current, include_attributes=False), name
    for module in (cache, campaign):
        tree = ast.parse(Path(module.__file__).read_text())
        assert not any(isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in
                       {"ConditionalPG09", "EMA", "SliderSession", "build_final_form_campaign"}
                       for node in ast.walk(tree))


@pytest.mark.parametrize("strength", [0., -.5, .25, 1.])
def test_native_parameter_schema_forward_and_gradients_match_reference(reference, strength):
    root, _ = reference
    spec = importlib.util.spec_from_file_location("supra_original_native", root / "supra_model.py")
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    args = dict(latent_ch=4, d_model=16, depth=1, n_heads=2, ctx_dim=8,
                mlp_ratio=2., num_tokens=256, patch=2)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(71)
        left = original.SupraDiT(**args)
        old_loras = original.attach_supra_lora(left, rank=2, alpha=2, targets=TARGETS)
        for module in old_loras:
            torch.nn.init.normal_(module.up.weight, std=.01)
        right = native_model.SupraDiT(**args)
        new_loras = native_model.attach_supra_lora(right, rank=2, alpha=2, targets=TARGETS)
        right.load_state_dict(left.state_dict())
        latent, context = torch.randn(1, 4, 32, 32), torch.randn(1, 3, 8)
    assert [(name, tuple(p.shape), p.dtype, p.requires_grad) for name, p in left.named_parameters()] == [
        (name, tuple(p.shape), p.dtype, p.requires_grad) for name, p in right.named_parameters()]
    for module in old_loras + new_loras:
        module.multiplier = strength
    a = left(latent, torch.tensor([.5]), context, torch.ones(1, 3))
    b = right(latent, torch.tensor([.5]), context, torch.ones(1, 3))
    assert torch.equal(a, b)
    a.square().mean().backward()
    b.square().mean().backward()
    for (_, p), (_, q) in zip(left.named_parameters(), right.named_parameters()):
        assert p.grad is None and q.grad is None or torch.equal(p.grad, q.grad)


def original_provider(root):
    """Execute selected product rows only; never import the historical campaign."""
    names = ["tensor_identity", "required_supra_prompts", "SupraCampaignRows"]
    nodes = selected((root / "pg09_remaining_campaigns.py").read_text(), names)
    module = types.ModuleType("supra_original_provider")
    module.__file__ = str(root / "pg09_remaining_campaigns.py")
    from particle_sliders.pg09_backends import PreparedResponse, ResponseRow
    module.__dict__.update(deepcopy=deepcopy, hashlib=hashlib, json=json, math=__import__("math"),
        Path=Path, torch=torch, file_digest=file_sha, PreparedResponse=PreparedResponse, ResponseRow=ResponseRow)
    sys.modules[module.__name__] = module
    exec(compile(ast.Module(body=list(nodes.values()), type_ignores=[]), module.__file__, "exec"), module.__dict__)
    return module.SupraCampaignRows


def equal(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(equal(a, b) for a, b in zip(left, right))
    return left == right


class TinyVAE(torch.nn.Module):
    def decode(self, value):
        return types.SimpleNamespace(sample=value[:, :3])


class TinyRuntime(SupraRuntime):
    """Synthetic native geometry; never load weights or call CUDA."""
    def __init__(self, config=None):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(71)
            self.model = native_model.SupraDiT(latent_ch=4, d_model=16, depth=1, n_heads=2,
                ctx_dim=8, mlp_ratio=2., num_tokens=256, patch=2)
            self.loras = native_model.attach_supra_lora(self.model, rank=2, alpha=2, targets=TARGETS)
        self.device, self.rank, self.cfg_adapter_branch = torch.device("cpu"), 2, "both"
        self.cache, self.token_counts, self.stored_unconditional = {}, {}, True
        self.vae = TinyVAE()
        for prompt in cache.required_supra_prompts(config) if config else [""]:
            self.encode(prompt)

    def encode(self, prompt):
        if prompt not in self.cache:
            seed = int(hashlib.sha256(prompt.encode()).hexdigest()[:8], 16)
            generator = torch.Generator().manual_seed(seed)
            self.cache[prompt] = torch.randn(1, 3, 8, generator=generator), torch.ones(1, 3)
            self.token_counts[prompt] = len(prompt)
        return self.cache[prompt]


@pytest.fixture
def assets(tmp_path, monkeypatch):
    import yaml
    monkeypatch.setattr(torch, "autocast", lambda *args, **kwargs: nullcontext())
    config = yaml.safe_load((Path(__file__).resolve().parents[2] / "configs/supra/prompts-final-boss.yaml").read_text())
    runtime = TinyRuntime(config)
    from particle_sliders.pg09_backends import SupraRuntimeResponseAdapter, ResponseRow
    backend = SupraRuntimeResponseAdapter(runtime)
    records, holds = [], []
    with torch.no_grad():
        for row in config["rows"]:
            for role in ("neutral", "positive"):
                z = torch.randn(1, 4, 32, 32, generator=torch.Generator().manual_seed(91))
                prepared = backend.prepare(ResponseRow(row["neutral"], row["positive"], "", z,
                    timestep=torch.tensor(.5), guidance=3.))
                records.append(dict(row=row["name"], prompt=row["neutral"], context=role,
                    seed=91, t=.5, z=z, neutral=prepared.neutral, target=prepared.target))
        for prompt in config["preservation"]:
            z = torch.randn(1, 4, 32, 32, generator=torch.Generator().manual_seed(92))
            value = backend.prepare(ResponseRow(prompt, prompt, "", z,
                timestep=torch.tensor(.5), guidance=3.)).neutral
            holds.append(dict(prompt=prompt, seed=92, t=.5, z=z, neutral=value, target=value.clone()))
    path = tmp_path / "synthetic-native-cache.pt"
    torch.save(dict(records=records, holds=holds), path)
    contexts = tmp_path / "synthetic-contexts.safetensors"
    cache.export_live_supra_contexts(runtime, config, contexts,
        runtime_identity=dict(model_revision=runtime.MODEL_REV, text_revision=runtime.T5_REV))
    return runtime, config, path, contexts


def test_native_rows_conditions_identity_and_targets_match_reference(reference, assets):
    root, _ = reference
    runtime, config, path, _ = assets
    before = torch.get_rng_state().clone()
    kwargs = dict(cache_sha256=file_sha(path), config=config, contexts=runtime.cache)
    a, b = original_provider(root)(path, **kwargs), cache.SupraCampaignRows(path, **kwargs)
    assert a.identity() == b.identity() and a.routes == b.routes
    assert a.response_scale() == b.response_scale()
    assert torch.equal(a.conditions(range(len(a))), b.conditions(range(len(b))))
    for index in range(len(a)):
        assert equal(vars(a[index].row), vars(b[index].row))
        assert torch.equal(a[index].neutral, b[index].neutral)
        assert torch.equal(a[index].target, b[index].target)
    assert torch.equal(before, torch.get_rng_state())


def test_shared_critic_and_rng_initialization_stay_in_core(reference):
    root, _ = reference
    import particle_sliders.field_critic as core
    # The reviewed clean candidate removed one EOF blank line. Compare the
    # complete parsed definition, while the worker separately verifies every
    # installed core byte against the immutable canonical source map.
    assert ast.dump(ast.parse(Path(core.__file__).read_text()), include_attributes=False) == ast.dump(
        ast.parse((root / "field_critic.py").read_text()), include_attributes=False)
    assert campaign.CampaignFieldCritic is core.FieldCritic
    assert campaign._seed_initialization_rng is core.seed_initialization_rng
    assert inspect.getmodule(campaign.CampaignFieldCritic).__name__ == "particle_sliders.field_critic"

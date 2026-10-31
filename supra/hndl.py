"""Supra native LoRA binding to the shared verified HyperGAN/HNDL builder.

This is a distinct network/source cohort. The native DiT, CFG, sampler and
shared ParticleGAN game keep their existing owners; no GAN law is defined here.
"""
from copy import deepcopy
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import torch
from torch import nn

from . import native_model
from .shared import file_sha, verify_imported_sources

GRAPH = Path(__file__).with_name("supra_lora.hndl")
PROFILE = "supra-hypergan-hndl-v1"
LOCK = Path(__file__).with_name("core012-receipt.json")


def verify_sources():
    """Hash the actually imported core; source-root overrides are irrelevant."""
    receipt = verify_imported_sources(LOCK)
    for package, field in (("particle-sliders-core", "version"), ("particlegan", "pg_version"),
                           ("hndl", "hndl_version")):
        if version(package) != receipt[field]:
            raise ValueError("Installed " + package + " version contradicts the source receipt")
    from particle_sliders.pg09 import _released_sources
    from particle_sliders.fibo_hndl import verified_builder
    _released_sources()
    _, identity = verified_builder()
    if identity["hypergan_commit"] != receipt["hypergan_commit"]:
        raise ValueError("HyperGAN build provenance contradicts the source receipt")


def model_identity(runtime):
    from .native_io import model_identity as native_identity
    identity = native_identity(runtime)
    identity["core_receipt_sha256"] = file_sha(LOCK)
    identity["network_profile"] = PROFILE
    identity["product_sources"].update({name: file_sha(Path(__file__).with_name(name))
        for name in ("hndl.py", "supra_lora.hndl", "hndl_lifecycle.py", "hndl_inference.py")})
    return identity


class HNDLLoRALinear(nn.Module):
    """Native base linear plus a real HNDL down/up graph at signed strength."""
    def __init__(self, original, *, builder=None, identity=None):
        super().__init__()
        if not isinstance(original, native_model.LoRALinear):
            raise TypeError("An admitted native Supra LoRALinear is required")
        if builder is None:
            verify_sources()
            from particle_sliders.fibo_hndl import verified_builder
            builder, identity = verified_builder()
        source = GRAPH.read_text()
        # These temporary random initializers are replaced by the admitted
        # native weights; they must not consume the caller's RNG stream.
        with torch.random.fork_rng(devices=[]):
            graph = builder(source,
                input_shape={"value": ["B", 1, original.base.in_features]},
                output_shape=["B", 1, original.base.out_features],
                parameters={"rank": original.down.out_features,
                            "output_width": original.base.out_features})
        self.base = original.base
        self.down, self.up = graph["down"], graph["up"]
        # Register each trainable leaf exactly once under the native names.
        # The graph owns no buffers or other leaves: it executes these modules.
        if list(graph.buffers()) or {id(p) for p in graph.parameters()} != {
                id(p) for layer in (self.down, self.up) for p in layer.parameters()}:
            raise ValueError("Unexpected state in the Supra HNDL projection graph")
        object.__setattr__(self, "_graph", graph)
        self.down.to(original.down.weight)
        self.up.to(original.up.weight)
        self.down.load_state_dict(original.down.state_dict())
        self.up.load_state_dict(original.up.state_dict())
        self.training = original.training
        self.down.training, self.up.training = original.down.training, original.up.training
        self.down.weight.requires_grad_(original.down.weight.requires_grad)
        self.up.weight.requires_grad_(original.up.weight.requires_grad)
        self.scale, self.multiplier = original.scale, original.multiplier
        self._identity = dict(profile=PROFILE, runtime=deepcopy(identity),
            graph_sha256=hashlib.sha256(source.encode()).hexdigest(),
            plan_sha256=hashlib.sha256(graph.plan.to_json().encode()).hexdigest(),
            input_width=original.base.in_features, output_width=original.base.out_features,
            rank=original.down.out_features)

    def extra_repr(self):
        return json.dumps(self._identity, sort_keys=True)

    def forward(self, value):
        shape = value.shape
        delta = self._graph(value=value.reshape(-1, 1, shape[-1]))
        delta = delta.reshape(*shape[:-1], self.base.out_features)
        return self.base(value) + self.multiplier * self.scale * delta


def bind_native_loras(runtime):
    """Replace only the existing native LoRA wrappers, retaining base/weights.

    Requires a fresh runtime owned by the new cohort, before optimizer creation.
    Existing runtime/checkpoint cohorts must use their original environments.
    """
    from particle_sliders.fibo_hndl import verified_builder
    verify_sources()
    builder, identity = verified_builder()
    original_list = runtime.loras
    originals = list(original_list)
    found = [(name, module) for name, module in runtime.model.named_modules()
             if isinstance(module, native_model.LoRALinear)]
    if (not originals or {id(module) for _, module in found} != {id(module) for module in originals}
            or len(found) != len(originals)):
        raise ValueError("Runtime LoRA ownership differs from the native model")
    # Construct everything before changing the native module tree.
    replacements = [(name, HNDLLoRALinear(module, builder=builder, identity=identity))
                    for name, module in found]
    by_original = {id(module): replacement for (_, module), (_, replacement)
                   in zip(found, replacements)}
    targets = [(native_model._module_by_path(runtime.model, name), original, replacement)
               for (_, original), (name, replacement) in zip(found, replacements)]
    try:
        for (parent, attr), _, replacement in targets:
            if attr.isdigit():
                parent[int(attr)] = replacement
            else:
                setattr(parent, attr, replacement)
        runtime.loras = [by_original[id(module)] for module in originals]
    except BaseException as error:
        # Torch stores these native child modules in _modules. Restore all
        # targets directly, including a setter that mutated before raising.
        for (parent, attr), original, _ in reversed(targets):
            try:
                parent._modules[attr] = original
            except BaseException as cleanup_error:
                error.add_note("Supra binding rollback failed: " + str(cleanup_error))
        try:
            runtime.loras = original_list
        except BaseException as cleanup_error:
            error.add_note("Supra LoRA-list rollback failed: " + str(cleanup_error))
        raise
    return dict(profile=PROFILE, runtime=identity,
                graph_sha256=hashlib.sha256(GRAPH.read_bytes()).hexdigest())


def native_module():
    """Inject only the HNDL LoRA wrapper into the existing local native loader."""
    from types import SimpleNamespace

    def attach(root, rank, alpha, targets):
        loras = native_model.attach_supra_lora(root, rank, alpha, targets)
        runtime = SimpleNamespace(model=root, loras=loras)
        bind_native_loras(runtime)
        return runtime.loras

    return SimpleNamespace(SupraDiT=native_model.SupraDiT,
        load_supra_weights=native_model.load_supra_weights, attach_supra_lora=attach)


def load_native(options):
    from .runtime import SupraRuntime
    verify_sources()
    if options["device"] == "cpu":
        raise ValueError("Native Supra HNDL requires CUDA; CPU uses an explicit fixture profile")
    return SupraRuntime(options["device"], rank=options.get("rank", 16),
                        allow_hub=False, native_model=native_module())

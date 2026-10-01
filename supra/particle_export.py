"""Lean, clean-inference export of a served Supra particle adapter.

Frozen native weights, text encoders, critics and optimizer state are supplied
by the pinned runtime rather than copied into the adapter. This export serves
the selected clean routing law; it does not claim to replay training DV12.
"""
from copy import deepcopy
import json
import math
from pathlib import Path

import torch
from torch import nn
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from particlegan import RoutedCandidate, RoutedRows
from .particle_adapter import (
    GATED_PARTICLE_V3, LINEAR_MODULATED_V2, NONLINEAR_V1, SupraParticleHost, SupraParticleRouter,
    adapter_input_dims, adapter_sites, validate_particle_architecture,
)
from .runtime import MODEL_ID, MODEL_REV, T5_REV, VAE_REV


FORMAT = "supra_particlegan_clean_v1"
FORMAT_V2 = "supra_particlegan_clean_v2"
FORMAT_V3 = "supra_particlegan_clean_v3"
_FORMAT_ARCHITECTURES = {FORMAT: NONLINEAR_V1, FORMAT_V2: LINEAR_MODULATED_V2,
                       FORMAT_V3: GATED_PARTICLE_V3}


def _export_architecture(metadata, config):
    """Tensor shapes cannot distinguish the formulas; their versions must agree."""
    export_format = metadata.get("format")
    if export_format not in _FORMAT_ARCHITECTURES:
        raise ValueError("unsupported Supra particle export format")
    architecture = validate_particle_architecture(config.get("architecture", NONLINEAR_V1))
    expected = _FORMAT_ARCHITECTURES[export_format]
    if architecture != expected:
        raise ValueError("particle export architecture and format version differ")
    return architecture


def native_pins():
    lock = json.loads((Path(__file__).resolve().parents[1] / "backend.lock.json").read_text())
    return dict(model_id=MODEL_ID, model_revision=MODEL_REV,
                text_encoder_id="google/flan-t5-base", text_encoder_revision=T5_REV,
                vae_id="stabilityai/sd-vae-ft-mse", vae_revision=VAE_REV,
                backend_sha256=lock["sha256"])


def _branch_state(generator):
    state = {}
    for site, branch in zip(generator.sites, generator.particle_branches()):
        for name in ("down", "bridge", "up"):
            for key, value in getattr(branch, name).state_dict().items():
                state[f"generator.model.{site}.{name}.{key}"] = value
    return state


def _export_state(generator, router, table, log_mass):
    return {**_branch_state(generator),
            **{f"router.{key}": value for key, value in router.state_dict().items() if key != "log_mass"},
            "bank.table": table, "bank.log_mass": log_mass}


def export_particle_adapter(loop, path, *, extra_metadata=None):
    """Export the public policy's currently selected clean served model."""
    return export_served_adapter(loop.policy.served_model(), path, extra_metadata=extra_metadata)


def export_served_adapter(served, path, *, extra_metadata=None):
    """Copy only particle tensors from an independent public serving snapshot."""
    generator, router = served.generator, served.router
    if not isinstance(generator, SupraParticleHost) or not isinstance(router, SupraParticleRouter):
        raise TypeError("export requires a native Supra particle host and router")
    if generator.sites != router.sites or served.routing is None:
        raise ValueError("export requires matching explicit routed sites")
    architecture = validate_particle_architecture(generator.architecture)
    if any(branch.architecture != architecture for branch in generator.particle_branches()):
        raise ValueError("particle branch architecture differs from its host")
    candidate = served.routing.candidate_for(served.models, served.table, averaged=served.source == "averaged")
    state = {key: value.detach().cpu().contiguous().clone() for key, value in
             _export_state(generator, router, candidate.table, candidate.log_mass).items()}
    config = dict(rank=generator.rank, z_dim=generator.z_dim, cfg=generator.cfg,
                  num_particles=len(candidate.table), sites=list(generator.sites),
                  site_input_dims=deepcopy(router.site_input_dims), sampling="clean",
                  routed_geometry="mass_atoms_v1", served_source=served.source,
                  completed_steps=served.completed_steps,
                  extra={} if extra_metadata is None else deepcopy(extra_metadata))
    # Keep legacy metadata unchanged. New exports declare their formula because
    # the same tensor names and shapes are valid for both architectures.
    export_format = FORMAT
    if architecture != NONLINEAR_V1:
        export_format = {LINEAR_MODULATED_V2: FORMAT_V2, GATED_PARTICLE_V3: FORMAT_V3}[architecture]
        config["architecture"] = architecture
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(state, str(path), metadata={"format": export_format, "config": json.dumps(config, sort_keys=True),
                                          "native_pins": json.dumps(native_pins(), sort_keys=True)})
    return dict(path=str(path), bytes=path.stat().st_size, tensors=len(state),
                parameters=sum(value.numel() for value in state.values()), config=config)


class _EncodedCondition(nn.Module):
    """Explicit per-call text ownership; no cached training caption lookup."""

    def __init__(self, z, ctx, mask, uncond_ctx, uncond_mask, strength, cfg):
        super().__init__()
        self.z_shape = tuple(z.shape[1:])
        self.strength, self.cfg = float(strength), float(cfg)
        for name, value in (("ctx", ctx), ("mask", mask), ("uncond_ctx", uncond_ctx),
                            ("uncond_mask", uncond_mask)):
            self.register_buffer(name, value)


def _native_forward(models, context, candidate, routing):
    condition = models["conditioning"]
    z = context[:, :-1].reshape(-1, *condition.z_shape)
    return models["generator"].forward_routed(
        z, context[:, -1], condition.ctx, condition.mask, condition.uncond_ctx, condition.uncond_mask,
        candidate, routing, models["router"], strength=condition.strength, cfg=condition.cfg)


def _unused_features(*args):
    raise RuntimeError("a clean inference export has no learned critic or row controller")


class CleanParticleAdapter(nn.Module):
    """Independent full native host with the exported clean particle state."""

    def __init__(self, generator, router, table, metadata):
        super().__init__()
        self.generator, self.router = generator, router
        self.register_buffer("table", table.detach().clone())
        self.metadata = deepcopy(metadata)
        self.routing = RoutedRows(model_forward=_native_forward, features=_unused_features,
                                  sites=generator.sites)
        self.eval().requires_grad_(False)

    @torch.no_grad()
    def velocity(self, z, t, ctx, mask, uncond_ctx=None, uncond_mask=None, *, strength=1., cfg=None):
        cfg = self.generator.cfg if cfg is None else float(cfg)
        if not math.isfinite(float(strength)) or not math.isfinite(cfg):
            raise ValueError("strength and CFG must be finite")
        t = torch.as_tensor(t, dtype=torch.float32, device=z.device).reshape(-1)
        if len(t) == 1:
            t = t.expand(len(z))
        if len(t) != len(z):
            raise ValueError("time must be scalar or contain one value per context")
        condition = _EncodedCondition(z, ctx, mask, uncond_ctx, uncond_mask, strength, cfg)
        context = torch.cat((z.flatten(1), t[:, None]), dim=1)
        candidate = RoutedCandidate(self.table, self.router.log_mass, {"log_mass": self.router.log_mass},
                                    averaged=self.metadata["served_source"] == "averaged")
        return self.routing.forward({"generator": self.generator, "router": self.router,
                                     "conditioning": condition}, context, candidate)


def load_particle_adapter(base_model, path, *, device=None):
    """Strictly validate pins, sites, tensor names/shapes/dtypes before loading."""
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    if metadata.get("format") not in _FORMAT_ARCHITECTURES:
        raise ValueError("unsupported Supra particle export format")
    if json.loads(metadata.get("native_pins", "null")) != native_pins():
        raise ValueError("particle export native model/backend pins differ from this runtime")
    config = json.loads(metadata["config"])
    architecture = _export_architecture(metadata, config)
    if config.get("sampling") != "clean" or config.get("routed_geometry") != "mass_atoms_v1":
        raise ValueError("particle export must use the clean mass-atoms routing law")
    if config.get("served_source") not in ("fast", "averaged"):
        raise ValueError("invalid served source")
    for name in ("rank", "z_dim", "num_particles"):
        if type(config.get(name)) is not int or config[name] < 1:
            raise ValueError(f"invalid exported {name}")
    if not isinstance(config.get("cfg"), (int, float)) or not math.isfinite(config["cfg"]):
        raise ValueError("invalid exported CFG")
    sites = tuple(config["sites"])
    if sites != adapter_sites(base_model) or config["site_input_dims"] != adapter_input_dims(base_model, sites):
        raise ValueError("particle export must cover exactly the native base's ordered LoRA sites")
    generator = SupraParticleHost(base_model, rank=config["rank"], z_dim=config["z_dim"],
                                  cfg=config["cfg"], sites=sites, architecture=architecture)
    router = SupraParticleRouter(num_particles=config["num_particles"], z_dim=config["z_dim"],
                                 site_input_dims=adapter_input_dims(base_model, sites))
    table = torch.zeros(config["num_particles"], config["z_dim"])
    expected = _export_state(generator, router, table, router.log_mass)
    state = load_file(str(path), device="cpu")
    if set(state) != set(expected):
        raise ValueError("particle export tensor names differ from the complete adapter schema")
    for key, value in state.items():
        if value.shape != expected[key].shape or value.dtype != expected[key].dtype:
            raise ValueError(f"particle export tensor shape or dtype mismatch: {key}")
        if key == "bank.log_mass":
            valid = bool((torch.isfinite(value) | torch.isneginf(value)).all()) and bool(torch.isfinite(value).any())
        else:
            valid = bool(torch.isfinite(value).all())
        if not valid:
            raise ValueError(f"invalid particle export tensor values: {key}")
    # Load branch children strictly; the omitted frozen host parameters come
    # from the validated runtime and must never be populated from this file.
    for site, branch in zip(sites, generator.particle_branches()):
        for name in ("down", "bridge", "up"):
            prefix = f"generator.model.{site}.{name}."
            getattr(branch, name).load_state_dict({key.removeprefix(prefix): value for key, value in state.items()
                                                  if key.startswith(prefix)}, strict=True)
    router.load_state_dict({**{key.removeprefix("router."): value for key, value in state.items()
                               if key.startswith("router.")}, "log_mass": state["bank.log_mass"]}, strict=True)
    device = next(base_model.parameters()).device if device is None else torch.device(device)
    return CleanParticleAdapter(generator, router, state["bank.table"], config).to(device)

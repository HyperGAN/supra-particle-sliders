"""Clean particle exports with explicit student final-head arithmetic.

This separate format retains the shared-Up adapter tensor schema. Frozen host
weights are supplied by the pinned runtime; the final class is reconstructed
before strict learned-state loading, without reinitializing trained tensors.
"""
from copy import deepcopy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from particlegan import RoutedRows
from .particle_adapter import (GATED_PARTICLE_V3, SupraParticleHost,
                               SupraParticleRouter, adapter_input_dims, adapter_sites)
from .particle_export import CleanParticleAdapter, _export_state, native_pins
from .particle_final_precision import (
    BF16_FINAL, FP32_FINAL, PRECISION_KEY, assert_policy_precision,
    install_student_final_, precision_spec, validate_precision,
)
from .particle_training import raw_model_forward

FORMAT = "supra_particlegan_clean_final_precision_v1"


def _head_contract(generator):
    head = generator.model.final.linear
    return dict(weight_shape=list(head.weight.shape),
                bias_shape=None if head.bias is None else list(head.bias.shape),
                storage="torch.float32", frozen=True)


def _precision_class(generator):
    # Avoid a metadata-only declaration that silently relabels actual arithmetic.
    from .particle_final_precision import BF16StudentFinalLinear, FP32StudentFinalLinear
    if type(generator.model.final.linear) is BF16StudentFinalLinear:
        return BF16_FINAL
    if type(generator.model.final.linear) is FP32StudentFinalLinear:
        return FP32_FINAL
    raise ValueError("export requires an explicit student final precision class")


def _unused_features(*args):
    raise RuntimeError("clean precision export has no critic or training controller")


def precision_snapshot(loop, *, source="fast"):
    """Independent, explicitly selected public FAST/EMA owner snapshot.

Native settlement recommendations do not select this source implicitly. No
public served-weight swap or live owner mutation is needed for serialization.
"""
    if source not in ("fast", "averaged"):
        raise ValueError("export source must be fast or averaged")
    policy = loop.policy
    precision = validate_precision(loop.config[PRECISION_KEY])
    assert_policy_precision(policy, precision)
    averaged = source == "averaged"
    generator, router, encoder, table = deepcopy((
        policy.ema_G if averaged else policy.G,
        policy.ema_router if averaged else policy.router,
        policy.ema_encoder if averaged else policy.encoder,
        (policy.averaged_table if averaged else policy.table).detach()))
    models = dict(generator=generator, router=router, encoder=encoder)
    for owner in models.values():
        owner.eval().requires_grad_(False)
    table.requires_grad_(False)
    rows = RoutedRows(model_forward=raw_model_forward, features=_unused_features,
                      sites=generator.sites)
    return SimpleNamespace(generator=generator, router=router, encoder=encoder,
                           models=models, table=table, routing=rows, source=source,
                           completed_steps=policy.completed_steps, precision=precision)


def export_particle_adapter(loop, path, *, source="fast", extra_metadata=None):
    """Serialize a fixed requested source, independently of served_model()."""
    precision = validate_precision(loop.config[PRECISION_KEY])
    return export_served_adapter(precision_snapshot(loop, source=source), path,
                                 precision=precision, extra_metadata=extra_metadata)


export_precision_adapter = export_particle_adapter


def export_served_adapter(served, path, *, precision, extra_metadata=None):
    validate_precision(precision)
    generator, router = served.generator, served.router
    if not isinstance(generator, SupraParticleHost) or not isinstance(router, SupraParticleRouter):
        raise TypeError("precision export requires Supra particle owners")
    if _precision_class(generator) != precision:
        raise ValueError("export declared precision differs from its actual arithmetic")
    if generator.architecture != GATED_PARTICLE_V3 or any(
            branch.architecture != GATED_PARTICLE_V3 for branch in generator.particle_branches()):
        raise ValueError("precision export requires the unchanged shared-Up gated architecture")
    if served.source not in ("fast", "averaged") or generator.sites != router.sites or served.routing is None:
        raise ValueError("precision export source/sites/routing mismatch")
    # Revalidate storage/flags without changing the already matching class.
    install_student_final_(generator, precision)
    candidate = served.routing.candidate_for(served.models, served.table,
                                             averaged=served.source == "averaged")
    state = {key: value.detach().cpu().contiguous().clone() for key, value in
             _export_state(generator, router, candidate.table, candidate.log_mass).items()}
    config = dict(rank=generator.rank, z_dim=generator.z_dim, cfg=generator.cfg,
                  num_particles=len(candidate.table), sites=list(generator.sites),
                  site_input_dims=deepcopy(router.site_input_dims), sampling="clean",
                  routed_geometry="mass_atoms_v1", served_source=served.source,
                  completed_steps=served.completed_steps, architecture=GATED_PARTICLE_V3,
                  student_final_precision=precision, precision_spec=precision_spec(precision),
                  student_final_head=_head_contract(generator),
                  extra={} if extra_metadata is None else deepcopy(extra_metadata))
    path = Path(path)
    if path.exists():
        raise FileExistsError("precision export refuses to overwrite an existing artifact")
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(state, str(path), metadata={"format": FORMAT,
              "config": json.dumps(config, sort_keys=True),
              "native_pins": json.dumps(native_pins(), sort_keys=True)})
    return dict(path=str(path), bytes=path.stat().st_size, tensors=len(state),
                parameters=sum(value.numel() for value in state.values()), config=config)


def load_particle_adapter(base_model, path, *, precision=None, expected_precision=None,
                          device=None, source=None):
    """Strict arithmetic-aware reload; initialized values never replace state.

The caller must name the expected arithmetic. `precision` is a convenient
alias for `expected_precision`; neither can silently override file metadata.
"""
    if precision is None and expected_precision is None:
        raise ValueError("serving reload requires explicit expected precision")
    if precision is not None and expected_precision is not None and precision != expected_precision:
        raise ValueError("conflicting expected precision arguments")
    expected_precision = validate_precision(expected_precision if precision is None else precision)
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    if metadata.get("format") != FORMAT:
        raise ValueError("unsupported precision export format")
    if json.loads(metadata.get("native_pins", "null")) != native_pins():
        raise ValueError("precision export native model/backend pins differ")
    config = json.loads(metadata["config"])
    if (config.get(PRECISION_KEY) != expected_precision
            or config.get("precision_spec") != precision_spec(expected_precision)):
        raise ValueError("export precision/arithmetic differs from expected mode")
    if (config.get("architecture") != GATED_PARTICLE_V3 or config.get("sampling") != "clean"
            or config.get("routed_geometry") != "mass_atoms_v1"):
        raise ValueError("precision export architecture/clean routing law differs")
    if config.get("served_source") not in ("fast", "averaged") or (
            source is not None and source != config["served_source"]):
        raise ValueError("precision export source differs")
    if any(type(config.get(key)) is not int or config[key] < 1
           for key in ("rank", "z_dim", "num_particles")):
        raise ValueError("invalid precision export dimensions")
    if (type(config.get("completed_steps")) is not int or config["completed_steps"] < 0
            or type(config.get("cfg")) not in (int, float) or not math.isfinite(config["cfg"])):
        raise ValueError("invalid precision export clock/CFG")
    sites = tuple(config["sites"])
    dimensions = adapter_input_dims(base_model, sites)
    if sites != adapter_sites(base_model) or config["site_input_dims"] != dimensions:
        raise ValueError("precision export must cover exactly the native ordered adapter sites")
    # Constructors create owner structure, not restored values. Preserve caller
    # CPU RNG and install forward semantics before strict learned tensor loading.
    with torch.random.fork_rng(devices=[]):
        generator = SupraParticleHost(base_model, rank=config["rank"], z_dim=config["z_dim"],
            cfg=config["cfg"], sites=sites, architecture=GATED_PARTICLE_V3)
        install_student_final_(generator, expected_precision)
        router = SupraParticleRouter(num_particles=config["num_particles"], z_dim=config["z_dim"],
                                     site_input_dims=dimensions)
        table = torch.zeros(config["num_particles"], config["z_dim"])
    if config.get("student_final_head") != _head_contract(generator):
        raise ValueError("precision export frozen final-head storage/shape differs")
    expected = _export_state(generator, router, table, router.log_mass)
    state = load_file(str(path), device="cpu")
    if set(state) != set(expected):
        raise ValueError("precision export tensor names differ from the complete schema")
    for key, value in state.items():
        if value.shape != expected[key].shape or value.dtype != expected[key].dtype:
            raise ValueError("precision export tensor shape/dtype mismatch: " + key)
        valid = (bool((torch.isfinite(value) | torch.isneginf(value)).all())
                 and bool(torch.isfinite(value).any())) if key == "bank.log_mass" else bool(torch.isfinite(value).all())
        if not valid:
            raise ValueError("invalid precision export tensor values: " + key)
    for site, branch in zip(sites, generator.particle_branches()):
        for name in ("down", "bridge", "up"):
            prefix = f"generator.model.{site}.{name}."
            getattr(branch, name).load_state_dict(
                {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}, strict=True)
    router.load_state_dict({**{key[len("router."):]: value for key, value in state.items()
                               if key.startswith("router.")}, "log_mass": state["bank.log_mass"]}, strict=True)
    device = next(base_model.parameters()).device if device is None else torch.device(device)
    return CleanParticleAdapter(generator, router, state["bank.table"], config).to(device)

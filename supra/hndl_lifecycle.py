"""Distinct Supra core0.12/HyperGAN develop/HNDL lifecycle candidate.

The product validates its native assets and owns orchestration. All network
execution, GAN updates, optimizers, penalties and checkpoint validation stay in
the verified shared packages. Historical factories and receipts are unchanged.
"""
from contextlib import contextmanager
import inspect
import json
import math
from pathlib import Path

import torch
from particle_sliders.lifecycle import Backend, Binding, Capabilities
from particle_sliders.field_critic import seed_initialization_rng

from . import cache, campaign, hndl, native_io
from .runtime import MODEL_REV, T5_REV
from .shared import file_sha, source_cohort

CAPABILITIES = Capabilities(generator_arm="adapter_gan", render=True,
    strength_min=None, strength_max=None, reference_images=False,
    base_precisions=("float32",), render_outputs=("pil",),
    qualification="Distinct HNDL CPU software cohort; pretrained/native CUDA qualification pending",
    operations=("train", "resume", "evaluate", "export", "response", "render"))


def validate(options):
    allowed = {"native_config", "cache", "cache_sha256", "contexts", "contexts_sha256",
               "frozen_base_sha256", "device", "rank", "steps", "batch", "seed", "ema_decay",
               "critic_width", "base_precision"}
    if set(options) - allowed:
        raise ValueError("Unknown Supra HNDL option")
    if not isinstance(options.get("native_config"), dict):
        raise ValueError("Exact reviewed native_config required")
    for key in ("cache", "contexts"):
        if not isinstance(options.get(key), str) or not Path(options[key]).is_file():
            raise ValueError(key + " must be a staged local file")
    for key in ("cache_sha256", "contexts_sha256", "frozen_base_sha256"):
        value = options.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(key + " requires exact lowercase SHA256")
    for key, default, low, high in (("steps", 4, 1, 28800), ("batch", 4, 4, 4),
                                   ("rank", 16, 1, 256), ("critic_width", 32, 1, 1024)):
        value = options.get(key, default)
        if type(value) is not int or not low <= value <= high:
            raise ValueError(key + " differs from the bounded Supra recipe")
    decay = options.get("ema_decay")
    if decay is not None and (isinstance(decay, bool) or not isinstance(decay, (int, float))
                              or not math.isfinite(decay) or not 0 <= decay < 1):
        raise ValueError("EMA decay must be a finite number in [0,1)")
    if options.get("base_precision", "float32") != "float32":
        raise ValueError("Supra HNDL requires FP32 native storage")
    device = options.get("device")
    if not isinstance(device, str) or (device != "cpu" and not (
            device.startswith("cuda:") and device[5:].isdigit())):
        raise ValueError("Explicit cpu fixture or cuda:N device required")
    hndl.verify_sources()
    cache.validate_live_supra_context_preparation(options["native_config"],
        dict(model_revision=MODEL_REV, text_revision=T5_REV))
    receipt = native_io.read_context_receipt(options["contexts"])
    if (receipt.get("sha256") != options["contexts_sha256"]
            or file_sha(options["contexts"]) != options["contexts_sha256"]
            or file_sha(options["cache"]) != options["cache_sha256"]):
        raise ValueError("Frozen local asset checksum differs")


def preflight(options, request, staged):
    runtime = staged.get("runtime")
    if options["device"] == "cpu" and runtime is None:
        raise ValueError("CPU requires an explicitly staged HNDL fixture runtime")
    if request["operation"] == "render":
        native_io.validate_render(request["native"], partial=True)
    if runtime is not None and (str(runtime.device) != options["device"]
            or runtime.rank != options.get("rank", 16) or not hasattr(runtime, "vae")
            or not runtime.loras or any(not isinstance(layer, hndl.HNDLLoRALinear) for layer in runtime.loras)):
        raise ValueError("Staged HNDL runtime device/rank/decoder/network differs")


def prepare(options, *, runtime=None):
    from particle_sliders.fibo_hndl import HNDLFieldCritic
    from particle_sliders.pg09 import ConditionalPG09
    from particle_sliders.pg09_backends import SupraRuntimeResponseAdapter

    hndl.verify_sources()
    config = options["native_config"]
    receipt = native_io.read_context_receipt(options["contexts"])
    contexts = cache.load_supra_contexts(options["contexts"], options["contexts_sha256"], config)
    if (receipt.get("config_sha256") != cache.canonical_digest(config)
            or receipt.get("runtime_identity") != dict(model_revision=MODEL_REV, text_revision=T5_REV)
            or receipt.get("unique_contexts") != 19 or receipt.get("max_encode_calls") != 19
            or receipt.get("stored_unconditional") is not True):
        raise ValueError("Frozen context config/revision/preparation receipt differs")
    rows = cache.SupraCampaignRows(options["cache"], cache_sha256=options["cache_sha256"],
        config=config, contexts=contexts)
    if ({row["row"] for row in rows.records} != {row["name"] for row in config["rows"]}
            or {row["prompt"] for row in rows.holds} != set(config["preservation"])
            or not math.isclose(rows.response_scale(), cache.SUPRA_SCALE, rel_tol=0, abs_tol=1e-12)):
        raise ValueError("Native cache coverage/calibrated response scale differs")
    if runtime is None:
        if options["device"] == "cpu":
            raise ValueError("Native Supra requires CUDA; CPU uses staged fixtures")
        runtime = hndl.load_native(options)
    if (receipt.get("runtime_source_sha256") != file_sha(inspect.getsourcefile(type(runtime)))
            or str(runtime.device) != options["device"] or runtime.rank != options.get("rank", 16)
            or (runtime.MODEL_REV, runtime.T5_REV) != (MODEL_REV, T5_REV)
            or not runtime.stored_unconditional
            or native_io.frozen_digest(runtime) != options["frozen_base_sha256"]
            or any(parameter.dtype != torch.float32 for parameter in runtime.model.parameters())
            or any(torch.count_nonzero(layer.up.weight) for layer in runtime.loras)):
        raise ValueError("Fresh native HNDL base/context source/precision differs")
    close = native_io.snapshot_native(runtime)
    original_encode = runtime.encode
    try:
        runtime.cache = {prompt: tuple(value.to(runtime.device) for value in pair)
                         for prompt, pair in contexts.items()}
        admitted = runtime.cache

        def frozen_encode(prompt):
            if prompt not in admitted:
                raise ValueError("Prompt is outside frozen admitted Supra contexts")
            return admitted[prompt]

        runtime.encode = frozen_encode
        identity = hndl.model_identity(runtime)
        identity.update(context_receipt_sha256=file_sha(Path(options["contexts"]).with_suffix(".json")),
            training_sources={name: file_sha(Path(__file__).with_name(name))
                              for name in ("cache.py", "campaign.py", "hndl_lifecycle.py")})
        backend = SupraRuntimeResponseAdapter(runtime, model_identity=identity)
        device = torch.device(runtime.device)
        rows.device = device
        devices = [device.index] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            seed_initialization_rng(options.get("seed", 7), devices)
            critic = HNDLFieldCritic(4, rows.condition_width, (32, 32), options.get("critic_width", 32)).to(device)
        game = ConditionalPG09(backend, rows, critic, conditions=None, particles=None,
            response_domain="continuous_field", response_scale=rows.response_scale(),
            preset="bispf_optimized", batch_size=4, seed=options.get("seed", 7),
            total_steps=options.get("steps", 4), sampling="balanced")
        run = campaign.SupraCampaignRun(runtime, rows, backend, game)
        selected = {}
        for index in range(len(rows)):
            hold, source = rows._source(index)
            selected.setdefault((hold, source["prompt"], source.get("context")), index)
        campaign.teacher_parity(run, tuple(selected.values()))
        runtime.cfg_adapter_branch = "conditional_only"
        from .hndl_inference import BACKEND as INFERENCE_BACKEND, PROTOCOL
        specification = INFERENCE_BACKEND.specification(backend,
            model_identity=hndl.model_identity(runtime), protocol=PROTOCOL,
            configuration=dict(rank=runtime.rank))
        ownership = [(parameter, parameter.requires_grad) for parameter in runtime.model.parameters()]

        @contextmanager
        def observation():
            restore = native_io.snapshot_native(runtime)
            runtime.cache = dict(runtime.cache)
            runtime.encode = original_encode
            try:
                with source_cohort(hndl.LOCK):
                    yield
            finally:
                restore()

        return Binding(game, close,
            lambda **request: native_io.render(backend, runtime, **request),
            inference=specification,
            before_update=lambda request: (runtime.cfg_adapter_branch == "conditional_only"
                and all(parameter.requires_grad == flag for parameter, flag in ownership)),
            observation=observation)
    except BaseException:
        close()
        raise


BACKEND = Backend("supra_hndl", CAPABILITIES, prepare, validate, preflight=preflight)

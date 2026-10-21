"""Owned Supra factory for the exact image-v1 milestone; no GAN math here."""
from contextlib import contextmanager
from copy import deepcopy
import inspect
import json
import math
from pathlib import Path

from particle_sliders.lifecycle import Backend, Binding, Capabilities, checkpoint_cleanup

from .shared import file_sha, shared_module, source_cohort

LOCK = Path(__file__).with_name("lifecycle.lock.json")
CAPABILITIES = Capabilities(
    generator_arm="adapter_gan", render=True,
    strength_min=None, strength_max=None, reference_images=False,
    base_precisions=("float32",), render_outputs=("pil",),
    qualification="Source integration only; installed CPU conformance and native GPU parity pending",
)


def _assets(options):
    """Validate staged assets on CPU before native model construction."""
    with source_cohort(LOCK):
        bridge = shared_module("particle_sliders.pg09_remaining_campaigns")
        from .runtime import MODEL_REV, T5_REV
        config = options["native_config"]
        bridge.validate_live_supra_context_preparation(config,
            dict(model_revision=MODEL_REV, text_revision=T5_REV))
        contexts = bridge.load_supra_contexts(options["contexts"], options["contexts_sha256"], config)
        receipt = json.loads(Path(options["contexts"]).with_suffix(".json").read_text())
        if (receipt.get("config_sha256") != bridge.canonical_digest(config)
                or receipt.get("runtime_identity") != dict(model_revision=MODEL_REV, text_revision=T5_REV)
                or receipt.get("unique_contexts") != 19 or receipt.get("max_encode_calls") != 19
                or receipt.get("stored_unconditional") is not True):
            raise ValueError("Frozen context config/revision/preparation receipt differs")
        rows = bridge.SupraCampaignRows(options["cache"], cache_sha256=options["cache_sha256"],
            config=config, contexts=contexts)
        if ({r["row"] for r in rows.records} != {r["name"] for r in config["rows"]}
                or {r["prompt"] for r in rows.holds} != set(config["preservation"])):
            raise ValueError("Cache must cover every training subject and preservation prompt")
        if not math.isclose(rows.response_scale(), bridge.SUPRA_SCALE, rel_tol=0, abs_tol=1e-12):
            raise ValueError("Reviewed native cache response scale differs")
        return bridge, contexts, receipt, rows


def validate(options):
    allowed = {"native_config", "cache", "cache_sha256", "contexts", "contexts_sha256",
               "frozen_base_sha256", "device", "rank", "steps", "batch", "seed", "ema_decay",
               "critic_width", "base_precision"}
    if set(options) - allowed:
        raise ValueError("Unknown Supra session option; legacy/hub loading is unsupported")
    if not isinstance(options.get("native_config"), dict):
        raise ValueError("native_config must contain the exact reviewed local campaign configuration")
    for key in ("cache", "contexts"):
        if not isinstance(options.get(key), str) or not Path(options[key]).is_file():
            raise ValueError(key + " must be an existing local staged file")
    for key in ("cache_sha256", "contexts_sha256", "frozen_base_sha256"):
        value = options.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(key + " requires an exact lowercase SHA256")
    for key, default, low, high in (("steps", 4, 1, 28800), ("batch", 4, 4, 4),
                                   ("rank", 16, 1, 256), ("critic_width", 32, 1, 1024)):
        if type(options.get(key, default)) is not int or not low <= options.get(key, default) <= high:
            raise ValueError(key + " differs from the bounded Supra recipe")
    if options.get("base_precision", "float32") != "float32":
        raise ValueError("Supra preserves FP32 base storage and native CUDA BF16 autocast")
    device = options.get("device")
    if not isinstance(device, str) or (device != "cpu" and not (
            device.startswith("cuda:") and device[5:].isdigit())):
        raise ValueError("Explicit device must be cpu (staged fixtures only) or cuda:N")
    _assets(options)


def prepare(options, *, runtime=None):
    """Load locally, bind the complete native campaign and transfer ownership."""
    import torch
    from .pg09 import frozen_digest
    from .runtime import SupraRuntime, VAE_REV

    bridge, contexts, receipt, rows = _assets(options)
    if runtime is None:
        if options["device"] == "cpu":
            raise ValueError("CPU requires a separately admitted staged runtime; native Supra requires CUDA")
        with source_cohort(LOCK):
            runtime = SupraRuntime(options["device"], rank=options.get("rank", 16), allow_hub=False)
    if torch.device(runtime.device) != torch.device(options["device"]) or runtime.rank != options.get("rank", 16):
        raise ValueError("Staged runtime device/rank differs")
    if receipt["runtime_identity"] != dict(model_revision=runtime.MODEL_REV, text_revision=runtime.T5_REV):
        raise ValueError("Staged runtime model/T5 revisions differ")
    if receipt.get("runtime_source_sha256") != file_sha(inspect.getsourcefile(type(runtime))):
        raise ValueError("Frozen context runtime source differs")
    frozen = frozen_digest(runtime)
    if frozen != options["frozen_base_sha256"]:
        raise ValueError("Admitted frozen native base SHA differs")
    if any(p.dtype != torch.float32 for p in runtime.model.parameters()):
        raise ValueError("Supra native parameters must retain FP32 storage")

    # Snapshot caller-owned native state before the shared adapter changes modes
    # or checkpoint hooks. Native LoRA attachment itself belongs to SupraRuntime.
    restore_hooks = checkpoint_cleanup(runtime.model)
    modes = [(module, module.training) for module in runtime.model.modules()]
    multipliers = [(module, module.multiplier) for module in runtime.loras]
    old_cache, old_branch = runtime.cache, runtime.cfg_adapter_branch
    had_encode, old_encode = "encode" in vars(runtime), vars(runtime).get("encode")
    closed = False

    def close():
        nonlocal closed
        if closed:
            return
        closed = True
        restore_hooks()
        runtime.cache, runtime.cfg_adapter_branch = old_cache, old_branch
        if had_encode:
            runtime.encode = old_encode
        else:
            vars(runtime).pop("encode", None)
        for module, value in multipliers:
            module.multiplier = value
        for module, value in modes:
            module.training = value

    try:
        runtime.cache = {p: tuple(value.to(runtime.device) for value in pair) for p, pair in contexts.items()}
        admitted_cache = runtime.cache
        original_encode = runtime.encode

        def frozen_encode(prompt):
            if prompt not in admitted_cache:
                raise ValueError("Training prompt is outside frozen admitted Supra contexts")
            return admitted_cache[prompt]

        runtime.encode = frozen_encode
        identity = dict(model_revision=runtime.MODEL_REV, text_revision=runtime.T5_REV,
            vae_revision=VAE_REV, frozen_parameter_sha256=frozen,
            context_receipt_sha256=file_sha(Path(options["contexts"]).with_suffix(".json")),
            lifecycle_source_lock_sha256=file_sha(LOCK),
            product_sources={name: file_sha(Path(__file__).with_name(name))
                             for name in ("runtime.py", "shared.py", "pg09.py", "lifecycle.py")})
        run = bridge.build_supra_campaign(runtime, rows, model_identity=identity,
            total_steps=options.get("steps", 4), batch_size=4, seed=options.get("seed", 7),
            critic_width=options.get("critic_width", 32), response_scale=bridge.SUPRA_SCALE)
        runtime.cfg_adapter_branch = "conditional_only"
        selected = {}
        for index in range(len(rows)):
            hold, source = rows._source(index)
            selected.setdefault((hold, source["prompt"], source.get("context")), index)
        bridge.teacher_parity(run, tuple(selected.values()))

        @contextmanager
        def rendering_context():
            # New-prompt rendering may encode locally, but never extends training
            # contexts or leaves token counters/modes altered after observation.
            old_tokens = deepcopy(getattr(runtime, "token_counts", None))
            native_modes = [(module, module.training) for root in
                (runtime.model, getattr(runtime, "text_encoder", None), getattr(runtime, "vae", None))
                if root is not None for module in root.modules()]
            runtime.cache = dict(admitted_cache)
            runtime.encode = original_encode
            try:
                with source_cohort(LOCK):
                    yield
            finally:
                runtime.cache, runtime.encode = admitted_cache, frozen_encode
                if old_tokens is not None:
                    runtime.token_counts = old_tokens
                for module, value in native_modes:
                    module.training = value

        def render(*, prompt, strength, seed, images=None, steps=50, cfg=3., output_type="pil"):
            if (images or output_type != "pil" or not isinstance(prompt, str) or not prompt
                    or type(steps) is not int or not 1 <= steps <= 1000
                    or not math.isfinite(cfg) or cfg < 0):
                raise ValueError("Supra needs a prompt, no references, PIL output, bounded steps and finite nonnegative CFG")
            with rendering_context():
                return run.render(dict(prompt=prompt, seed=seed, scale=strength, steps=steps, cfg=cfg))

        return Binding(run.game, close, render)
    except BaseException:
        close()
        raise


BACKEND = Backend("supra", CAPABILITIES, prepare, validate)

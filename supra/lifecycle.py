"""Thin Supra factory for clean core 0.8.1; unexecuted source proposal."""
import inspect
import json
import math
from pathlib import Path

from particle_sliders.lifecycle import Backend, Binding, Capabilities

from .shared import file_sha
from . import native_io

LOCK = native_io.LOCK
CAPABILITIES = Capabilities(
    generator_arm="adapter_gan", render=True,
    strength_min=None, strength_max=None, reference_images=False,
    base_precisions=("float32",), render_outputs=("pil",),
    qualification="Unexecuted 0.8.1 source proposal; installed CPU conformance and native GPU parity pending",
    operations=("train", "resume", "evaluate", "export", "response", "render"),
)


def _assets(options):
    """Validate staged assets on CPU before native model construction."""
    native_io.verify_sources()
    from . import cache as bridge
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
    native_io.verify_sources()
    from . import cache
    from .runtime import MODEL_REV, T5_REV
    cache.validate_live_supra_context_preparation(options["native_config"],
        dict(model_revision=MODEL_REV, text_revision=T5_REV))
    receipt = json.loads(Path(options["contexts"]).with_suffix(".json").read_text())
    if (receipt.get("sha256") != options["contexts_sha256"]
            or file_sha(options["contexts"]) != options["contexts_sha256"]
            or file_sha(options["cache"]) != options["cache_sha256"]):
        raise ValueError("Frozen local asset checksum differs before native preparation")


def preflight(options, request, staged):
    if options["device"] == "cpu" and staged.get("runtime") is None:
        raise ValueError("CPU is supported only by an explicitly staged fixture runtime")
    if request["operation"] == "render":
        native_io.validate_render(request["native"], partial=True)
    runtime = staged.get("runtime")
    if runtime is not None and (str(runtime.device) != options["device"]
            or runtime.rank != options.get("rank", 16)):
        raise ValueError("Staged Supra device/rank differs before RNG/device preparation")
    if runtime is not None and not hasattr(runtime, "vae"):
        raise ValueError("The staged Supra image runtime must include its native decoder")


def prepare(options, *, runtime=None):
    """Load locally, bind the complete native campaign and transfer ownership."""
    import torch
    from .native_io import frozen_digest
    from . import campaign

    bridge, contexts, receipt, rows = _assets(options)
    if runtime is None:
        if options["device"] == "cpu":
            raise ValueError("CPU requires a separately admitted staged runtime; native Supra requires CUDA")
        runtime = native_io.load_native(options)
    if torch.device(runtime.device) != torch.device(options["device"]) or runtime.rank != options.get("rank", 16):
        raise ValueError("Staged runtime device/rank differs")
    if receipt["runtime_identity"] != dict(model_revision=runtime.MODEL_REV, text_revision=runtime.T5_REV):
        raise ValueError("Staged runtime model/T5 revisions differ")
    if not runtime.stored_unconditional:
        raise ValueError("Reviewed native stored unconditional context required")
    if receipt.get("runtime_source_sha256") != file_sha(inspect.getsourcefile(type(runtime))):
        raise ValueError("Frozen context runtime source differs")
    frozen = frozen_digest(runtime)
    if frozen != options["frozen_base_sha256"]:
        raise ValueError("Admitted frozen native base SHA differs")
    if any(p.dtype != torch.float32 for p in runtime.model.parameters()):
        raise ValueError("Supra native parameters must retain FP32 storage")

    restore_native = native_io.snapshot_native(runtime)
    original_encode = runtime.encode
    closed = False

    def close():
        nonlocal closed
        if not closed:
            closed = True
            restore_native()

    try:
        runtime.cache = {p: tuple(value.to(runtime.device) for value in pair) for p, pair in contexts.items()}
        admitted_cache = runtime.cache

        def frozen_encode(prompt):
            if prompt not in admitted_cache:
                raise ValueError("Training prompt is outside frozen admitted Supra contexts")
            return admitted_cache[prompt]

        runtime.encode = frozen_encode
        native_identity = native_io.model_identity(runtime)
        identity = dict(native_identity,
            context_receipt_sha256=file_sha(Path(options["contexts"]).with_suffix(".json")),
            training_sources={name: file_sha(Path(__file__).with_name(name))
                              for name in ("cache.py", "campaign.py", "lifecycle.py")})
        run = campaign.build_supra_campaign(runtime, rows, model_identity=identity,
            total_steps=options.get("steps", 4), batch_size=4, seed=options.get("seed", 7),
            critic_width=options.get("critic_width", 32), response_scale=bridge.SUPRA_SCALE)
        runtime.cfg_adapter_branch = "conditional_only"
        selected = {}
        for index in range(len(rows)):
            hold, source = rows._source(index)
            selected.setdefault((hold, source["prompt"], source.get("context")), index)
        campaign.teacher_parity(run, tuple(selected.values()))

        from .native_inference import BACKEND as INFERENCE_BACKEND, PROTOCOL
        specification = INFERENCE_BACKEND.specification(run.backend,
            model_identity=native_identity, protocol=PROTOCOL,
            configuration=dict(rank=runtime.rank))
        ownership = [(parameter, parameter.requires_grad) for parameter in runtime.model.parameters()]

        def before_update(request):
            # Pure admission over already verified native evidence, without a
            # forward, asset read or RNG consumption inside the update hook.
            return (runtime.cfg_adapter_branch == "conditional_only"
                    and all(parameter.requires_grad == grad for parameter, grad in ownership))

        def render(**request):
            return native_io.render(run.backend, runtime, **request)

        return Binding(run.game, close, render, inference=specification,
            before_update=before_update,
            observation=lambda: native_io.observation(runtime, original_encode))

    except BaseException:
        close()
        raise


BACKEND = Backend("supra", CAPABILITIES, prepare, validate, preflight=preflight)

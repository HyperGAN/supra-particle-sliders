"""Product IO around the shared GAN. No product loss, optimizer or sampler."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import tempfile

import torch

from supra.shared import LOCK, file_sha, shared_module

OBJECTIVE = "pg09_gan_only"


def frozen_digest(runtime):
    """Hash every frozen parameter and buffer without copying the whole model."""
    h = hashlib.sha256()
    for name, value in list(runtime.model.named_parameters()) + list(runtime.model.named_buffers()):
        if value.requires_grad:
            continue
        value = value.detach().cpu().contiguous()
        h.update(json.dumps([name, list(value.shape), str(value.dtype)]).encode())
        h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def binding(config, cache_path, contexts_path, frozen_sha256):
    return dict(objective=OBJECTIVE, shared_lock_sha256=file_sha(LOCK),
                product_sources={p: file_sha(Path(__file__).parent / p)
                                 for p in ("runtime.py", "shared.py", "pg09.py")},
                trainer_sha256=file_sha(Path(__file__).resolve().parents[1] / "scripts/train_pg09.py"),
                config=deepcopy(config), cache_sha256=file_sha(cache_path),
                contexts_sha256=file_sha(contexts_path),
                context_receipt_sha256=file_sha(Path(contexts_path).with_suffix(".json")),
                frozen_parameter_sha256=frozen_sha256)


def build_run(runtime, config, cache_path, contexts_path, *, cache_sha256,
              contexts_sha256, expected_frozen_sha256, total_steps=28800, seed=7,
              critic_width=32):
    bridge = shared_module("particle_sliders.pg09_remaining_campaigns")
    identity = dict(model_revision=runtime.MODEL_REV, text_revision=runtime.T5_REV)
    bridge.validate_live_supra_context_preparation(config, identity)
    receipt = json.loads(Path(contexts_path).with_suffix(".json").read_text())
    if (receipt.get("config_sha256") != bridge.canonical_digest(config)
            or receipt.get("runtime_identity") != identity
            or receipt.get("unique_contexts") != 19 or receipt.get("max_encode_calls") != 19
            or receipt.get("stored_unconditional") is not True
            or receipt.get("runtime_source_sha256") != file_sha(inspect.getsourcefile(type(runtime)))):
        raise ValueError("Frozen context native runtime/config preparation binding differs")
    contexts = bridge.load_supra_contexts(contexts_path, contexts_sha256, config)
    actual = frozen_digest(runtime)
    if actual != expected_frozen_sha256:
        raise ValueError("Admitted frozen base SHA differs")
    # Populate the native runtime from the checksum-bound export, never encode.
    runtime.cache = {p: tuple(v.to(runtime.device) for v in pair) for p, pair in contexts.items()}
    rows = bridge.SupraCampaignRows(cache_path, cache_sha256=cache_sha256, config=config,
                                    contexts=contexts)
    if ({r["row"] for r in rows.records} != {r["name"] for r in config["rows"]}
            or {r["prompt"] for r in rows.holds} != set(config["preservation"])):
        raise ValueError("Native cache must cover every training subject and preservation prompt")
    run = bridge.build_supra_campaign(runtime, rows,
        model_identity=dict(model_revision=runtime.MODEL_REV, text_revision=runtime.T5_REV,
                            frozen_parameter_sha256=actual),
        total_steps=total_steps, seed=seed, critic_width=critic_width,
        response_scale=bridge.SUPRA_SCALE)
    # The exported native adapter selects this same shared CFG path on inference.
    runtime.cfg_adapter_branch = "conditional_only"
    run.product_binding = binding(config, cache_path, contexts_path, actual)
    return run


def save_checkpoint(run, directory):
    """Publish an immutable full game packet, adapter and checksum atomically."""
    directory = Path(directory)
    if frozen_digest(run.runtime) != run.product_binding["frozen_parameter_sha256"]:
        raise ValueError("Frozen base changed before checkpoint")
    if directory.exists():
        raise FileExistsError(directory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".pending-", dir=directory.parent))
    try:
        state = run.game.state_dict()
        torch.save(dict(schema="supra-shared-pg09-v1", binding=run.product_binding,
                        game=state), temporary / "checkpoint.pt")
        run.runtime.save(temporary / "adapter.safetensors", dict(
            training_objective=OBJECTIVE, cfg_adapter_branch="conditional_only",
            step=run.game.completed_steps, total_steps=run.game.recipe.total_steps,
            shared_lock_sha256=file_sha(LOCK)))
        receipt = dict(schema="supra-shared-pg09-receipt-v1", binding=run.product_binding,
                       step=run.game.completed_steps,
                       files={p.name: file_sha(p) for p in temporary.iterdir()})
        (temporary / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        temporary.rename(directory)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return directory


def load_checkpoint(run, directory):
    directory = Path(directory)
    if frozen_digest(run.runtime) != run.product_binding["frozen_parameter_sha256"]:
        raise ValueError("Frozen base changed before checkpoint reload")
    receipt = json.loads((directory / "receipt.json").read_text())
    if receipt.get("schema") != "supra-shared-pg09-receipt-v1" or receipt.get("binding") != run.product_binding:
        raise ValueError("Checkpoint product/config/cache/source binding differs")
    if set(receipt["files"]) != {"checkpoint.pt", "adapter.safetensors", "adapter.json"}:
        raise ValueError("Checkpoint files differ")
    for name, expected in receipt["files"].items():
        if file_sha(directory / name) != expected:
            raise ValueError("Checkpoint checksum differs: " + name)
    # CPU-map all streams and CUDA RNG state before the shared transactional load.
    packet = torch.load(directory / "checkpoint.pt", map_location="cpu", weights_only=True)
    if (packet.get("schema") != "supra-shared-pg09-v1" or packet.get("binding") != run.product_binding
            or packet["game"]["completed_steps"] != receipt["step"]):
        raise ValueError("Checkpoint packet binding/step differs")
    run.game.load_state_dict(packet["game"])
    return receipt


def verify_teacher_cache(run):
    """Check both native trajectories for every subject and preservation prompt."""
    bridge = shared_module("particle_sliders.pg09_remaining_campaigns")
    selected = {}
    for i in range(len(run.rows)):
        hold, source = run.rows._source(i)
        selected.setdefault((hold, source["prompt"], source.get("context")), i)
    return bridge.teacher_parity(run, tuple(selected.values()))


@torch.no_grad()
def render(runtime, prompt, seed, scale, steps, cfg):
    """Shared response renderer with the product's native RNG and VAE."""
    backend_module = shared_module("particle_sliders.pg09_backends")
    backend = backend_module.SupraRuntimeResponseAdapter(runtime,
        model_identity=dict(frozen_parameter_sha256=frozen_digest(runtime)))
    row = backend_module.ResponseRow(prompt, prompt, "", runtime.noise(seed),
        schedule=backend_module.FlowSchedule.supra(steps), guidance=cfg)
    image = backend.render(row, scale=scale)
    if not torch.isfinite(image).all():
        raise FloatingPointError("Nonfinite native VAE output")
    from PIL import Image
    pixels = ((image.float().clamp(-1, 1) + 1) * 127.5).round().byte()[0]
    return Image.fromarray(pixels.permute(1, 2, 0).cpu().numpy())

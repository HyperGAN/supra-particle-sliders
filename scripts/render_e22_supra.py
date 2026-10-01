#!/usr/bin/env python3
"""Render a restored Supra E22 hybrid beside its untrained two-site baseline."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from supra.particle_pilot import make_loop, restore
from supra.runtime import MODEL_REV, T5_REV, VAE_REV, SupraRuntime


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(tensor):
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def emit(value):
    print(json.dumps(value, sort_keys=True), flush=True)


@torch.no_grad()
def render(served, runtime, *, caption_id=1, seed=42, steps=50):
    """Advance the complete served model on its own native Euler trajectory."""
    stream = torch.Generator(device=runtime.device).manual_seed(seed)
    z = torch.randn(1, 4, 32, 32, generator=stream, device=runtime.device)
    initial_hash = tensor_sha256(z)
    for index in range(steps):
        # The encoder snapshot owns the exact conditional/unconditional text
        # tensors. Each new latent is repacked and the entire DiT is rerun.
        extra = z.new_tensor([[index / steps, caption_id, 1.]])
        context = torch.cat((z.reshape(1, 4096), extra), dim=1)
        velocity = (served.routed_forward(context, perturb=False, output_noise=False)
                    + served.encoder.teacher_velocity(context))
        if velocity.shape != z.shape or not bool(torch.isfinite(velocity).all()):
            raise RuntimeError("served Supra returned an invalid native velocity")
        z = z + velocity / steps
        if (index + 1) % 10 == 0:
            emit(dict(event="render_progress", served_source=served.source, step=index + 1, steps=steps))
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        decoded = runtime.vae.decode(z / .18215).sample
    pixels = ((decoded.float().clamp(-1, 1) + 1) * 127.5).round().byte()[0]
    image = Image.fromarray(pixels.permute(1, 2, 0).cpu().numpy())
    if image.size != (256, 256):
        raise RuntimeError(f"unexpected native Supra image size {image.size}")
    return image, dict(initial_latent_sha256=initial_hash, endpoint_sha256=tensor_sha256(z),
                       served_source=served.source, completed_steps=served.completed_steps)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("outputs/e22-pilot/data.pt"))
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--mode", choices=("fixed", "movable", "full"), required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--probe-interval", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.probe_interval < 1:
        parser.error("probe interval must be positive")
    checkpoint_path = args.checkpoint or args.data.parent / "verification" / f"{args.mode}-final.pt"
    output = args.output or args.data.parent / "verification" / f"render-{args.mode}"
    if any((output / name).exists() for name in ("trained.png", "initial-hybrid.png", "metadata.json")):
        parser.error("output already contains render evidence; choose another --output")
    for path in (args.data, args.teacher, checkpoint_path):
        if not path.is_file():
            parser.error(f"missing input {path}")
    data = torch.load(args.data, map_location="cpu", weights_only=False)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    dataset = data["metadata"]
    teacher_hash = file_sha256(args.teacher)
    if teacher_hash != dataset["adapter_sha256"]:
        raise ValueError("teacher differs from the published adapter used to build the paired cache")
    if state["config"]["mode"] != args.mode:
        raise ValueError("requested mode differs from the saved checkpoint")
    caption_id, seed, steps, strength = 1, 42, 50, 1.
    prompt = dataset["prompts"][caption_id]
    root = Path(__file__).resolve().parents[1]
    sources = ("scripts/render_e22_supra.py", "supra/particle_adapter.py", "supra/particle_game.py",
               "supra/particle_pilot.py", "supra/runtime.py")
    metadata = dict(schema="supra-e22-hybrid-render-v1", mode=args.mode,
                    data_path=str(args.data.resolve()), data_sha256=file_sha256(args.data),
                    checkpoint_path=str(checkpoint_path.resolve()), checkpoint_sha256=file_sha256(checkpoint_path),
                    teacher_path=str(args.teacher.resolve()), teacher_sha256=teacher_hash,
                    adapter_hf_id=dataset["adapter_hf_id"], adapter_hf_revision=dataset["adapter_hf_revision"],
                    model_revision=MODEL_REV, text_encoder_revision=T5_REV, vae_revision=VAE_REV,
                    source_sha256={name: file_sha256(root / name) for name in sources},
                    prompt=prompt, caption_id=caption_id, seed=seed, strength=strength,
                    steps=steps, cfg=3., image_size=256, sampler="native Euler t=i/50, z+=velocity/50",
                    served_law="clean routed snapshot; no DV12 or output noise during rendering",
                    task=state["config"],
                    limitation="Hybrid: only two of 71 published ordinary LoRA branches are replaced by nonlinear routed particle branches; 69 published branches remain frozen.",
                    initial_hybrid="Same frozen 69-branch background; both new particle up-projections zero.")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    runtime = SupraRuntime(device=args.device, rank=16, allow_hub=False)
    runtime.load(args.teacher)
    runtime.model.requires_grad_(False)
    runtime.text_encoder.to("cpu")
    loop = make_loop(runtime.model, data, mode=args.mode, device=args.device,
                     batch_size=state["config"]["batch_size"],
                     branch_lr=state["config"]["branch_lr"], probe_interval=args.probe_interval)
    # Both images start from the same private seed and evolve separately. The
    # initial hybrid is a fresh initialization, not the post-training fast bank.
    variants = []
    for label, filename in (("initial_hybrid", "initial-hybrid.png"), ("trained", "trained.png")):
        if label == "trained":
            restore(loop, state)
            del state
            gc.collect()
        served = loop.policy.served_model()
        started = time.perf_counter()
        image, details = render(served, runtime, caption_id=caption_id, seed=seed, steps=steps)
        torch.cuda.synchronize(runtime.device)
        row = dict(label=label, filename=filename, seconds=time.perf_counter() - started, **details)
        png_info = PngInfo()
        png_info.add_text("supra_e22", json.dumps(dict(prompt=prompt, seed=seed, steps=steps,
                                                      strength=strength, mode=args.mode, **row), sort_keys=True))
        image.save(output / filename, pnginfo=png_info)
        row["image_sha256"] = file_sha256(output / filename)
        variants.append(row)
        emit(dict(event="render_complete", output=str(output / filename), **row))
        del served
        gc.collect()
        torch.cuda.empty_cache()
    metadata["samples"] = variants
    metadata["matched_initial_latents"] = len({row["initial_latent_sha256"] for row in variants}) == 1
    if not metadata["matched_initial_latents"]:
        raise RuntimeError("render variants did not use identical initial latent noise")
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    emit(dict(event="complete", metadata_path=str(output / "metadata.json")))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Cache real released Supra teacher pairs for the bounded E22 routing pilot.

The complete rank-16 release supplies the targets. The partial background
retains 69 frozen LoRA branches and disables the last two cross-attention
output branches. No optimizer, fitted target, or initializer is used here.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time

import torch
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from supra.runtime import MODEL_ID, MODEL_REV, MODEL_SOURCE, T5_REV, VAE_REV, SupraRuntime

ADAPTER_SHA256 = "b2191a46f59fa058bfacbcc659e8d7b6a7da4eeacbf65a3b619c438c3f449d3e"
ADAPTER_REPO = "ntc-ai/supra-particle-sliders"
ADAPTER_REVISION = "dca42f84c37d74748b4f45f4d7e75ec2492541c5"
SITES = ("blocks.12.cross_attn.proj", "blocks.13.cross_attn.proj")
TRAIN_NAMES = ("knight", "cave", "sorceress")
TEST_NAMES = ("bridge_heldout", "fruit_control")
TIMES = {"fit": (.1, .3, .5, .7), "guard": (.2, .4, .6, .8),
         "test": (.15, .35, .55, .75)}


def emit(event, **values):
    print(json.dumps({"event": event, **values}), flush=True)


def file_sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def tensor_sha256(value):
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


@torch.no_grad()
def selective_velocity(runtime, prompt, z, t, *, mode, cfg=3.):
    """Use the original full-context CFG forward with restored branch strengths."""
    if mode not in ("teacher", "partial", "base"):
        raise ValueError("mode must be teacher, partial, or base")
    if cfg <= 1:
        raise ValueError("this preparation fixes the published conditional/unconditional CFG contract")
    ctx, mask = runtime.encode(prompt)
    uncond_ctx, uncond_mask = runtime.encode("")
    ctx = torch.cat((ctx.expand(len(z), -1, -1), uncond_ctx.expand(len(z), -1, -1)))
    mask = torch.cat((mask.expand(len(z), -1), uncond_mask.expand(len(z), -1)))
    t = torch.as_tensor(t, device=runtime.device, dtype=torch.float32).reshape(-1)
    if t.numel() == 1:
        t = t.expand(len(z))
    if t.numel() != len(z):
        raise ValueError("time must supply one scalar per latent")
    previous = [(module, module.multiplier) for module in runtime.loras]
    disabled = {runtime.model.get_submodule(name) for name in SITES}
    try:
        for module in runtime.loras:
            module.multiplier = 0. if mode == "base" or (mode == "partial" and module in disabled) else 1.
        with torch.autocast("cuda", dtype=torch.bfloat16):
            values = runtime.model(torch.cat((z, z)), torch.cat((t, t)), ctx, mask).float()
        conditional, unconditional = values.chunk(2)
        return unconditional + cfg * (conditional - unconditional)
    finally:
        for module, multiplier in previous:
            module.multiplier = multiplier


@contextmanager
def bypass_lora_branches(runtime):
    """Evaluate actual frozen base linears independently of the strength-zero path."""
    previous = [(layer, "forward" in layer.__dict__, layer.__dict__.get("forward"))
                for layer in runtime.loras]
    try:
        for layer, _, _ in previous:
            layer.forward = layer.base.forward
        yield
    finally:
        for layer, existed, value in previous:
            if existed:
                layer.forward = value
            else:
                del layer.forward


@torch.no_grad()
def decode(runtime, latent, path):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        image = runtime.vae.decode(latent / .18215).sample
    pixels = ((image.float().clamp(-1, 1) + 1) * 127.5).round().byte()[0]
    Image.fromarray(pixels.permute(1, 2, 0).cpu().numpy()).save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--adapter", type=Path,
                        default=Path("/ml2/hypergan/supra-concept-sliders/outputs/final-boss-supra-converged/final-boss-supra.safetensors"))
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/e22-pilot")
    args = parser.parse_args()
    if file_sha256(args.adapter) != ADAPTER_SHA256:
        raise ValueError("use the exact published converged rank-16 teacher adapter")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    started = time.perf_counter()
    emit("prepare_begin", device=args.device, adapter=str(args.adapter), output=str(args.out))
    runtime = SupraRuntime(args.device, rank=16, allow_hub=False)
    runtime.load(args.adapter)
    runtime.model.requires_grad_(False)
    if runtime.base_params != 104_094_736 or len(runtime.loras) != 71:
        raise ValueError("expected the released full-size Supra model and all 71 adapter branches")
    torch.cuda.synchronize(runtime.device)
    loaded_seconds = time.perf_counter() - started
    torch.cuda.reset_peak_memory_stats(runtime.device)

    prompts_file = ROOT / "configs/supra/prompts-final-boss.yaml"
    card = yaml.safe_load(prompts_file.read_text())
    rows = {row["name"]: row["neutral"] for row in card["rows"]}
    rows.update({row["name"]: row["prompt"] for row in card["verification"]})
    names = ("unconditional", *TRAIN_NAMES, *TEST_NAMES)
    prompts = ["", *(rows[name] for name in (*TRAIN_NAMES, *TEST_NAMES))]
    caption_ids = {name: index for index, name in enumerate(names)}
    for prompt in prompts:
        runtime.encode(prompt)
    text_contexts = torch.cat([runtime.encode(prompt)[0] for prompt in prompts]).detach().float().cpu()
    text_masks = torch.cat([runtime.encode(prompt)[1] for prompt in prompts]).detach().float().cpu()
    if text_contexts.shape != (6, 128, 768) or text_masks.shape != (6, 128):
        raise ValueError("unexpected released text conditioning shapes")
    emit("teacher_loaded", seconds=loaded_seconds, base_parameters=runtime.base_params,
         adapter_branches=len(runtime.loras), replacement_sites=list(SITES), captions=names)

    # One fixed data stream produces different latent inputs across splits.
    # These draws define a paired dataset, rather than repeated seed experiments.
    rng = torch.Generator(device=runtime.device).manual_seed(42)
    initial_rng = rng.get_state().cpu()
    trajectory_receipts, splits = [], {}
    steps, cfg = 20, 3.
    cache_started = time.perf_counter()
    for split, times in TIMES.items():
        split_names = TEST_NAMES if split == "test" else TRAIN_NAMES
        buffers = {key: [] for key in ("context", "targets", "partial", "base")}
        chosen = {round(t * steps): t for t in times}
        if any(abs(index / steps - value) > 1e-12 for index, value in chosen.items()):
            raise ValueError("all requested times must belong to the fixed Euler grid")
        for name in split_names:
            prompt = rows[name]
            before = rng.get_state().cpu()
            z = torch.randn(1, 4, 32, 32, device=runtime.device, generator=rng)
            initial_z_sha = tensor_sha256(z)
            trajectory_started = time.perf_counter()
            for index in range(steps):
                t = index / steps
                teacher = selective_velocity(runtime, prompt, z, t, mode="teacher", cfg=cfg)
                if index in chosen:
                    partial = selective_velocity(runtime, prompt, z, t, mode="partial", cfg=cfg)
                    base = selective_velocity(runtime, prompt, z, t, mode="base", cfg=cfg)
                    packed = torch.cat((z.flatten(1),
                                        z.new_tensor([[t, caption_ids[name], 1.]])), dim=1)
                    buffers["context"].append(packed.detach().float().cpu())
                    for key, value in (("targets", teacher), ("partial", partial), ("base", base)):
                        if not torch.isfinite(value).all():
                            raise RuntimeError(f"nonfinite frozen output for {split}/{name}/{t}/{key}")
                        buffers[key].append(value.detach().float().cpu())
                z = z + teacher / steps
            torch.cuda.synchronize(runtime.device)
            row = dict(split=split, name=name, caption_id=caption_ids[name], times=list(times),
                       z0_sha256=initial_z_sha, data_rng_before_sha256=tensor_sha256(before),
                       data_rng_after_sha256=tensor_sha256(rng.get_state()),
                       endpoint_sha256=tensor_sha256(z), seconds=time.perf_counter() - trajectory_started)
            trajectory_receipts.append(row)
            emit("trajectory_cached", **row)
        splits[split] = {key: torch.cat(values) for key, values in buffers.items()}
        emit("split_cached", split=split, contexts=len(splits[split]["context"]),
             partial_teacher_rmse=float((splits[split]["partial"] - splits[split]["targets"]).square().mean().sqrt()),
             base_teacher_rmse=float((splits[split]["base"] - splits[split]["targets"]).square().mean().sqrt()))
    torch.cuda.synchronize(runtime.device)
    cache_seconds = time.perf_counter() - cache_started
    assert [len(splits[name]["context"]) for name in TIMES] == [12, 12, 8]
    assert len({row["z0_sha256"] for row in trajectory_receipts}) == 8

    # Preflight compares native inference, the selective helper, and a true
    # branch-bypassed base at identical inputs before rendering one matched pair.
    prompt = rows["knight"]
    probe_z = runtime.noise(42)
    full = selective_velocity(runtime, prompt, probe_z, .5, mode="teacher", cfg=cfg)
    full_native = runtime.velocity(prompt, probe_z, .5, scale=1., cfg=cfg)
    zero = selective_velocity(runtime, prompt, probe_z, .5, mode="base", cfg=cfg)
    zero_native = runtime.velocity(prompt, probe_z, .5, scale=0., cfg=cfg)
    with bypass_lora_branches(runtime):
        pure_base = selective_velocity(runtime, prompt, probe_z, .5, mode="base", cfg=cfg)
    if not (torch.equal(full, full_native) and torch.equal(zero, zero_native) and torch.equal(zero, pure_base)):
        raise RuntimeError("selective CFG or strength-zero inference does not exactly match the pinned release")
    preflight_started = time.perf_counter()
    base_endpoint = runtime.trajectory(prompt, 42, steps=50, cfg=cfg, scale=0.)
    teacher_endpoint = runtime.trajectory(prompt, 42, steps=50, cfg=cfg, scale=1.)
    endpoint_edit = float((teacher_endpoint - base_endpoint).square().mean().sqrt())
    if not endpoint_edit > 0. or not torch.isfinite(teacher_endpoint).all():
        raise RuntimeError("the released teacher must produce a finite, nonzero endpoint edit")
    decode(runtime, base_endpoint, args.out / "preflight-base.png")
    decode(runtime, teacher_endpoint, args.out / "preflight-teacher.png")
    torch.cuda.synchronize(runtime.device)
    preflight = dict(prompt=prompt, seed=42, image_size=256, steps=50, cfg=cfg,
                     teacher_selective_matches_native_exact=True,
                     teacher_zero_matches_native_and_branch_bypassed_base_exact=True,
                     velocity_edit_rms=float((full - zero).square().mean().sqrt()),
                     endpoint_edit_rms=endpoint_edit,
                     base_endpoint_sha256=tensor_sha256(base_endpoint),
                     teacher_endpoint_sha256=tensor_sha256(teacher_endpoint),
                     seconds=time.perf_counter() - preflight_started,
                     images=["preflight-base.png", "preflight-teacher.png"])
    emit("preflight_complete", **preflight)

    source_paths = [Path(__file__).resolve(), ROOT / "supra/runtime.py", MODEL_SOURCE, prompts_file]
    metadata = dict(
        schema="supra-e22-paired-cache-v1",
        source_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        source_sha256={str(path.relative_to(ROOT)): file_sha256(path) for path in source_paths},
        model_id=MODEL_ID, model_revision=MODEL_REV, text_encoder_revision=T5_REV, vae_revision=VAE_REV,
        checkpoint=str(runtime.checkpoint), checkpoint_sha256=file_sha256(runtime.checkpoint),
        adapter_hf_id=ADAPTER_REPO, adapter_hf_revision=ADAPTER_REVISION,
        adapter_hf_filename="weights/final-boss-converged.safetensors",
        adapter_path=str(args.adapter.resolve()), adapter_sha256=ADAPTER_SHA256, adapter_rank=16,
        base_parameters=runtime.base_params, teacher="complete released 71-branch ordinary LoRA, strength 1",
        partial="same released background, last two cross-attention output LoRA branches disabled",
        partial_frozen_branches=69, replacement_sites=list(SITES),
        target_contract="full frozen teacher velocities at its own native latent trajectory states",
        latent_shape=[4, 32, 32], image_size=256, cfg=cfg, trajectory_steps=steps,
        conditioning_contract="context[:, :4096]=latent; [:,4096]=time; [:,4097]=caption_id; [:,4098]=strength",
        caption_names=list(names), prompts=prompts, split_times={key: list(value) for key, value in TIMES.items()},
        split_sizes={key: len(value["context"]) for key, value in splits.items()},
        data_rng_seed=42, data_rng_policy="single ordered stream, one distinct latent draw per split/prompt",
        data_rng_initial_sha256=tensor_sha256(initial_rng), data_rng_final_sha256=tensor_sha256(rng.get_state()),
        trajectories=trajectory_receipts, preflight=preflight,
        packages={name: importlib.metadata.version(name) for name in ("torch", "transformers", "diffusers", "safetensors")},
        hardware=dict(device=str(runtime.device), gpu=torch.cuda.get_device_name(runtime.device),
                      cuda=torch.version.cuda, cpu_threads=torch.get_num_threads(),
                      peak_allocated_mib=torch.cuda.max_memory_allocated(runtime.device) / 1024 ** 2),
        loading_seconds=loaded_seconds, cache_seconds=cache_seconds,
        preparation_seconds=time.perf_counter() - started,
        objective="none; frozen inference preparation only", output_errors="evaluation logs only",
    )
    dataset = {**splits, "text_contexts": text_contexts, "text_masks": text_masks,
               "prompts": prompts, "metadata": metadata, "data_rng_initial": initial_rng,
               "data_rng_final": rng.get_state().cpu()}
    data_path = args.out / "data.pt"
    torch.save(dataset, data_path)
    metadata["data_file_sha256"] = file_sha256(data_path)
    (args.out / "data.json").write_text(json.dumps(metadata, indent=2) + "\n")
    emit("prepare_complete", data=str(data_path), data_sha256=metadata["data_file_sha256"],
         split_sizes=metadata["split_sizes"], loading_seconds=loaded_seconds,
         cache_seconds=cache_seconds, total_seconds=metadata["preparation_seconds"],
         peak_allocated_mib=metadata["hardware"]["peak_allocated_mib"])


if __name__ == "__main__":
    with torch.no_grad():
        main()

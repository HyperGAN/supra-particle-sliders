#!/usr/bin/env python3
"""Final-boss LoRA: frozen positive velocity teachers, neutral student captions.

Uses the completed HyperGAN/krea2-particle-sliders recipe on Supra's native
Euler schedule and short prose prompts. No example images or text tuning.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageDraw

from supra.runtime import MODEL_REV, MODEL_SOURCE, T5_REV, VAE_REV, SupraRuntime


def write_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.no_grad()
def build_cache(rt, config, args):
    """Both frozen neutral and positive trajectories, shared state per teacher pair."""
    records = []
    positions = list(range(0, args.sample_steps, max(1, args.sample_steps // 10)))
    for row in config["rows"]:
        for seed_index in range(args.cache_seeds):
            seed = args.seed * 1009 + seed_index
            for context in ("neutral", "positive"):
                _, states = rt.trajectory(row[context], seed, args.sample_steps, args.cfg, keep=True)
                for i in positions:
                    z = states[i].to(rt.device)
                    t = i / args.sample_steps
                    neutral = rt.velocity(row["neutral"], z, t, cfg=args.cfg)
                    positive = rt.velocity(row["positive"], z, t, cfg=args.cfg)
                    records.append(dict(prompt=row["neutral"], z=z.cpu(), t=t,
                                        neutral=neutral.cpu(), target=positive.cpu(),
                                        hold=False, row=row["name"], seed=seed, context=context))
        print(f"Teacher cache: {row['name']}, {len(records)} states", flush=True)
    holds = []
    for idx, prompt in enumerate(config["preservation"]):
        _, states = rt.trajectory(prompt, args.seed * 1009 + idx, args.sample_steps, args.cfg, keep=True)
        for i in positions:
            z = states[i].to(rt.device)
            base = rt.velocity(prompt, z, i / args.sample_steps, cfg=args.cfg)
            holds.append(dict(prompt=prompt, z=z.cpu(), t=i / args.sample_steps,
                              neutral=base.cpu(), target=base.cpu(), hold=True))
    return records, holds


@torch.no_grad()
def compare(rt, config, args, folder, *, seeds=(42, 1234), full=True):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    rows = config["verification"] if full else config["verification"][:1]
    positive_by_prompt = {r["neutral"]: r["positive"] for r in config["rows"]}
    grid = Image.new("RGB", (4 * 256, 50 + len(rows) * len(seeds) * 282), "#151921")
    draw = ImageDraw.Draw(grid)
    for j, title in enumerate(("Off / strength 0", "LoRA / strength 0.5", "LoRA / strength 1", "Frozen positive prompt")):
        draw.text((j * 256 + 10, 18), title, fill="white")
    metadata, times = [], []
    idx = 0
    for row in rows:
        for seed in seeds:
            y = 50 + idx * 282
            draw.text((10, y + 5), f"{row['name']} | seed {seed}", fill="#77d8c1")
            for col, scale in enumerate((0.0, 0.5, 1.0, 0.0)):
                prompt = row["prompt"]
                if col == 3:
                    prompt = row.get("teacher", positive_by_prompt.get(prompt, prompt))
                started = time.perf_counter()
                img = rt.render(prompt, seed, scale, args.sample_steps, args.cfg)
                torch.cuda.synchronize()
                seconds = time.perf_counter() - started
                times.append(seconds)
                name = f"{row['name']}_seed{seed}_" + (f"scale{scale:g}" if col < 3 else "teacher") + ".png"
                img.save(folder / name)
                grid.paste(img, (col * 256, y + 26))
                metadata.append(dict(file=name, name=row["name"], prompt=prompt,
                                     seed=seed, scale=scale, teacher=col == 3,
                                     seconds=seconds, steps=args.sample_steps, cfg=args.cfg))
            idx += 1
            print(f"Rendered {row['name']} seed {seed}", flush=True)
    grid.save(folder / "grid.png")
    write_json(folder / "metadata.json", dict(samples=metadata, width=256, height=256,
                                               mean_seconds_per_image=float(np.mean(times))))
    return folder / "grid.png"


@torch.no_grad()
def evaluate(rt, config, args):
    """Fresh seed and held-out guardian; compare direction as well as magnitude."""
    results = []
    train = config["rows"][0]
    heldout = config["verification"][2]
    cases = [("trained_knight", train["neutral"], train["positive"]),
             ("heldout_bridge", heldout["prompt"], heldout["teacher"]),
             ("fruit_control", config["verification"][3]["prompt"], None)]
    for name, prompt, target in cases:
        _, states = rt.trajectory(prompt, 29001, args.sample_steps, args.cfg, keep=True)
        for i in (0, args.sample_steps // 5, args.sample_steps // 2, args.sample_steps * 4 // 5):
            z, t = states[i].to(rt.device), i / args.sample_steps
            base = rt.velocity(prompt, z, t, cfg=args.cfg)
            pred = rt.velocity(prompt, z, t, scale=1, cfg=args.cfg)
            delta = pred - base
            row = dict(name=name, t=t, relative_drift=float(delta.norm() / base.norm().clamp_min(1e-8)))
            if target:
                teacher = rt.velocity(target, z, t, cfg=args.cfg)
                direction = teacher - base
                row.update(gap_mse=float(F.mse_loss(base, teacher)),
                           trained_mse=float(F.mse_loss(pred, teacher)),
                           direction_cosine=float(F.cosine_similarity(delta.flatten(), direction.flatten(), dim=0)),
                           gap_fraction=float(F.mse_loss(pred, teacher) / F.mse_loss(base, teacher).clamp_min(1e-12)))
            results.append(row)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-legacy-mse", action="store_true", help="Explicit archival MSE reproduction; new training uses train_pg09.py")
    parser.add_argument("--prompts", type=Path, default=ROOT / "configs/supra/prompts-final-boss.yaml")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/final-boss-supra")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--hold-weight", type=float, default=0.1)
    parser.add_argument("--cache-seeds", type=int, default=2)
    parser.add_argument("--sample-steps", type=int, default=50)
    parser.add_argument("--cfg", type=float, default=3.0)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-hub", action="store_true")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if not args.allow_legacy_mse:
        parser.error("Historical MSE trainer requires --allow-legacy-mse; use scripts/train_pg09.py for shared PG0.9 GAN-only training")
    if min(args.steps, args.batch_size, args.cache_seeds, args.sample_steps, args.save_every, args.rank) < 1:
        parser.error("steps, batch, cache seeds, sampling, save interval and rank must be positive")
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    if (out / "run.json").exists() and args.resume is None:
        parser.error(f"{out} already contains a run; select another --out or --resume")
    start = time.perf_counter()
    status = {"phase": "loading", "step": 0, "pid": os.getpid()}

    def update(phase, step=None, **extra):
        status.update(phase=phase, elapsed_seconds=time.perf_counter() - start, **extra)
        if step is not None:
            status["step"] = step
        write_json(out / "status.json", status)

    try:
        update("loading")
        torch.set_num_threads(8)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        config = yaml.safe_load(args.prompts.read_text())
        rt = SupraRuntime(args.device, args.rank, args.allow_hub)
        run = dict(args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                   model_revision=MODEL_REV, text_revision=T5_REV, vae_revision=VAE_REV,
                   base_parameters=rt.base_params, trainable_parameters=sum(p.numel() for p in rt.parameters()),
                   backend_source_sha256=sha(MODEL_SOURCE), prompts_sha256=sha(args.prompts),
                   packages={n: importlib.metadata.version(n) for n in ("torch", "diffusers", "transformers", "safetensors")},
                   recipe="MSE(CFG adapter(neutral, 1), CFG frozen(positive)) on both frozen trajectories; periodic preservation",
                   stored_unconditional=rt.stored_unconditional)
        run["hardware"] = dict(gpu=torch.cuda.get_device_name(rt.device),
                               cuda=torch.version.cuda, cpu_threads=torch.get_num_threads())
        if args.resume:
            previous = json.loads((out / "run.json").read_text())
            for key in ("model_revision", "text_revision", "vae_revision", "backend_source_sha256", "prompts_sha256"):
                if previous[key] != run[key]:
                    raise ValueError(f"Resume provenance changed: {key}")
            for key in ("rank", "batch_size", "cfg", "sample_steps", "seed", "cache_seeds", "lr", "hold_weight"):
                if previous["args"][key] != run["args"][key]:
                    raise ValueError(f"Resume setting changed: {key}")
            write_json(out / "resume.json", run)
        else:
            write_json(out / "run.json", run)
            source = out / "source"
            source.mkdir(exist_ok=True)
            for file in (Path(__file__), ROOT / "supra/runtime.py", args.prompts, MODEL_SOURCE):
                shutil.copy2(file, source / file.name)
        all_prompts = {""}
        for row in config["rows"]:
            all_prompts.update([row["neutral"], row["positive"]])
        all_prompts.update(config["preservation"])
        for row in config["verification"]:
            all_prompts.add(row["prompt"])
            if "teacher" in row:
                all_prompts.add(row["teacher"])
        for prompt in sorted(all_prompts):
            rt.encode(prompt)
        write_json(out / "prompt_token_counts.json", rt.token_counts)
        update("teacher_cache")
        cache_path = out / "teacher_cache.pt"
        if args.resume and cache_path.exists():
            cache = torch.load(cache_path, map_location="cpu", weights_only=True)
            records, holds = cache["records"], cache["holds"]
        else:
            records, holds = build_cache(rt, config, args)
            torch.save(dict(records=records, holds=holds), cache_path)
        # Encoder is frozen, and all prompts are now cached.
        rt.text_encoder.to("cpu")
        torch.cuda.empty_cache()
        opt = torch.optim.AdamW(rt.parameters(), lr=args.lr, weight_decay=0.01)
        rng = torch.Generator().manual_seed(args.seed)
        first_step = 0
        if args.resume:
            rt.load(args.resume / "final-boss-supra.safetensors")
            state = torch.load(args.resume / "training_state.pt", map_location="cpu", weights_only=True)
            opt.load_state_dict(state["optimizer"])
            rng.set_state(state["sampling_rng"])
            torch.set_rng_state(state["torch_rng"])
            torch.cuda.set_rng_state(state["cuda_rng"], rt.device)
            first_step = int(state["step"])
        probe_z = rt.noise(991)
        with torch.no_grad():
            off_before = rt.velocity(config["rows"][0]["neutral"], probe_z, 0.3, cfg=args.cfg).clone()
        update("training", first_step)
        torch.cuda.synchronize(rt.device)
        training_start = time.perf_counter()
        update_seconds = 0.0
        with (out / "train.jsonl").open("a") as log:
            for step in range(first_step + 1, args.steps + 1):
                tick = time.perf_counter()
                is_hold = step % 5 == 0
                pool = holds if is_hold else records
                indices = torch.randint(len(pool), (args.batch_size,), generator=rng).tolist()
                batch = [pool[i] for i in indices]
                z = torch.cat([r["z"] for r in batch]).to(rt.device)
                target = torch.cat([r["target"] for r in batch]).to(rt.device)
                base = torch.cat([r["neutral"] for r in batch]).to(rt.device)
                t = torch.tensor([r["t"] for r in batch], device=rt.device)
                opt.zero_grad(set_to_none=True)
                pred = rt.velocity([r["prompt"] for r in batch], z, t, scale=1, cfg=args.cfg)
                mse = F.mse_loss(pred, target)
                loss = mse * (args.hold_weight if is_hold else 1.0)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Nonfinite loss at step {step}")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(rt.parameters(), 1.0, error_if_nonfinite=True)
                opt.step()
                torch.cuda.synchronize(rt.device)
                row = dict(step=step, hold=is_hold, loss=float(loss), mse=float(mse),
                           teacher_gap=float(F.mse_loss(base, target)),
                           adapter_delta=float(F.mse_loss(pred.detach(), base)), grad_norm=float(norm),
                           seconds=time.perf_counter()-tick,
                           gpu_peak_mb=torch.cuda.max_memory_allocated()/1024**2)
                update_seconds += row["seconds"]
                log.write(json.dumps(row) + "\n")
                log.flush()
                if step % 10 in (0, 1):
                    print(f"Step {step}/{args.steps} loss={row['loss']:.6g} hold={is_hold} "
                          f"grad={row['grad_norm']:.4g} {row['seconds']:.3f}s", flush=True)
                    update("training", step)
                if step % args.save_every == 0 or step == args.steps:
                    folder = out / f"checkpoint-{step:04d}"
                    rt.save(folder / "final-boss-supra.safetensors", {"step": step, "prompts_sha256": run["prompts_sha256"]})
                    torch.save(dict(step=step, optimizer=opt.state_dict(), sampling_rng=rng.get_state(),
                                    torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(rt.device)),
                               folder / "training_state.pt")
                    update("training", step, checkpoint=str(folder))
                if step == args.save_every:
                    update("first_comparison", step)
                    compare(rt, config, args, out / "progress" / f"step-{step:04d}", seeds=(42,), full=False)
                    update("training", step)
        torch.cuda.synchronize(rt.device)
        training_seconds = time.perf_counter() - training_start
        path = out / "final-boss-supra.safetensors"
        rt.save(path, {"step": args.steps, "prompts_sha256": run["prompts_sha256"]})
        with torch.no_grad():
            off_after = rt.velocity(config["rows"][0]["neutral"], probe_z, 0.3, cfg=args.cfg)
            on_before = rt.velocity(config["rows"][0]["neutral"], probe_z, 0.3, scale=1, cfg=args.cfg).clone()
            for p in rt.parameters():
                p.zero_()
            rt.load(path)
            on_reload = rt.velocity(config["rows"][0]["neutral"], probe_z, 0.3, scale=1, cfg=args.cfg)
        validation = dict(scale_zero_exact=torch.equal(off_before, off_after),
                          export_reload_exact=torch.equal(on_before, on_reload),
                          adapter_changes_velocity=bool((on_reload-off_after).abs().max()>0),
                          adapter_sha256=sha(path), adapter_bytes=path.stat().st_size,
                          training_seconds=training_seconds,
                          optimizer_update_seconds=update_seconds,
                          preparation_seconds=training_start-start,
                          timed_updates=args.steps-first_step,
                          timing_scope="CUDA-synchronized updates; training_seconds also includes checkpointing and first preview")
        if not all(validation[k] for k in ("scale_zero_exact", "export_reload_exact", "adapter_changes_velocity")):
            raise RuntimeError(f"Adapter validation failed: {validation}")
        update("verification", args.steps)
        validation["velocity_probes"] = evaluate(rt, config, args)
        write_json(out / "validation.json", validation)
        grid = compare(rt, config, args, out / "samples")
        shutil.copy2(grid, out / "grid.png")
        validation["total_wall_seconds"] = time.perf_counter() - start
        write_json(out / "validation.json", validation)
        update("complete", args.steps, adapter=str(path), grid=str(out / "grid.png"))
        print(f"Done: {path}\nGrid: {out / 'grid.png'}", flush=True)
    except Exception as exc:
        update("failed", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()

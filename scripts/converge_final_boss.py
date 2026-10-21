#!/usr/bin/env python3
"""Continue the existing final-boss LoRA to a measured validation plateau."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageDraw

from supra.convergence import Plateau
from supra.runtime import model_source, SupraRuntime
from train_final_boss import compare, evaluate, sha, write_json

VALIDATION_SEEDS = (39001, 39002)


@torch.no_grad()
def validation_cache(rt, config, sample_steps, cfg):
    records, holds, endpoints = [], [], []
    positions = list(range(0, sample_steps, 5))
    for row in config["rows"]:
        for seed in VALIDATION_SEEDS:
            neutral_z, neutral_states = rt.trajectory(row["neutral"], seed, sample_steps, cfg, keep=True)
            positive_z, positive_states = rt.trajectory(row["positive"], seed, sample_steps, cfg, keep=True)
            endpoints.append(dict(prompt=row["neutral"], seed=seed, name=row["name"],
                                  neutral=neutral_z.cpu(), target=positive_z.cpu()))
            for states in (neutral_states, positive_states):
                for i in positions:
                    z, t = states[i].to(rt.device), i / sample_steps
                    base = rt.velocity(row["neutral"], z, t, cfg=cfg)
                    teacher = rt.velocity(row["positive"], z, t, cfg=cfg)
                    records.append(dict(prompt=row["neutral"], z=z.cpu(), t=t, row=row["name"],
                                        neutral=base.cpu(), target=teacher.cpu()))
        print(f"Validation cache: {row['name']}", flush=True)
    for prompt in config["preservation"]:
        for seed in VALIDATION_SEEDS:
            _, states = rt.trajectory(prompt, seed, sample_steps, cfg, keep=True)
            for i in positions:
                z, t = states[i].to(rt.device), i / sample_steps
                base = rt.velocity(prompt, z, t, cfg=cfg)
                holds.append(dict(prompt=prompt, z=z.cpu(), t=t, neutral=base.cpu(), target=base.cpu()))
    return dict(records=records, holds=holds, endpoints=endpoints, seeds=VALIDATION_SEEDS)


@torch.no_grad()
def velocity_metrics(rt, records, cfg):
    errors, gaps, powers = [], [], []
    for i in range(0, len(records), 8):
        batch = records[i:i+8]
        z = torch.cat([x["z"] for x in batch]).to(rt.device)
        target = torch.cat([x["target"] for x in batch]).to(rt.device)
        base = torch.cat([x["neutral"] for x in batch]).to(rt.device)
        t = torch.tensor([x["t"] for x in batch], device=rt.device)
        pred = rt.velocity([x["prompt"] for x in batch], z, t, scale=1, cfg=cfg)
        errors.extend((pred-target).square().flatten(1).mean(1).cpu().tolist())
        gaps.extend((base-target).square().flatten(1).mean(1).cpu().tolist())
        powers.extend(base.square().flatten(1).mean(1).cpu().tolist())
    return dict(mse=float(np.mean(errors)), gap=float(np.mean(gaps)), power=float(np.mean(powers)))


@torch.no_grad()
def validate(rt, cache, train_records, args):
    edit = velocity_metrics(rt, cache["records"], args.cfg)
    hold = velocity_metrics(rt, cache["holds"], args.cfg)
    train = velocity_metrics(rt, train_records, args.cfg)
    endpoint_results = []
    for row in cache["endpoints"]:
        pred = rt.trajectory(row["prompt"], row["seed"], args.sample_steps, args.cfg, scale=1).cpu()
        mse = float(F.mse_loss(pred, row["target"]))
        gap = float(F.mse_loss(row["neutral"], row["target"]))
        endpoint_results.append(dict(name=row["name"], seed=row["seed"], mse=mse, gap=gap,
                                     ratio=mse / max(gap, 1e-12)))
    endpoint_ratio = float(np.mean([r["ratio"] for r in endpoint_results]))
    velocity_ratio = edit["mse"] / edit["gap"]
    # Endpoint fit matters because the student visits its own trajectory at inference.
    # Preservation coefficient matches the 1-in-5, weight-0.1 training expectation.
    hold_penalty = 0.025 * hold["mse"] / edit["gap"]
    score = 0.5 * (velocity_ratio + endpoint_ratio) + hold_penalty
    return dict(score=score, validation_velocity_ratio=velocity_ratio,
                validation_endpoint_ratio=endpoint_ratio, preservation_penalty=hold_penalty,
                preservation_relative_rms=(hold["mse"] / hold["power"])**0.5,
                training_velocity_ratio=train["mse"] / train["gap"], endpoints=endpoint_results)


@torch.no_grad()
def compare_iterations(rt, config, args, original_adapter, selected_adapter, out):
    rows = config["verification"]
    images = {}
    metadata = []
    for column, adapter, scale in ((0, None, 0), (1, original_adapter, 1), (2, selected_adapter, 1)):
        if adapter:
            rt.load(adapter)
        for row_index, row in enumerate(rows):
            for seed_index, seed in enumerate((42, 1234)):
                image = rt.render(row["prompt"], seed, scale, args.sample_steps, args.cfg)
                index = row_index * 2 + seed_index
                images[index, column] = image
                filename = f"{row['name']}_seed{seed}_column{column}.png"
                image.save(out / filename)
                metadata.append(dict(file=filename, prompt=row["prompt"], seed=seed, scale=scale,
                                     adapter=str(adapter) if adapter else None))
    grid = Image.new("RGB", (768, 50 + 282 * len(rows) * 2), "#151921")
    draw = ImageDraw.Draw(grid)
    for col, label in enumerate(("Base / strength 0", "Original / step 400", "Converged / selected checkpoint")):
        draw.text((col*256+8, 18), label, fill="white")
    for idx in range(len(rows)*2):
        y = 50+idx*282
        draw.text((8, y+5), f"{rows[idx//2]['name']} | seed {(42,1234)[idx%2]}", fill="#77d8c1")
        for col in range(3):
            grid.paste(images[idx, col], (col*256, y+26))
    grid.save(out / "progress_comparison.png")
    write_json(out / "progress_metadata.json", metadata)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--allow-legacy-mse", action="store_true", help="Explicit archival MSE continuation; new training uses train_pg09.py")
    p.add_argument("--source", type=Path, default=ROOT / "outputs/final-boss-supra")
    p.add_argument("--out", type=Path, default=ROOT / "outputs/final-boss-supra-converged")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--eval-every", type=int, default=400)
    p.add_argument("--max-step", type=int, default=0, help="Optional interruption limit; never claims convergence")
    p.add_argument("--resume", type=Path, help="Continuation checkpoint from this script")
    args = p.parse_args()
    if not args.allow_legacy_mse:
        p.error("Historical MSE continuation requires --allow-legacy-mse; use scripts/train_pg09.py for full shared GAN resume")
    MODEL_SOURCE = model_source()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    if (out / "run.json").exists() and not args.resume:
        p.error("Output already contains a run; use --resume or another --out")
    if args.eval_every < 1:
        p.error("--eval-every must be positive")
    start = time.perf_counter()
    status = dict(phase="loading", pid=os.getpid(), step=400)

    def update(phase, **extra):
        status.update(phase=phase, elapsed_seconds=time.perf_counter()-start, **extra)
        write_json(out / "status.json", status)

    try:
        update("loading")
        original = json.loads((args.source / "run.json").read_text())
        settings = original["args"]
        args.sample_steps, args.cfg = settings["sample_steps"], settings["cfg"]
        config = yaml.safe_load(Path(settings["prompts"]).read_text())
        if sha(settings["prompts"]) != original["prompts_sha256"] or sha(MODEL_SOURCE) != original["backend_source_sha256"]:
            raise ValueError("Original prompt/backend provenance changed")
        torch.set_num_threads(8)
        torch.manual_seed(settings["seed"])
        torch.cuda.manual_seed_all(settings["seed"])
        rt = SupraRuntime(args.device, settings["rank"])
        for row in config["rows"]:
            rt.encode(row["neutral"])
            rt.encode(row["positive"])
        for prompt in config["preservation"]:
            rt.encode(prompt)
        for row in config["verification"]:
            rt.encode(row["prompt"])
            if "teacher" in row:
                rt.encode(row["teacher"])
        rt.encode("")
        train_cache = torch.load(args.source / "teacher_cache.pt", map_location="cpu", weights_only=True)
        cache_file = out / "validation_cache.pt"
        update("validation_cache")
        if args.resume:
            cache = torch.load(cache_file, map_location="cpu", weights_only=True)
        else:
            cache = validation_cache(rt, config, args.sample_steps, args.cfg)
            torch.save(cache, cache_file)
        rt.text_encoder.to("cpu")
        torch.cuda.empty_cache()
        opt = torch.optim.AdamW(rt.parameters(), lr=settings["lr"], weight_decay=0.01)
        rng = torch.Generator()
        checkpoint = args.resume or args.source / "checkpoint-0400"
        rt.load(checkpoint / "final-boss-supra.safetensors")
        state = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=True)
        opt.load_state_dict(state["optimizer"])
        rng.set_state(state["sampling_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state(state["cuda_rng"], rt.device)
        step = int(state["step"])
        controller = Plateau(**state["controller"]) if args.resume else Plateau()
        if not args.resume:
            run = dict(source_run=original, source_adapter_sha256=sha(checkpoint / "final-boss-supra.safetensors"),
                       source_step=step, controller=asdict(controller), eval_every=args.eval_every,
                       validation_seeds=VALIDATION_SEEDS,
                       selection="0.5*(velocity residual/gap + mean endpoint residual/gap) + 0.025*hold_mse/velocity_gap",
                       bridge_and_fruit_used_for_selection=False)
            write_json(out / "run.json", run)
            source_dir = out / "source"
            source_dir.mkdir(exist_ok=True)
            for path in (Path(__file__), ROOT / "scripts/train_final_boss.py", ROOT / "supra/runtime.py",
                         ROOT / "supra/convergence.py", Path(settings["prompts"]), MODEL_SOURCE):
                shutil.copy2(path, source_dir / path.name)
        probe = rt.noise(991)
        with torch.no_grad():
            base_before = rt.velocity(config["rows"][0]["neutral"], probe, 0.3, cfg=args.cfg).clone()

        def save_checkpoint(metrics):
            folder = out / f"checkpoint-{step:05d}"
            rt.save(folder / "final-boss-supra.safetensors", {"step": step, "selection_score": metrics["score"],
                    "prompts_sha256": original["prompts_sha256"]})
            torch.save(dict(step=step, optimizer=opt.state_dict(), sampling_rng=rng.get_state(),
                            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(rt.device),
                            controller=asdict(controller)), folder / "training_state.pt")
            write_json(folder / "validation.json", metrics)
            return folder

        def assess():
            update("validation", step=step)
            metrics = validate(rt, cache, train_cache["records"], args)
            action, selected = controller.observe(step, metrics["score"])
            for group in opt.param_groups:
                group["lr"] = controller.lr
            metrics.update(step=step, lr=controller.lr, action=action,
                           selected=selected, plateau_stale=controller.stale,
                           elapsed_seconds=time.perf_counter()-start)
            folder = save_checkpoint(metrics)
            with (out / "validation.jsonl").open("a") as file:
                file.write(json.dumps(metrics)+"\n")
            if selected:
                write_json(out / "best.json", dict(step=step, score=metrics["score"], checkpoint=str(folder)))
            print(f"EVAL step={step} score={metrics['score']:.6f} "
                  f"vel={metrics['validation_velocity_ratio']:.5f} endpoint={metrics['validation_endpoint_ratio']:.5f} "
                  f"hold={metrics['preservation_relative_rms']:.4f} lr={controller.lr:g} "
                  f"stale={controller.stale} action={action} best={controller.best_step}", flush=True)
            update("training", step=step, best_step=controller.best_step, best_score=controller.best,
                   lr=controller.lr, stale=controller.stale, action=action)
            return action

        if not args.resume:
            assess()
        update("training", step=step)
        converged = False
        with (out / "train.jsonl").open("a") as log:
            while True:
                step += 1
                tick = time.perf_counter()
                is_hold = step % 5 == 0
                pool = train_cache["holds" if is_hold else "records"]
                indices = torch.randint(len(pool), (settings["batch_size"],), generator=rng).tolist()
                batch = [pool[i] for i in indices]
                z = torch.cat([r["z"] for r in batch]).to(rt.device)
                target = torch.cat([r["target"] for r in batch]).to(rt.device)
                t = torch.tensor([r["t"] for r in batch], device=rt.device)
                opt.zero_grad(set_to_none=True)
                pred = rt.velocity([r["prompt"] for r in batch], z, t, scale=1, cfg=args.cfg)
                mse = F.mse_loss(pred, target)
                loss = mse * (settings["hold_weight"] if is_hold else 1)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Nonfinite loss at {step}")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(rt.parameters(), 1.0, error_if_nonfinite=True)
                opt.step()
                record = dict(step=step, hold=is_hold, mse=float(mse.detach()), loss=float(loss.detach()),
                              grad_norm=float(norm), lr=controller.lr, seconds=time.perf_counter()-tick)
                log.write(json.dumps(record)+"\n")
                log.flush()
                if step % 100 == 1:
                    print(f"TRAIN step={step} loss={record['loss']:.6f} lr={controller.lr:g}", flush=True)
                    update("training", step=step)
                if step % args.eval_every == 0:
                    converged = assess() == "converged"
                    if converged:
                        break
                if args.max_step and step >= args.max_step:
                    if step % args.eval_every:
                        assess()
                    update("interrupted_at_step_limit", step=step, converged=False)
                    return
        update("exporting_best", step=step, converged=True)
        best = json.loads((out / "best.json").read_text())
        selected_path = Path(best["checkpoint"]) / "final-boss-supra.safetensors"
        rt.load(selected_path)
        export_path = out / "final-boss-supra.safetensors"
        rt.save(export_path, {"step": best["step"], "stopped_step": step, "converged": True,
                             "selection_score": best["score"], "prompts_sha256": original["prompts_sha256"]})
        with torch.no_grad():
            base_after = rt.velocity(config["rows"][0]["neutral"], probe, 0.3, cfg=args.cfg)
            pred_before = rt.velocity(config["rows"][0]["neutral"], probe, 0.3, scale=1, cfg=args.cfg).clone()
            for param in rt.parameters():
                param.zero_()
            rt.load(export_path)
            pred_after = rt.velocity(config["rows"][0]["neutral"], probe, 0.3, scale=1, cfg=args.cfg)
        result = dict(converged=converged, stopped_step=step, selected_step=best["step"],
                      score=best["score"], controller=asdict(controller),
                      scale_zero_exact=torch.equal(base_before, base_after),
                      export_reload_exact=torch.equal(pred_before, pred_after),
                      adapter_sha256=sha(export_path), adapter_bytes=export_path.stat().st_size)
        assert result["scale_zero_exact"] and result["export_reload_exact"], result
        result["heldout_velocity_probes"] = evaluate(rt, config, args)
        write_json(out / "validation.json", result)
        update("final_samples", step=step, selected_step=best["step"])
        grid = compare(rt, config, args, out / "samples")
        shutil.copy2(grid, out / "grid.png")
        compare_iterations(rt, config, args, args.source / "final-boss-supra.safetensors", export_path, out / "samples")
        shutil.copy2(out / "samples/progress_comparison.png", out / "progress_comparison.png")
        update("complete", step=step, selected_step=best["step"], adapter=str(export_path), converged=True)
        print(f"CONVERGED at {step}; selected {best['step']}: {export_path}", flush=True)
    except Exception as exc:
        update("failed", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()

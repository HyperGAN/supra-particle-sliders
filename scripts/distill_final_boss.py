#!/usr/bin/env python3
"""Fit a rank-8 Supra LoRA to a rank-16 teacher; evaluate and render fresh exports."""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import yaml
from safetensors import safe_open
from supra.distillation import compress_lora, projection_error
from supra.runtime import SupraRuntime, load_adapter_state, MODEL_REV, T5_REV, VAE_REV


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2) + "\n")


@torch.no_grad()
def capture(rt, prompts, seeds):
    modules = {n: m for n, m in rt.model.named_modules()
               if hasattr(m, "down") and hasattr(m, "up")}
    cov = {n: torch.zeros(rt.rank, rt.rank, dtype=torch.float64, device=rt.device) for n in modules}
    counts = {n: 0 for n in modules}
    enabled = False
    handles = []
    for name, module in modules.items():
        def hook(mod, args, name=name):
            if not enabled:
                return
            # Evenly spaced tokens cover both conditional and unconditional halves.
            x = args[0].detach().float().reshape(-1, args[0].shape[-1])
            idx = torch.linspace(0, len(x)-1, min(128, len(x)), device=x.device).long()
            with torch.autocast("cuda", enabled=False):
                h = (x[idx] @ mod.down.weight.float().T).double()
                cov[name].add_(h.T @ h)
            counts[name] += len(h)
        handles.append(module.register_forward_pre_hook(hook))
    try:
        for pi, prompt in enumerate(prompts):
            for seed in seeds:
                z = rt.noise(seed)
                for step in range(50):
                    enabled = step % 5 == 0
                    z = z + rt.velocity(prompt, z, step/50, scale=1, cfg=3)/50
            print(f"Captured {pi+1}/{len(prompts)} prompts, seeds={seeds}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    return cov, counts, modules


@torch.no_grad()
def evaluate(teacher, student, rows):
    reports = []
    for row in rows:
        for seed in (59001, 59002):
            end, states = teacher.trajectory(row["prompt"], seed, scale=1, keep=True)
            error, gap = 0., 0.
            for i in range(0, 50, 5):
                z = states[i].to(teacher.device)
                vt = teacher.velocity(row["prompt"], z, i/50, scale=1)
                vs = student.velocity(row["prompt"], z, i/50, scale=1)
                vb = teacher.velocity(row["prompt"], z, i/50, scale=0)
                error += float((vs-vt).square().sum())
                gap += float((vt-vb).square().sum())
            student_end = student.trajectory(row["prompt"], seed, scale=1)
            base_end = teacher.trajectory(row["prompt"], seed, scale=0)
            reports.append(dict(name=row["name"], seed=seed,
                velocity_squared_error=error, teacher_delta_squared_norm=gap,
                velocity_relative_mse=error/max(gap, 1e-20),
                endpoint_relative_mse=float((student_end-end).square().sum() /
                    (end-base_end).square().sum().clamp_min(1e-20))))
        print("Evaluated", row["name"], flush=True)
    return dict(cases=reports,
        velocity_relative_mse=sum(r["velocity_squared_error"] for r in reports) /
                              max(sum(r["teacher_delta_squared_norm"] for r in reports), 1e-20),
        mean_endpoint_relative_mse=sum(r["endpoint_relative_mse"] for r in reports)/len(reports))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--allow-hub", action="store_true")
    p.add_argument("--shared-root", type=Path, help="Reviewed shared checkout (or PARTICLE_SLIDERS_ROOT)")
    args = p.parse_args()
    if args.shared_root:
        os.environ["PARTICLE_SLIDERS_ROOT"] = str(args.shared_root.resolve())
    if args.out.exists() and any(args.out.iterdir()):
        p.error("Choose a new output directory; distillation does not overwrite evidence")
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    torch.set_num_threads(8)
    with safe_open(args.teacher, framework="pt") as f:
        teacher_metadata = {k: json.loads(v) for k, v in f.metadata().items()}
    teacher = SupraRuntime(args.device, teacher_metadata["rank"], args.allow_hub)
    teacher.load(args.teacher)
    teacher.model.requires_grad_(False)
    config = yaml.safe_load((ROOT/"configs/supra/prompts-final-boss.yaml").read_text())
    prompts = [r["neutral"] for r in config["rows"]] + config["preservation"]
    for prompt in prompts + [r["prompt"] for r in config["verification"]]:
        teacher.encode(prompt)
    teacher.text_encoder.to("cpu")
    torch.cuda.synchronize()
    fit_start = time.perf_counter()
    train, counts, modules = capture(teacher, prompts, (49001, 49002))
    state = {}
    with torch.no_grad():
        for name, module in modules.items():
            down, up = compress_lora(module.down.weight, module.up.weight, train[name], args.rank)
            state[name+".down.weight"] = down.cpu().contiguous()
            state[name+".up.weight"] = up.cpu().contiguous()
    torch.cuda.synchronize()
    fit_seconds = time.perf_counter()-fit_start
    # Dev seeds are never used to choose rank, factors or other hyperparameters.
    dev, dev_counts, _ = capture(teacher, prompts, (59001, 59002))
    projections = []
    for name, module in modules.items():
        error, norm = projection_error(module.up.weight,
            state[name+".up.weight"].to(teacher.device), dev[name])
        projections.append(dict(target=name, train_tokens=counts[name], dev_tokens=dev_counts[name],
            squared_error=error, teacher_squared_norm=norm, relative_mse=error/max(norm, 1e-20)))
    student = SupraRuntime(args.device, args.rank, args.allow_hub)
    student.cfg_adapter_branch = teacher.cfg_adapter_branch
    student.cache = teacher.cache
    student.text_encoder.to("cpu")
    load_adapter_state(student.model, state)
    path = args.out/f"final-boss-supra-rank{args.rank}.safetensors"
    student.save(path, dict(teacher_sha256=sha(args.teacher), teacher_step=teacher_metadata.get("step"),
        training_objective=teacher_metadata.get("training_objective", "legacy_mse"),
        **({"shared_lock_sha256": teacher_metadata["shared_lock_sha256"]}
           if teacher.cfg_adapter_branch == "conditional_only" else {}),
        method="activation-weighted reduced-rank regression", calibration_seeds=[49001,49002]))
    probe = teacher.noise(991)
    with torch.no_grad():
        before = student.velocity(prompts[0], probe, .3, scale=1)
        for param in student.parameters():
            param.zero_()
        student.load(path)
        after = student.velocity(prompts[0], probe, .3, scale=1)
        checks = dict(export_reload_exact=torch.equal(before, after), scale_zero_exact=torch.equal(
            student.velocity(prompts[0], probe, .3), teacher.velocity(prompts[0], probe, .3)))
        assert all(checks.values()), checks
    rows = [dict(name=r["name"], prompt=r["neutral"]) for r in config["rows"]]
    rows += config["verification"][2:]
    evaluation = evaluate(teacher, student, rows)
    samples = args.out/"samples"
    samples.mkdir()
    metadata = []
    for row in config["verification"]:
        for seed in (42, 1234):
            for label, rt, strength, adapter in (("original",teacher,1,args.teacher),
                    ("distill",student,1,path), ("off",teacher,0,None)):
                name = f"{row['name']}-seed{seed}-{label}.png"
                rt.render(row["prompt"], seed, scale=strength).save(samples/name)
                item = dict(file=name, name=row["name"], variant=label, prompt=row["prompt"], seed=seed,
                    strength=strength, steps=50, cfg=3, width=256, height=256,
                    model_revision=MODEL_REV, text_encoder_revision=T5_REV, vae_revision=VAE_REV,
                    adapter_sha256=sha(adapter) if adapter else None)
                write((samples/name).with_suffix(".json"), item)
                metadata.append(item)
            print("Rendered", row["name"], seed, flush=True)
    write(samples/"metadata.json", metadata)
    report = dict(teacher_sha256=sha(args.teacher), teacher_metadata=teacher_metadata,
        adapter_sha256=sha(path), adapter_bytes=path.stat().st_size,
        teacher_rank=teacher.rank, rank=args.rank, alpha=args.rank,
        method="activation-weighted reduced-rank regression; fixed rank chosen before dev evaluation",
        calibration_prompts=prompts, calibration_seeds=[49001,49002], dev_seeds=[59001,59002],
        calibration_steps=list(range(0,50,5)), tokens_per_projection_call=128,
        fit_and_capture_seconds=fit_seconds, checks=checks,
        heldout_projection_relative_mse=sum(r["squared_error"] for r in projections) /
            max(sum(r["teacher_squared_norm"] for r in projections), 1e-20),
        projections=projections, evaluation=evaluation, total_seconds=time.perf_counter()-start)
    write(args.out/"evaluation.json", report)
    print(json.dumps({k:v for k,v in report.items() if k not in ("projections","evaluation","calibration_prompts","teacher_metadata")}), flush=True)


if __name__ == "__main__":
    main()

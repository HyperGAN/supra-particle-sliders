#!/usr/bin/env python3
"""Stage measured checkpoints, matched comparisons, provenance and public docs."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
REPO = "ntc-ai/supra-particle-sliders"
HF = f"https://huggingface.co/{REPO}"
RAW = HF + "/resolve/main/"
GITHUB = "https://github.com/HyperGAN/supra-particle-sliders"


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2)+"\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def grid(folder, rows, output, title):
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 15)
    except OSError:
        font = ImageFont.load_default()
    canvas = Image.new("RGB", (768, 62 + len(rows)*284), "#111827")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), title, font=font, fill="white")
    for col, label in enumerate(("Original / rank 16 / strength 1", "Distill / rank 8 / strength 1", "Off / strength 0")):
        draw.text((col*256+8, 36), label, font=font, fill="#a5b4fc")
    for i, (name, seed) in enumerate(rows):
        y = 62+i*284
        draw.text((8, y+4), f"{name.replace('_', ' ')} | seed {seed}", font=font, fill="white")
        for col, variant in enumerate(("original", "distill", "off")):
            canvas.paste(Image.open(folder/f"{name}-seed{seed}-{variant}.png"), (col*256, y+28))
    canvas.save(output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=ROOT/"artifacts/release-v0.1.0")
    args = p.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    benchmark = ROOT/"outputs/final-boss-supra-1600"
    converged = ROOT/"outputs/final-boss-supra-converged"
    validation = read(benchmark/"validation.json")
    run = read(benchmark/"run.json")
    summary = read(converged/"summary.json")
    timing = dict(**{k:validation[k] for k in ("optimizer_update_seconds", "training_seconds",
        "preparation_seconds", "total_wall_seconds", "timed_updates", "timing_scope")},
        hardware=run["hardware"], packages=run["packages"], batch_size=run["args"]["batch_size"],
        rank=run["args"]["rank"], model_revision=run["model_revision"],
        base_parameters=run["base_parameters"], trainable_parameters=run["trainable_parameters"],
        checkpoint_interval=run["args"]["save_every"],
        peak_allocated_mib=max(r["gpu_peak_mb"] for r in
            [json.loads(line) for line in (benchmark/"train.jsonl").read_text().splitlines()]),
        downloads_included=False, cache_generated_in_run=True,
        caveat="One measured run on physical GPU 1; GPU 0 was used by another workload. This is not time to convergence.")
    write(out/"evidence/benchmark-1600.json", timing)
    for src, dest in ((benchmark/"train.jsonl", "benchmark-1600-updates.jsonl"),
            (benchmark/"validation.json", "benchmark-1600-validation.json"),
            (converged/"summary.json", "convergence-summary.json"),
            (converged/"validation.jsonl", "convergence-validation.jsonl")):
        shutil.copy2(src, out/"evidence"/dest)
    write(out/"evidence/benchmark-source-hashes.json",
        {p.name:digest(p) for p in sorted((benchmark/"source").iterdir()) if p.is_file()})
    shutil.copytree(benchmark/"source", out/"evidence/benchmark-source", dirs_exist_ok=True)
    for name in ("LICENSE", "LICENSE-Supra2-IMG", "NOTICE"):
        shutil.copy2(ROOT/"licenses/particle-sliders"/name,
                     out/"evidence/benchmark-source"/("BACKEND-"+name))
    write(out/"evidence/convergence-summary.json", dict(**summary,
        total_wall_seconds=read(converged/"status.json")["elapsed_seconds"]))
    (out/"assets").mkdir(exist_ok=True)
    shutil.copy2(converged/"convergence.png", out/"assets/convergence.png")
    catalog = dict(base_model="SupraLabs/Supra2-IMG", model_revision=run["model_revision"],
        source_repository=GITHUB, timing=timing, format="supra-native-lora-v1", sliders=[])
    for label, folder, step in (("1600", benchmark, 1600), ("converged", converged, summary["selected_step"])):
        distilled = ROOT/f"outputs/final-boss-distill-{label}"
        report = read(distilled/"evaluation.json")
        assert all(report["checks"].values())
        assert report["teacher_sha256"] == digest(folder/"final-boss-supra.safetensors")
        entry = dict(id=f"final-boss-{label}", label=f"Final Boss ({label})", step=step,
            recommended_strength=1, recommended_range=[0,1], original_rank=16, distilled_rank=8,
            original=f"weights/final-boss-{label}.safetensors",
            distill=f"distilled/final-boss-{label}-rank8.safetensors",
            original_sha256=report["teacher_sha256"], distill_sha256=report["adapter_sha256"],
            evaluation=f"evidence/distill-{label}.json", samples=[])
        for src, dst in ((folder/"final-boss-supra.safetensors", entry["original"]),
                (distilled/"final-boss-supra-rank8.safetensors", entry["distill"])):
            target = out/dst
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
            shutil.copy2(src.with_suffix(".json"), target.with_suffix(".json"))
        shutil.copy2(distilled/"evaluation.json", out/entry["evaluation"])
        samples = out/"samples"/label
        samples.mkdir(parents=True, exist_ok=True)
        for item in read(distilled/"samples/metadata.json"):
            for suffix in (".png", ".json"):
                path = Path(item["file"]).with_suffix(suffix)
                shutil.copy2(distilled/"samples"/path, samples/path)
            entry["samples"].append(dict(**item, image=f"samples/{label}/{item['file']}",
                metadata=f"samples/{label}/{Path(item['file']).with_suffix('.json')}"))
        grid(samples, [("knight",42),("cave",42)], out/f"assets/{label}-hero.png",
             f"Final Boss | {step:,} training steps | matched prompts and seeds")
        grid(samples, [("bridge_heldout",42),("fruit_control",42)], out/f"assets/{label}-heldout.png",
             f"{step:,} steps | unseen guardian and fruit control")
        grid(samples, [(n,s) for n in ("knight","cave","bridge_heldout","fruit_control") for s in (42,1234)],
             out/f"assets/{label}-all.png", f"{step:,} steps | every published comparison")
        catalog["sliders"].append(entry)
    write(out/"catalog.json", catalog)
    train = timing["training_seconds"]
    core = timing["optimizer_update_seconds"]
    total = timing["total_wall_seconds"]
    distills = [read(out/e["evaluation"]) for e in catalog["sliders"]]
    body = f'''# Final Boss: 1,600 training steps in {train:.1f} seconds

**A 104.1M-parameter image model. A 6.8 MB LoRA. A 3.4 MB rank-8 distill.**

Measured on one RTX A6000 at native 256×256: **{train:.1f} seconds for 1,600
updates, checkpoint saves and the first preview**. The synchronized optimizer
updates alone took {core:.1f} seconds. Loading cached base weights, creating
teacher targets, training, verification and 32 final images took **{total:.1f}
seconds**. Downloads are excluded. [Timing evidence]({RAW}evidence/benchmark-1600.json).

[Download on Hugging Face]({HF}) · [Code and reproduction]({GITHUB})

## See it

**Original → Distill → Off.** Same neutral prompt, seed, 256×256 resolution,
50 Euler steps and CFG 3. Both adapters use strength 1; Off uses strength 0.
These are ordinary LoRAs. The distill is a fitted rank-8 approximation of the
rank-16 teacher; this release does not use nonlinear particle adapters.

### The 1,600-step result

![1,600-step original, distill and off]({RAW}assets/1600-hero.png)

### The converged result

The longer run selected step **28,000** and stopped at 28,800 under a validation
plateau rule. It took about **47.5 minutes including validation and sampling**.
The speed headline describes the separate 1,600-step run, not convergence.

![Converged original, distill and off]({RAW}assets/converged-hero.png)

### Unseen subject and preservation control

Neither the bridge guardian nor fruit prompt was used to train the original
or fit the distill. The guardian effect is milder; fruit appearance can change.

![Converged held-out guardian and fruit control]({RAW}assets/converged-heldout.png)

All four subjects at both seeds (42 and 1234), without cherry-picking variants:
[1,600 steps]({RAW}assets/1600-all.png) · [converged]({RAW}assets/converged-all.png).
Every original PNG has a JSON sidecar with its exact prompt, seed and adapter hash.

## Get the adapters

| Version | Original rank 16 | Distilled rank 8 |
|---|---|---|
| Fast / 1,600 steps | [6.8 MB LoRA]({RAW}weights/final-boss-1600.safetensors) | [3.4 MB distill]({RAW}distilled/final-boss-1600-rank8.safetensors) |
| Converged / selected step 28,000 | [6.8 MB LoRA]({RAW}weights/final-boss-converged.safetensors) | [3.4 MB distill]({RAW}distilled/final-boss-converged-rank8.safetensors) |

Use the converged original for the closest fit to this recipe; use its distill
for half as many adapter parameters (849,408 versus 1,698,816). The 1,600-step
pair is the exact artifact from the timed run. Base model weights are separate.
The recommended strength range is 0–1. Native loading is verified; ComfyUI
compatibility has not been validated.

## Run it

Install a CUDA-enabled PyTorch build, then:

```bash
git clone {GITHUB}.git
cd supra-particle-sliders
pip install -r requirements.txt
hf download {REPO} distilled/final-boss-converged-rank8.safetensors --local-dir adapters
python scripts/infer_supra.py --allow-hub \\
  --adapter adapters/distilled/final-boss-converged-rank8.safetensors \\
  --prompt "An armored knight holding a sword in a ruined cathedral, full body, game concept art." \\
  --scale 1 --seed 42 --out final-boss.png
```

The first inference downloads pinned Supra2-IMG, Flan-T5 Base and VAE weights.
The inference script reads adapter rank from metadata and checks model pins.
Set `CUDA_VISIBLE_DEVICES` to select a GPU. See the
[training recipe]({GITHUB}/blob/main/docs/final-boss.md) and
[release reproduction]({GITHUB}/blob/main/docs/release.md).

## How it learns

Six matched neutral/final-boss prompt pairs teach the slider to add imposing
silhouettes, dark armor, crowns and oversized weapons while preserving the
subject. Frozen positive-prompt velocities supervise neutral-prompt LoRA
velocities on cached trajectories. Every fifth update preserves a lake,
bicycle or cat. AdamW, batch 4, rank/alpha 16, bf16 forward, fp32 weights.
No training images are supplied. This follows the
[Krea2 final-boss recipe](https://github.com/HyperGAN/krea2-particle-sliders)
and the release format of
[Anima sliders](https://github.com/HyperGAN/anima-particle-sliders).

Distillation uses activation-weighted reduced-rank regression. Rank 8 is fixed
before evaluation. Calibration seeds 49001/49002 and evaluation seeds
59001/59002 are disjoint. The projection fits minimize each teacher branch's
activation error; this is not a new 1,600-step optimizer run.

| Distill teacher | Held-out projection relative MSE | Velocity relative MSE | Mean endpoint relative MSE |
|---|---:|---:|---:|
| 1,600 steps | {distills[0]['heldout_projection_relative_mse']:.4f} | {distills[0]['evaluation']['velocity_relative_mse']:.4f} | {distills[0]['evaluation']['mean_endpoint_relative_mse']:.4f} |
| Converged | {distills[1]['heldout_projection_relative_mse']:.4f} | {distills[1]['evaluation']['velocity_relative_mse']:.4f} | {distills[1]['evaluation']['mean_endpoint_relative_mse']:.4f} |

Projection errors are normalized by the teacher branch output. Velocity and
endpoint errors are normalized by the teacher-minus-base edit. Lower is better;
these measure approximation error, not image quality. The reports include all
per-subject values, including the control where the teacher edit is small.
[1,600-step report]({RAW}evidence/distill-1600.json) ·
[converged report]({RAW}evidence/distill-converged.json).

Both distills pass exact export/reload and strength-zero equality checks.
The convergence run reduced its selection score from 0.2240 at step 400 to
0.0555 at step 28,000. Fresh-seed validation covers all six training subjects;
showcase seeds and the guardian/fruit prompts do not select checkpoints.

![Validation convergence]({RAW}assets/convergence.png)

## Limits and provenance

Supra is a small native 256px model. Fine detail, faces and lettering are limited;
unwanted text can appear in cave images. The original can shift unrelated
objects, and the smaller distill can alter details or weaken the edit. The
timing is one measured run, not a cross-model benchmark or a convergence claim.

The 104.1M count covers the DiT; the frozen text encoder and VAE are additional.
The original adapter trains only 1.70M parameters. Peak allocated PyTorch memory
for the timed run was {timing['peak_allocated_mib']:.0f} MiB; that is not total GPU
memory. Another workload used GPU 0 while this run used GPU 1.

Base: [SupraLabs/Supra2-IMG](https://huggingface.co/SupraLabs/Supra2-IMG), pinned to
`{run['model_revision']}`. Encoder and VAE revisions are embedded in the adapter
metadata. The DiT and ordinary LoRA come from the reviewed external shared
package. Historical licenses remain in `licenses/particle-sliders/`. Source
and adapters are Apache-2.0; the DiT and VAE are separately MIT licensed.
No base weights are redistributed.

[Catalog and sample metadata]({RAW}catalog.json) ·
[Release checksums]({RAW}release-manifest.json) ·
[Source provenance]({RAW}source-provenance.json)
'''
    (ROOT/"README.md").write_text(body)
    frontmatter = "---\nlicense: apache-2.0\nlanguage:\n- en\nbase_model: SupraLabs/Supra2-IMG\npipeline_tag: text-to-image\ntags:\n- lora\n- concept-slider\n- supra\n- distillation\n- text-to-image\n---\n\n"
    (out/"README.md").write_text(frontmatter+body)
    shutil.copy2(ROOT/"LICENSE", out/"LICENSE")
    shutil.copy2(ROOT/"docs/distillation.md", out/"DISTILLATION.md")
    shutil.copy2(ROOT/"configs/supra/prompts-final-boss.yaml", out/"prompts-final-boss.yaml")
    print(json.dumps(dict(folder=str(out), timing=timing, adapters=4), indent=2))


if __name__ == "__main__":
    main()

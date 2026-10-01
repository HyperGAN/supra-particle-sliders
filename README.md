# Final Boss: 1,600 training steps in 134.4 seconds

**A 104.1M-parameter image model. A 6.8 MB LoRA. A 3.4 MB rank-8 distill.**

Measured on one RTX A6000 at native 256×256: **134.4 seconds for 1,600
updates, checkpoint saves and the first preview**. The synchronized optimizer
updates alone took 126.3 seconds. Loading cached base weights, creating
teacher targets, training, verification and 32 final images took **248.7
seconds**. Downloads are excluded. [Timing evidence](https://huggingface.co/ntc-ai/supra-concept-sliders/resolve/main/evidence/benchmark-1600.json).

[Download on Hugging Face](https://huggingface.co/ntc-ai/supra-concept-sliders) · [Code and reproduction](https://github.com/HyperGAN/supra-concept-sliders)

## See it

**Original → Distill → Off.** Same neutral prompt, seed, 256×256 resolution,
50 Euler steps and CFG 3. Both adapters use strength 1; Off uses strength 0.
These are ordinary LoRAs. The distill is a fitted rank-8 approximation of the
rank-16 teacher; this release does not use nonlinear particle adapters.

### The 1,600-step result

![1,600-step original, distill and off](https://huggingface.co/ntc-ai/supra-concept-sliders/resolve/main/assets/1600-hero.png)

### The converged result

The longer run selected step **28,000** and stopped at 28,800 under a validation
plateau rule. It took about **47.5 minutes including validation and sampling**.
The speed headline describes the separate 1,600-step run, not convergence.

![Converged original, distill and off](https://huggingface.co/ntc-ai/supra-concept-sliders/resolve/main/assets/converged-hero.png)

### Unseen subject and preservation control

Neither the bridge guardian nor fruit prompt was used to train the original
or fit the distill. The guardian effect is milder; fruit appearance can change.

![Converged held-out guardian and fruit control](https://huggingface.co/ntc-ai/supra-concept-sliders/resolve/main/assets/converged-heldout.png)

All four subjects at both seeds (42 and 1234), without cherry-picking variants:
[1,600 steps](https://huggingface.co/ntc-ai/supra-concept-sliders/resolve/main/assets/1600-all.png) · [converged](https://huggingface.co/ntc-ai/supra-concept-sliders/resolve/main/assets/converged-all.png).
Every original PNG has a JSON sidecar with its exact prompt, seed and adapter hash.

## Get the adapters

| Version | Original rank 16 | Distilled rank 8 |
|---|---|---|
| Fast / 1,600 steps | [6.8 MB LoRA](https://huggingface.co/ntc-ai/supra-concept-sliders/resolve/main/weights/final-boss-1600.safetensors) | [3.4 MB distill](https://huggingface.co/ntc-ai/supra-concept-sliders/resolve/main/distilled/final-boss-1600-rank8.safetensors) |
| Converged / selected step 28,000 | [6.8 MB LoRA](https://huggingface.co/ntc-ai/supra-concept-sliders/resolve/main/weights/final-boss-converged.safetensors) | [3.4 MB distill](https://huggingface.co/ntc-ai/supra-concept-sliders/resolve/main/distilled/final-boss-converged-rank8.safetensors) |

Use the converged original for the closest fit to this recipe; use its distill
for half as many adapter parameters (849,408 versus 1,698,816). The 1,600-step
pair is the exact artifact from the timed run. Base model weights are separate.
The recommended strength range is 0–1. Native loading is verified; ComfyUI
compatibility has not been validated.

## Run it

Install a CUDA-enabled PyTorch build, then:

```bash
git clone https://github.com/HyperGAN/supra-concept-sliders.git
cd supra-concept-sliders
pip install -r requirements.txt
hf download ntc-ai/supra-concept-sliders distilled/final-boss-converged-rank8.safetensors --local-dir adapters
python scripts/infer_supra.py --allow-hub \
  --adapter adapters/distilled/final-boss-converged-rank8.safetensors \
  --prompt "An armored knight holding a sword in a ruined cathedral, full body, game concept art." \
  --scale 1 --seed 42 --out final-boss.png
```

The first inference downloads pinned Supra2-IMG, Flan-T5 Base and VAE weights.
The inference script reads adapter rank from metadata and checks model pins.
Set `CUDA_VISIBLE_DEVICES` to select a GPU. See the
[training recipe](https://github.com/HyperGAN/supra-concept-sliders/blob/main/docs/final-boss.md) and
[release reproduction](https://github.com/HyperGAN/supra-concept-sliders/blob/main/docs/release.md).

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
| 1,600 steps | 0.0003 | 0.0029 | 0.0008 |
| Converged | 0.0106 | 0.0058 | 0.0116 |

Projection errors are normalized by the teacher branch output. Velocity and
endpoint errors are normalized by the teacher-minus-base edit. Lower is better;
these measure approximation error, not image quality. The reports include all
per-subject values, including the control where the teacher edit is small.
[1,600-step report](https://huggingface.co/ntc-ai/supra-concept-sliders/resolve/main/evidence/distill-1600.json) ·
[converged report](https://huggingface.co/ntc-ai/supra-concept-sliders/resolve/main/evidence/distill-converged.json).

Both distills pass exact export/reload and strength-zero equality checks.
The convergence run reduced its selection score from 0.2240 at step 400 to
0.0555 at step 28,000. Fresh-seed validation covers all six training subjects;
showcase seeds and the guardian/fruit prompts do not select checkpoints.

![Validation convergence](https://huggingface.co/ntc-ai/supra-concept-sliders/resolve/main/assets/convergence.png)

## Limits and provenance

Supra is a small native 256px model. Fine detail, faces and lettering are limited;
unwanted text can appear in cave images. The original can shift unrelated
objects, and the smaller distill can alter details or weaken the edit. The
timing is one measured run, not a cross-model benchmark or a convergence claim.

The 104.1M count covers the DiT; the frozen text encoder and VAE are additional.
The original adapter trains only 1.70M parameters. Peak allocated PyTorch memory
for the timed run was 1541 MiB; that is not total GPU
memory. Another workload used GPU 0 while this run used GPU 1.

Base: [SupraLabs/Supra2-IMG](https://huggingface.co/SupraLabs/Supra2-IMG), pinned to
`10dec6e4b4b5d1c44fd1d7d3fe5e50137333da5b`. Encoder and VAE revisions are embedded in the adapter
metadata. Architecture and LoRA code are vendored from the
[pinned HyperGAN backend](https://github.com/HyperGAN/supra-concept-sliders/blob/main/backend.lock.json), with its license
and upstream attribution. Source and adapters are Apache-2.0; the vendored
backend is MIT and the VAE is separately MIT licensed. No base weights are redistributed.

[Catalog and sample metadata](https://huggingface.co/ntc-ai/supra-concept-sliders/resolve/main/catalog.json) ·
[Release checksums](https://huggingface.co/ntc-ai/supra-concept-sliders/resolve/main/release-manifest.json) ·
[Source provenance](https://huggingface.co/ntc-ai/supra-concept-sliders/resolve/main/source-provenance.json)

The optional ParticleGAN integration provides a shared particle bank across all
71 Supra adapter sites, native E22 game training, exact recovery and versioned
clean exports. Install `requirements-e22.txt` for the tested PR223 source.
Fresh full training uses its shared settled reopening guard and automatic
backend selection; structural changes still use the routed controller.
Historical checkpoints retain their saved profile. See
[the PR223 profile qualification](docs/e22_supra_pr223_profile.md).
The [particle architecture study](docs/e22_supra_particle_architecture.md)
documents the matched 6,400-update results and the live loss dashboard, including
the original LoRA reference curve. The original LoRA still leads this fixture
under both shared game critics; particles improve over the earlier particle
architecture. Output errors remain evaluation only for particle training.

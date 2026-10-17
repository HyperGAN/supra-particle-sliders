# Final boss on Supra2-IMG

The public release includes a separately timed 1,600-step run, the selected
28,000-step checkpoint from validation convergence, and rank-8 distills of
both. See [release reproduction](release.md) and [distillation](distillation.md).
The 400-step measurements below describe the original pilot run.

Run `bash scripts/train_final_boss_gpu1.sh` from this checkout. It uses the
existing `/ml2/ntc-image-studio/.venv-anima/bin/python` environment, physical
GPU 1, and cached weights. Set `SUPRA_PYTHON` to use another environment.
Outputs go to `outputs/final-boss-supra/`. The base models were downloaded
into the existing Hugging Face cache, not the repository.

The recipe follows
[HyperGAN/krea2-particle-sliders](https://github.com/HyperGAN/krea2-particle-sliders)
and its `docs/final-boss.md`. Export provenance, exact strength-zero replay,
and rendering from the reloaded artifact follow the practices in
[HyperGAN/anima-particle-sliders](https://github.com/HyperGAN/anima-particle-sliders).

Six matched pairs cover a knight, robot, sorceress, wolf warrior, cave warrior,
and two-panel knight comic. Supra uses short prose instead of Krea's bounding
box syntax. The 128-token limit is checked before training; there is no silent
truncation. A bridge guardian and fruit bowl are held out from all training.

The DiT has 104,094,736 parameters. Its pinned sources are:

| Component | Repository | Revision |
|---|---|---|
| DiT | `SupraLabs/Supra2-IMG` | `10dec6e4b4b5d1c44fd1d7d3fe5e50137333da5b` |
| Frozen encoder | `google/flan-t5-base` | `7bcac572ce56db69c1ea7c8af255c5d7c9672fc2` |
| Frozen VAE | `stabilityai/sd-vae-ft-mse` | `31f26fdeee1355a5c34592e401dd41e45d25a493` |

Rank 16, alpha 16 LoRA adapts the context projection and self/cross-attention
linears: 1,698,816 trainable parameters. The base and adapter weights are fp32;
forward computation uses bf16 autocast, matching upstream inference. Sampling
is native 256×256, 50 Euler steps, `t=i/50`, CFG 3. The checkpoint's stored
unconditional embeddings are used, as in its official `inference.py`.

Frozen neutral and positive trajectories supply cached states, two seeds per
pair, ten times per trajectory. Each training example compares the adapter's
CFG velocity on the neutral caption against the frozen positive-caption CFG
velocity at the same state and time. Batch size 4, AdamW at 5e-5, 400 updates.
Every fifth update holds a mountain lake, bicycle or cat to its frozen CFG
velocity with weight 0.1. This is the Krea final-boss velocity recipe, rather
than the incomplete Supra backend's advertised dummy trajectory loss.

The official Hub DiT and our wrapped DiT produced exactly equal bf16 forward
outputs at strength zero in the GPU preflight. CPU tests check a real optimizer
update, unchanged frozen parameters, exact zero-strength replay, exact adapter
export/reload, and rejection of incomplete files.

Every 50 updates saves a native LoRA and optimizer/RNG state. Resume using the
same settings and output directory:

```bash
bash scripts/train_final_boss_gpu1.sh \
  --resume outputs/final-boss-supra/checkpoint-0050
```

`status.json` reports the phase. `train.jsonl` records every update, teacher gap,
adapter change, gradient norm, time and peak allocated memory. `run.json`,
`prompt_token_counts.json`, `teacher_cache.pt`, and `source/` preserve provenance.
`validation.json` records strength-zero and reload checks, plus fresh-seed
velocity probes for the trained knight, held-out guardian and fruit control.

At completion `grid.png` compares Off / 0.5 / 1 / frozen positive teacher at
seeds 42 and 1234. The fruit row's teacher column repeats its base image.
`samples/metadata.json` includes exact prompts, seeds and render timings.
Images are native 256px; this architecture has fixed learned positions and is
not a drop-in high-resolution Krea replacement.

## Completed local run

400 updates completed on an RTX A6000: 25.15 seconds including checkpoint saves
and the first preview; the optimizer updates averaged 0.052 seconds each.
The full run took 98.68 seconds, including loading, teacher cache construction,
fresh-seed probes and 32 final images. Final sampling averaged 0.959 seconds
per image. Peak PyTorch allocated memory was 1,528 MiB. The LoRA is 6,810,344
bytes (6.8 MB decimal).

The export reload and scale-zero replay were exact. At a new seed, the trained
knight's velocity error against the teacher was 7–55% of the original gap over
four sampled times; the held-out guardian's was 53–68%. These are diagnostic
velocity errors, not image-quality scores. Visual inspection shows strong
dark armor and spikes on the knight and cave warrior, a weaker bridge-guardian
effect, and mild changes to fruit shape/appearance while it remains fruit.
The model occasionally draws unwanted lettering, visible in the cave row.

## Continue to convergence

`scripts/converge_final_boss.py` restores the original step-400 LoRA, AdamW
state and RNG, preserving the original run. It writes to
`outputs/final-boss-supra-converged/`:

```bash
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  /ml2/ntc-image-studio/.venv-anima/bin/python -u scripts/converge_final_boss.py
```

The training objective, cached training states, batch size and prompts are
unchanged. Every 400 updates, validation uses seeds 39001 and 39002 across all
six pairs (neither seed is in training), along both neutral and positive frozen
trajectories. It also samples each neutral prompt through the complete adapted
50-step trajectory. The endpoint comparison checks the path the student
actually visits, in addition to velocity fitting on cached states.

Checkpoint selection minimizes
`0.5 * (velocity_error/base_gap + mean(endpoint_error/base_gap)) + 0.025 * preservation_MSE/velocity_base_gap`.
The preservation term uses fresh seeds for the lake, bicycle and cat. Its
0.025 coefficient is the original hold-to-edit update weighting, 0.1 / 4.
The bridge guardian, fruit prompt and showcase seeds are excluded from selection.

After at least 2,000 total updates, four validation checks without a 0.5%
improvement halve the learning rate. It can fall from 5e-5 to 3.125e-6. At that
floor, six checks without that improvement end training. The lowest measured
score's checkpoint is exported, even if the final update is later. This is
empirical validation convergence, not a claim of a global optimum. An optional
`--max-step` only interrupts work; it never reports convergence.

Every evaluation saves optimizer/RNG/controller state and writes
`validation.jsonl`. `best.json` identifies the selected checkpoint and
`status.json` reports live progress. Resume with `--resume` pointing to one of
the continuation run's `checkpoint-NNNNN` folders. At the end, `grid.png` shows
the selected slider and `progress_comparison.png` compares the base, step 400
and selected checkpoint. Plot the measurements using:

```bash
.venv-plot/bin/python scripts/plot_convergence.py \
  outputs/final-boss-supra-converged
```

Plotting uses Matplotlib in the separate local `.venv-plot` environment.

## Load the LoRA

`final-boss-supra.safetensors` is an ordinary rank-16 LoRA with `alpha/rank=1`.
Its native keys are `<module>.down.weight` and `<module>.up.weight`. The adjacent
JSON and safetensors metadata identify the architecture, model revision, rank,
alpha and target modules. The included inference script validates and loads
this format. ComfyUI compatibility has not been validated.

```bash
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 \
  /ml2/ntc-image-studio/.venv-anima/bin/python scripts/infer_supra.py \
  --adapter outputs/final-boss-supra/final-boss-supra.safetensors \
  --prompt "A knight in armor, fantasy art." \
  --scale 1 --seed 42 --out outputs/my-final-boss.png
```

For new training, see [pg09.md](pg09.md). All public training aliases execute
the native shared PG0.9 worker. This document records the historical MSE
ordinary-LoRA recipe, whose trainer and continuation now require
`--allow-legacy-mse` explicitly. Architecture and LoRA come from the pinned
external shared package; historical license notices remain in `licenses/`.

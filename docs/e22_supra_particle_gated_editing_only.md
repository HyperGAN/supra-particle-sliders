# Editing-only V3 continuation

The user requested editing-only training after the qualified V3 update-1,600
boundary. The new segment samples only editing contexts, with task weight 1.
It keeps the particle-gated architecture, shared 128×4 bank, 71 routing sites,
native RpGAN/KA2 optimizers, per-site DV12 and zero learned-feature-harm
structural guards. Output metrics remain endpoint diagnostics.

The existing 1,600-update result and checkpoints retain their original
four-edit/one-preservation schedule. The new `editing_only_v1` configuration
explicitly records the schedule change; historical states are restored
exactly before opting in. The next 4,800 updates contain no preservation
training batches.

| Fixed point | Total updates | Editing updates | Preservation updates |
| --- | ---: | ---: | ---: |
| Original ordinary LoRA target | 6,400 | 5,120 | 1,280 |
| Inherited V3 boundary | 1,600 | 1,280 | 320 |
| New V3 matched editing budget | 5,440 | 5,120 | 320 |
| New V3 final endpoint | 6,400 | 6,080 | 320 |

Both new endpoints are declared before training and reported. No validation
metric chooses between them or stops the run. The final endpoint has a larger
editing budget than the original; the 5,440-update endpoint matches its
editing-update count. Both comparisons measure complete formulations,
including the intentionally different application schedule.

Five CPU tests check version rejection, config-only opt-in, fit-only sampling,
parity with the unchanged native game update, paired streams and exact recovery.
An independent historical-control reconstruction verifies all 6,400 original
particle-control rows and the source/receipt/native-boundary links across its
three archived segments. The first long-run attempt stopped before training
because its control trace contained only updates 1,857–6,400; those failed
artifacts remain in `outputs/e22-particle-gated-v3-6400/`.

The new driver checks the actual historical control restore and ordinary-LoRA
reference serving before future training. It then checks the tagged,
CPU-mapped V3 boundary and exact two-update native replay. Fixed-critic probes,
rolling game losses and plateau diagnostics remain read-only. Preservation
scores, when reported, are secondary diagnostics.

```bash
CUDA_VISIBLE_DEVICES=1 /ml2/ntc-image-studio/.venv-anima/bin/python -u \
  scripts/continue_e22_supra_particle_gated_long.py \
  --particlegan-root /ml2/hypergan/ParticleGAN-pr223-supra \
  --output outputs/e22-particle-gated-v3-editing-only-6400 \
  > outputs/e22-particle-gated-v3-editing-only-6400.log 2>&1
```

Live graph: http://pop-os:8783. Qualified 1,600-update graph:
http://pop-os:8782. Use a fresh output path for reproduction. The long run
has no completed outcome yet; it does not establish a new best or SOTA.

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

Completed graph: http://pop-os:8783. Qualified 1,600-update graph:
http://pop-os:8782. Use a fresh output path for reproduction.

The completed run improves on the old particle model at the matched editing
budget, but ordinary LoRA still leads on held-out editing. The longer endpoint
regresses. Both endpoints were fixed before training and remain reported;
the earlier result does not replace the final checkpoint.

Held-out editing game over all 240 test contexts, lower is better. Every model
uses the same paired panels and each of the two frozen common critics.

| Model | Total updates | Editing updates | Common D1,856 | Common V2 D6,400 |
| --- | ---: | ---: | ---: | ---: |
| Original ordinary LoRA | 6,400 | 5,120 | 0.912823 | 0.924374 |
| Historical particle V2 | 6,400 | 5,120 | 0.946709 | 0.956669 |
| Gated particle V3, matched editing budget | 5,440 | 5,120 | 0.925892 | 0.936672 |
| Gated particle V3, final endpoint | 6,400 | 6,080 | 0.964124 | 0.977456 |

At 5,120 editing updates, all six subject means improve versus particle V2
under both critics, while all six remain worse than ordinary LoRA. The fit
pool does beat ordinary LoRA at this endpoint (0.769989 / 0.773821 versus
0.773533 / 0.779163), so the training-panel lead does not establish a
held-out editing win. At the final endpoint, five of six held-out subject
means worsen versus particle V2, and all six worsen versus ordinary LoRA.
Preservation scores remain secondary diagnostics.

The runtime receipt passes all 33,720 checks. The independent CPU review
verifies 43 source snapshots, 39 immutable inputs, all 6,400 recorded
sampling/noise/DV12 streams, both fixed endpoints, 71 routing sites and
428 exported tensors. GPU forwards and gradient/recovery witnesses are
checked through their pinned source and runtime artifacts, rather than
re-executed by the CPU reviewer. The qualified artifacts are
`outputs/e22-particle-gated-v3-editing-only-6400/receipt.json` and
`outputs/e22-particle-gated-v3-editing-only-6400/independent-review.json`.

Particles remain material to the final learned game: removing codes worsens
the paired held-out ablation panel by 0.156707 / 0.149045, and mass-only
routing worsens it by 0.137440 / 0.133090. These are observational game
ablations using their own paired panels, not selection criteria.

A generator settling decision at step 6,222 doubles its learning rate
starting at update 6,223. Training game losses rise before the later
surprise event, so that event cannot explain the entire regression.
A native surprise event records completed step 6,387 with ratio 2.246;
its actions run at the start of update 6,388, leaving 13 updates through
the final endpoint. Final counters show
two surprise fires, one anchor release, one epoch rebase, zero structural
moves, zero controller reopens and zero optimizer restarts. All five
settling owners have reopened twice. The learning-rate change and surprise
actions are concrete leads; timing alone does not establish causation.
An exact replay with isolated interventions at both transitions is the
next diagnostic.

The historical V2 source/profile and sampling schedule differ from the
new run. This compares complete formulations on one fixed fixture and
stream; it does not establish a new best or general superiority.

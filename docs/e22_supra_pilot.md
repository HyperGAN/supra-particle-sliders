# Native Supra E22 verification

For the longer controlled optimizer comparison, see
[Supra recovery](e22_supra_recovery.md), including the primary null result and
the secondary active-R1 recovery test.

This bounded example uses the published `ntc-ai/supra-particle-sliders` rank-16
final-boss LoRA as a frozen teacher, with its pinned Supra2-IMG, T5 and VAE.
The Hub release is an ordinary LoRA. This example replaces only
`blocks.12.cross_attn.proj` and `blocks.13.cross_attn.proj` with nonlinear particle
branches. The other 69 published LoRA branches stay frozen. It verifies a
hybrid integration, not a complete conversion of the released adapter.

Both sites query the same 128-by-4 bank from their current host activation.
Conditional and unconditional CFG uses are token axes of one source context;
they do not double the evidence sample count. Each candidate reruns the full
student model and the full frozen teacher. Frozen model weights remain FP32,
and native CUDA forwards use BF16 autocast.

Fresh trainable networks and the bank use ParticleGAN's public
`init.deterministic_orthogonal_` initializer. The new output projections start
at zero. Training uses native E22 RpGAN and token-unit KA2, with frozen source
embeddings and flow-time conditioning supplied outside penalty coordinates:

```
residual = student_velocity - frozen_teacher_velocity
real = learned_sigma * paired_noise
fake = real + patchify(residual) / training_only_coordinate_scale
```

The policy generates residuals and uses literal zero targets. This representation
allows structural reruns to pair the teacher and student at exactly the same
batch shape. Cached BF16 teacher velocities alone are not exact targets for a
different batch shape: an initial batch-two check exposed rounding differences
larger than the two omitted branches' effect. The cache remains useful for
training-only scale calibration and evaluation; teacher subtraction is live.
Serving and Euler integration reconstruct actual student velocity by adding
the served encoder's matched teacher velocity to the routed residual.

Structural decisions use #155's current clean learned critic-feature proxy
and its default zero feature-harm allowance. Raw output error is evaluation
only. There is no reconstruction loss, raw-output guard, output-based split
selection or metric-based checkpoint selection. This proxy is not established
as a persistent game-repair criterion. Each arm learns its own critic, so its
feature errors cannot be used as a common quality ranking.

The preparation script collects actual frozen teacher trajectories: 12 fit and
12 guard contexts from three prompts, and eight held-out contexts from a new
fantasy prompt and a fruit control. Fit and guard have distinct flow times and
initial latent draws. One ordered data stream is used; there is no seed search.
Three arms share initialized weights, batch indices, paired noise and DV12
streams: fixed bank, movable bank without structural controls, and full E22.

## Reproduce locally

From this checkout, with the existing cached weights and environment:
`requirements-e22.txt` pins the optional ParticleGAN dependency to the tested
#155 head, `f459cb6d6aaaabeb1af076ec53ad7a963618de90` (R1 enabled by default).

```bash
export CUDA_VISIBLE_DEVICES=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=/ml2/hypergan/ParticleGAN-pr155-merge:.
supra_python=/ml2/ntc-image-studio/.venv-anima/bin/python
teacher_adapter=/ml2/hypergan/supra-concept-sliders/outputs/final-boss-supra-converged/final-boss-supra.safetensors
pilot_output=outputs/e22-pilot/verification-f459cb6d
mkdir -p outputs/e22-pilot

$supra_python scripts/prepare_e22_supra.py --adapter "$teacher_adapter" > outputs/e22-pilot/prepare.log 2>&1
$supra_python scripts/verify_e22_supra.py --teacher "$teacher_adapter" --steps 80 --output "$pilot_output" > outputs/e22-pilot/verification-f459cb6d.log 2>&1
$supra_python scripts/render_e22_supra.py --teacher "$teacher_adapter" --mode full --checkpoint "$pilot_output/full-final.pt" --output "$pilot_output/render-full" > outputs/e22-pilot/render-f459cb6d.log 2>&1
```

The verification begins with four updates and a disk checkpoint recovery test.
It compares every policy/model/optimizer/controller/RNG state, checks dense
bank gradients, unchanged frozen student/teacher parameters, and exact
strength-zero inference and 64-context structural-probe behavior. Checkpoints
also own T5 context buffers and bind the dataset identity and representation.
The run saves final checkpoints at the fixed 80-update horizon for all arms.
This short run stays in KA2's A phase; it does not qualify the later blend.

Reuse the existing `data.pt` when comparing optimizer revisions; skip preparation
in that case. Choose a new output directory to retain previous render evidence.

`outputs/e22-pilot/verification-f459cb6d/results.json` contains the current matched-arm receipt
and evaluation metrics. Per-arm trace files include native losses, penalty
phase/calls, gradients, noise digests and structural decisions. Render outputs
include an initial hybrid and trained hybrid image with model, source and
checkpoint hashes. Native base and full-teacher previews are saved beside the
dataset. Tail the JSON logs for progress.

## Interpretation

The optimizer update `f459cb6d` enables R1 by default. It detects sustained abrupt
surprise in gradients relative to Adam's own second moments, reopens stationarity
controls, and can release the KA2 anchor. It adds no output-based optimization or
structural criterion. The same Supra data, teacher, initialization, input streams,
and 80-update horizon were rerun with the new optimizer in a fresh policy.

| Configuration | Original `4325cd87` RMSE | Updated `f459cb6d` RMSE | R1 events | Accepted splits |
| --- | ---: | ---: | ---: | ---: |
| Movable bank | 0.009807 | 0.009807 | 0 | 0 |
| Full E22 | 0.009807 | 0.009807 | 0 | 0 |
| Fixed bank | 0.009827 | 0.009827 | 0 | 0 |

These remain evaluation-only metrics. The updated real-model smoke passed exact
checkpoint resume, dense bank gradients, frozen-weight integrity, and exact
strength-zero inference/probes. All 12 local CPU tests and 11 focused upstream
tests passed. The upstream tests exercised recovery and exact resume after a real
R1 event, including KA2 anchor release. R1 did not fire on this static 80-update
Supra fixture, so the rerun verifies compatibility rather than improved response
to a moving real-model target. KA2 remained in its A phase.

The original receipt remains in `outputs/e22-pilot/verification/results.json`;
the updated receipt and traces are in `verification-f459cb6d`. Native R1 counters
and ratios are recorded in the updated traces and per-arm summaries. Runtime
provenance records the imported optimizer revision rather than a hardcoded pin.

### Original run (`4325cd87`)

The completed A6000 run passed 12 CPU checks and all real-model smoke checks:
exact disk-checkpoint resume, gradients reaching all 128 movable bank rows,
unchanged frozen student/teacher weights, and exact strength-zero behavior at
both inference and 64-context probe sizes. Initialization and training input
digests matched across all arms. Both initial and trained hybrid models rendered
native 256px images with 50 Euler steps and CFG 3.

Evaluation-only leaderboard at the predeclared 80-update horizon:

| Configuration | Held-out velocity RMSE | Missing-branch MSE ratio | Training seconds | Accepted splits |
| --- | ---: | ---: | ---: | ---: |
| Initial hybrid, zero new branches | 0.008438 | 1.0000 | 0 | 0 |
| Movable bank | 0.009807 | 1.3507 | 27.33 | 0 |
| Full E22 | 0.009807 | 1.3507 | 78.95 | 0 |
| Fixed bank | 0.009827 | 1.3563 | 27.40 | 0 |

These output metrics did not select updates, splits, guards or checkpoints.
Training did not improve held-out paired-output accuracy at this budget.
Full E22 proposed splits at updates 60 and 80. Both failed its zero-harm
critic-feature guards; update 60 also worsened the averaged guard score, and
update 80 worsened the fast guard score. There were no birth-eligible parents
at probes 20 and 40. Full E22 therefore offers no accepted structural contrast
against movable here. Keep this fixture as an integration/recovery test; a
longer game study needs accepted structural interventions and paired evidence
after critic adaptation before it can support a game-repair claim.

A successful run establishes real model loading, two-site routing, paired
adversarial updates, dense shared-bank gradients, checkpoint recovery and
native image serving. It does not establish persistent game repair, whole
adapter conversion, or quality improvement. A longer game study should use
predeclared critic-adaptation horizons and paired game evidence; raw output
metrics should remain reporting only.

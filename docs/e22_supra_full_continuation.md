# Fixed-budget continuation to 6,400 updates

The user authorized extending the existing full, 71-site ParticleGAN run from
1,600 to a predeclared 6,400 total updates. This experiment tests whether more
training closes the gap to the original ordinary LoRA recipe. The game,
initialization, prompt task, model pins, sampling stream, structural criterion
and optimizer settings remain the same. Output metrics are evaluation only.
They do not control optimization, structural moves, stopping or checkpoint
selection. There is no seed sweep.

## Artifacts and comparison

Continuation: `outputs/e22-full-f459cb6d-6400/`. The original 1,600-update run
and its evaluation remain at `outputs/e22-full-f459cb6d/`.

The new runner restores the complete native checkpoint, including optimizer,
averages, controller, row evidence, routing reservoir, KA2, R1 and all sampling
RNG states. A digest of every tensor and scalar must exactly match the source
checkpoint before the first continuation update. The existing packed dataset
is copied byte-for-byte; its content digest and all native configuration fields
must match. Only the external update budget and evaluation baseline change.

The matched original baseline is
`/ml2/hypergan/supra-concept-sliders/outputs/final-boss-supra-converged/checkpoint-06400/final-boss-supra.safetensors`,
SHA256 `24a98ba055d0b5298136d67dc7491a339684da03416c510d1d443fe730eb7069`.
Its adapter tensors at step 1,600 exactly match the original standalone
1,600-update run; the teacher cache bytes match as well. The convergence runner
has a validation-based learning-rate controller, but it made no reductions
through step 6,400: all updates used `5e-5`. Its saved CPU sampling RNG matches
the original generator seeded with 7 after 6,400 batch draws.

The evaluation recomputes both models' errors under identical live batch shapes
and uses the same fixed final critic when comparing game losses. Cached original
validation ratios use a different batch shape and are not the comparison score.
The validation pool contains six subjects and two shared initial draws, yielding
24 correlated trajectories; these are not independent training replicates.

## Commands

```bash
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PYTHONPATH=/ml2/hypergan/ParticleGAN-pr155-merge:. \
/ml2/ntc-image-studio/.venv-anima/bin/python scripts/train_e22_supra_full.py \
  --output outputs/e22-full-f459cb6d-6400 --steps 6400 \
  --checkpoint-every 800 --resume outputs/e22-full-f459cb6d/final.pt \
  --baseline /ml2/hypergan/supra-concept-sliders/outputs/final-boss-supra-converged/checkpoint-06400/final-boss-supra.safetensors
```

Intermediate checkpoints are recovery artifacts. The final, fixed 6,400-update
checkpoint is the comparison checkpoint regardless of its output score.

```bash
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PYTHONPATH=/ml2/hypergan/ParticleGAN-pr155-merge:. \
/ml2/ntc-image-studio/.venv-anima/bin/python scripts/evaluate_e22_supra_full.py \
  --data outputs/e22-full-f459cb6d-6400/data.pt \
  --checkpoint outputs/e22-full-f459cb6d-6400/final.pt \
  --output outputs/e22-full-f459cb6d-6400/evaluation \
  --baseline /ml2/hypergan/supra-concept-sliders/outputs/final-boss-supra-converged/checkpoint-06400/final-boss-supra.safetensors
```

## Completed results

All 6,400 updates completed. The continuation added exactly 4,800 updates and
960 preservation batches, with the original CPU sampling sequence and all
recorded checkpoint RNG streams preserved. Frozen model weights are unchanged.
Every continued update had finite game quantities and nonzero gradients on all
128 bank rows. Continuation wall time was 3,525.4 seconds, including preparation
and final saving; evaluation took 593.5 seconds, including a substantial initial
delay from shared-system paging and swap I/O.

| Evaluation, lower is better | Original LoRA, 6,400 | ParticleGAN, 6,400 | New relative difference |
| --- | ---: | ---: | ---: |
| Validation velocity RMSE | 0.0733333 | 0.0860157 | +17.29% |
| Mean latent endpoint error ratio | 0.0921888 | 0.1181670 | +28.18% |
| Preservation relative RMS | 0.0302917 | 0.0453797 | +49.81% |
| Generator loss under the same final learned critic | 0.9318945 | 0.9947221 | +6.74% |

The new model improved its own 1,600-update validation RMSE by 23.18%, endpoint
ratio by 23.51%, and preservation relative RMS by 18.49%. The original also
improved: the new model's relative RMSE disadvantage was 16.58% at 1,600 and
17.29% at 6,400. Longer training helped, but did not close the matched-budget gap.
All six training subjects favor the original on validation MSE. The original
also wins each of the four common-critic paired-noise draws on validation and
preservation. That learned-critic panel is an endogenous game diagnostic, not
independent evidence of general image quality.

All 64 native structural checks ran, with 512 deletion probes, 504 proposals,
48 guard rejections and zero accepted moves. There were no R1 fires or anchor
events. By the final checkpoint, the generator's first native stationarity
decision had halved its learning rate to `2.5e-5`; router and bank rates remained
`5e-5` and `.0085`. The critic's actual rate was `.00293657`. Native KA2 anchor
EMA alpha had reached zero, while fast critic training remained enabled. The
controller's `closed` status reflects its mobility hysteresis threshold, not a
stop command. Native clean serving selected the fast model.

The serving export contains 428 tensors and is 7,611,640 bytes. Real native BF16
reload is exact on four contexts, strength zero exactly reproduces the base,
and evaluation leaves the full native training state unchanged. All 32 images
use the same initial latents within their four-way comparisons. No original
6,400-step published velocity-probe receipt exists; that optional reproduction
check is reported unavailable, while the new matched probes were evaluated.

The original recipe remains the accuracy reference for this bounded task.
Numerical stability is established; an accuracy advantage for the new full
formulation is not. Architecture, initialization, parameter count, game and
optimizer differ, so this experiment cannot isolate ParticleGAN optimizer blame.
The result does not justify adding raw-output MSE to structural decisions.

Artifacts:

- `outputs/e22-full-f459cb6d-6400/evaluation/comparison.json`
- `outputs/e22-full-f459cb6d-6400/evaluation/review.json` (independent CPU review)
- `outputs/e22-full-f459cb6d-6400/continuation-review-final.json`
- `outputs/e22-full-f459cb6d-6400/evaluation/grid.png`
- `outputs/e22-full-f459cb6d-6400/evaluation/budget-comparison.png`
- `outputs/e22-full-f459cb6d-6400/final-boss-particlegan.safetensors`

For execution costs and proposed implementation improvements that preserve the
game, see [the speed investigation](e22_supra_training_speed.md). The concrete
targets are deferred/batched DV12 diagnostics, routing finite checks and exact
live-batch teacher reuse, each requiring separate parity validation before use.

# Supra recovery: previous optimizer behavior versus R1

The real Supra two-site hybrid was compared using ParticleGAN `f459cb6d` with
R1 disabled (`reopen_signal=none`, `reopen_anchor=hold`) and enabled
(`optimizer`, `release`). The disabled path reproduces the previous `4325cd87`
optimizer behavior, already verified bit-for-bit in the original static pilot.
Both runs use the same public API initialization, 128-by-4 shared bank,
rank-16 branches, source/time conditioning, frozen 69-branch published LoRA
background, and exact same data, paired-noise and DV12 streams.

Training is native RpGAN with token-unit KA2. Output errors, recovery curves
and cross-critic game scores are reporting only. They never select updates,
guards, splits, hyperparameters, stopping points or checkpoints. Row proposals
retain #155's clean learned critic-feature criterion and zero feature-harm
allowance. Neither comparison accepted a split.

## Primary controlled reversal

The fitting, guard and held-out contexts come from the cached real Supra
trajectories. After 900 updates, both frozen teachers reversed only the two
LoRA branches the particle model replaces, from multiplier +1 to -1. The
other 69 teacher and student branches remained unchanged. Training continued
for exactly 300 updates. Native KA2 crossed call 800 into its blend phase.

Both complete 1,200-update trajectories were exact ties. All initial and
pre-shift model hashes, input streams, game losses, gradients, evaluation
curves and clean/DV12 critic panels matched. R1's global surprise reached
1.9485 at update 933, below its required greater-than-2 threshold, and never
fired. No optimizer ladder reopens or anchor releases occurred.

| Primary result | Previous behavior | R1 |
| --- | ---: | ---: |
| Final held-out velocity RMSE | 0.00941928 | 0.00941928 |
| Recovery-curve area, 300 updates | 2.818781 | 2.818781 |
| R1 events | 0 | 0 |

This is a null result, retained alongside the secondary comparison.

## Secondary stronger change

After observing the primary null result, one stronger controlled change was
declared. Each arm restored its fixed final checkpoint at update 1200. These
model and optimizer states matched; only R1's diagnostic state differed.
The two teacher branches changed from -1 to +2, and both arms received exactly
300 more updates. Optimizer rules, initialization, random streams, network
learning rates, guard tolerances and structural criteria were unchanged.
This was an activation stress test, not an independently predeclared primary
benchmark. No further target-strength or seed search was performed.

R1 recorded a fire at completed boundary 1217; the first changed training
update was 1218, the 18th update after the switch. All common training values
matched before that update. It reopened five stationarity controls, rescaled
Adam moment memory and released the KA2 anchor. The anchor weight was zero
at reporting points 1250–1400 and returned to one by 1450. No EMA critic
reseed occurred.

| Secondary result | Previous behavior | R1 | R1 versus previous |
| --- | ---: | ---: | ---: |
| Final held-out velocity RMSE | 0.01058264 | 0.01030487 | 2.62% lower |
| Recovery-curve area, 300 updates | 3.225505 | 3.163775 | 1.91% lower |
| Final fitting velocity RMSE | 0.01017364 | 0.00966225 | 5.03% lower |
| Final guard velocity RMSE | 0.00942640 | 0.00907817 | 3.69% lower |
| R1 events / ladder reopens | 0 / 0 | 1 / 5 | |
| Accepted splits | 0 | 0 | |

The new generator also scored better under **both** final critics, using the
same fixed paired-noise tensor across four draws, the same source/time
conditions, and common sigma .125. Lower RpGAN generator loss is better under
each fixed critic; comparisons are within a row, not across critics.

| Fixed critic and generator law | Previous generator loss | New generator loss |
| --- | ---: | ---: |
| Previous critic, clean | 0.928710 | 0.915832 |
| Previous critic, DV12 | 0.929902 | 0.918507 |
| New critic, clean | 0.928216 | 0.914847 |
| New critic, DV12 | 0.929376 | 0.916901 |

All seven matching checks passed: initialized/restored model state,
pre-shift state, input streams, warmup behavior, dataset/teacher provenance,
and teacher schedule. Both frozen student and teacher parameters stayed
unchanged. New and previous arms ran on separate A6000s; bit-for-bit matching
before the event rules out an observed device difference in this comparison.
Training times are not used to claim a speedup on these shared GPUs.

Checkpoint recovery also passed after the real R1 event. Restoring the new
arm's update-1500 checkpoint and repeating updates 1501–1502 reproduced
all 1,752 checkpoint tensors, 1,929 scalar state leaves and both training
traces exactly. The teacher multipliers, optimizer event history and frozen
parameters survived restoration. The receipt is `stronger/resume.json`.

## Practical conclusion

There is a modest recovery benefit when R1 activates on this controlled
Supra teacher change. The unchanged moderate comparison shows no gain.
The benefit came from optimizer reopening and anchor release; neither arm
accepted structural bank moves.

The held-out set contains eight flow-time contexts from only two prompts,
with one shared initialization. These are velocity comparisons at fixed
original trajectory contexts, not a broad generated-image quality study.
Four diagnostic noise draws are not four replicated optimizer runs. Both
final generators still have more held-out velocity error than the partial
frozen background on the stronger target (baseline RMSE 0.00973663), despite
the improvement relative to the previous optimizer.

Keep the primary and secondary receipts together. A larger real training
study should specify representative editing-policy changes and game/recovery
outcomes in advance while retaining output metrics as evaluation only.

## Artifacts and reproduction

`outputs/e22-moving-f459cb6d/comparison.json` is the primary receipt and
`stronger/comparison.json` is the secondary receipt. Each has a
`recovery.png` plot, per-arm final checkpoints, compact critic snapshots,
complete training traces and an independent `review.json`. The scalar teacher
site multiplier belongs to the encoder checkpoint and is changed in both
fast and averaged owners through public checkpoint restoration.

From this checkout, using the existing cache:

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=/ml2/hypergan/ParticleGAN-pr155-merge:.
supra_python=/ml2/ntc-image-studio/.venv-anima/bin/python
plot_python=/home/mikkel/anaconda3/envs/conceptmod/bin/python
teacher_adapter=/ml2/hypergan/supra-concept-sliders/outputs/final-boss-supra-converged/final-boss-supra.safetensors
study_output=outputs/e22-moving-f459cb6d

CUDA_VISIBLE_DEVICES=0 $supra_python scripts/compare_e22_supra_moving.py --arm old --teacher "$teacher_adapter" --output "$study_output"
CUDA_VISIBLE_DEVICES=1 $supra_python scripts/compare_e22_supra_moving.py --arm new --teacher "$teacher_adapter" --output "$study_output"
CUDA_VISIBLE_DEVICES='' $plot_python scripts/compare_e22_supra_moving.py --arm summarize --output "$study_output"

CUDA_VISIBLE_DEVICES=0 $supra_python scripts/compare_e22_supra_moving.py --arm old --teacher "$teacher_adapter" --resume "$study_output/old/final.pt" --warmup 0 --site-scale 2 --output "$study_output/stronger"
CUDA_VISIBLE_DEVICES=1 $supra_python scripts/compare_e22_supra_moving.py --arm new --teacher "$teacher_adapter" --resume "$study_output/new/final.pt" --warmup 0 --site-scale 2 --output "$study_output/stronger"
CUDA_VISIBLE_DEVICES='' $plot_python scripts/compare_e22_supra_moving.py --arm summarize --output "$study_output/stronger"
CUDA_VISIBLE_DEVICES=1 $supra_python scripts/verify_e22_supra_recovery_checkpoint.py --teacher "$teacher_adapter" --checkpoint "$study_output/stronger/new/final.pt" --output "$study_output/stronger/resume.json"
```

Use a fresh `study_output` to preserve completed evidence. The optional
`requirements-e22.txt` includes matplotlib for plotting in a single environment.

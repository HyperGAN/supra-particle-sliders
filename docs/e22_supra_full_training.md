# Full Supra final-boss training

The completed full ParticleGAN run learned the editing task, but did not beat
the original published training recipe at the matched 1,600-update budget.
Fresh-trajectory velocity RMSE was 16.58% higher; mean normalized complete-path
endpoint error was 19.81% higher. Preservation drift also increased.

| Evaluation only, lower is better | Original LoRA | Full ParticleGAN |
| --- | ---: | ---: |
| Fresh-trajectory velocity RMSE, 240 contexts | 0.096037 | 0.111964 |
| Velocity MSE / base-to-target MSE | 0.093290 | 0.126799 |
| Mean endpoint error / base-to-target error, 12 paths | 0.128940 | 0.154482 |
| Preservation RMS / base RMS, 60 contexts | 4.111% | 5.568% |
| Edit RpGAN generator loss under the fixed new critic | 1.068520 | 1.145125 |
| Preservation RpGAN loss under the fixed new critic | 0.733009 | 0.767850 |

The common critic panel uses identical paired noise across four private draws
for both generators and scores native RpGAN under the same final critic.
That critic also favors the original adapter. It was trained with the new
generator, so this panel is not independent image-quality evidence. The
original has no adversarial critic of its own.

The original trained-knight and unseen-guardian velocity probes favor the old
adapter at all four measured times. Mean fruit relative velocity drift was
3.709% for the original and 6.358% for the new model. The original published
probe receipt was reproduced to a maximum scalar difference of `1.77e-8`.

All 1,600 batch selections and 320 preservation updates followed the original
input schedule. No splits were accepted, no R1 event fired and no stationarity
ladder reopened. This stationary task does not establish structural or R1
benefit. Frozen parameters stayed unchanged. Thirty-six CPU tests, the native
GPU exact-resume preflight, exact strength-zero checks and exact clean adapter
export/reload all passed. Final evaluation left the complete training state
unchanged and retained all 32 matched showcase renders.
The standalone inference command also reproduced the saved knight render
exactly across all 196,608 image-channel values.

The native clean served adapter is **7,610,992 bytes**, with 428 tensors. Its
SHA-256 is `ad56898067d0a24853b59b994e8f9eaa1af20ac55d71e72948f0d5f3a536d1a7`.
Training and fixed intermediate checkpoint saves took 1,095.5 seconds on the
shared A6000, including the resume preflight; these timings do not establish
an isolated speed comparison with the original release.

Keep the original published slider as the quality reference. The new adapter
is usable for inspection and further game-dynamics research, but this result
does not justify replacing the original examples. A subsequent study should
predeclare its larger training budget and compare native game convergence;
retain output metrics as evaluation only. Architecture, initialization and
objective differences prevent attributing this gap solely to the optimizer.

The complete results are in
`outputs/e22-full-f459cb6d/evaluation/comparison.json`; matched images are in
`evaluation/grid.png` and `evaluation/samples/`. The lean adapter is
`outputs/e22-full-f459cb6d/final-boss-particlegan.safetensors`.

This comparison trains all 71 native Supra adapter sites from the pure base
model, using ParticleGAN `f459cb6d` and the original published final-boss task.
No trained LoRA branches are inherited. The reference is the published
1,600-update ordinary rank-16 LoRA, not the separate selected 28,000-update
convergence checkpoint.

Both systems use the same pinned 104.1M Supra DiT, T5, VAE, six neutral/positive
prompt pairs, cached trajectory states, batch size four and fixed 1,600-update
horizon. Sampling uses the original CPU generator and ordered record pools.
Every fifth update uses lake/bicycle/cat preservation contexts.

The new adapter uses 71 rank-16 nonlinear branches, activation-dependent
queries and one shared 128-by-4 bank. Branches, router and bank use the public
`particlegan.init.deterministic_orthogonal_` API; output projections start at
zero. There are 1,887,820 trainable generator/router/bank parameters, versus
1,698,816 ordinary LoRA parameters. The comparison therefore tests the complete
new formulation against the original training recipe, not an optimizer-only
ablation or an equal-time/parameter comparison.

## Training rules

The frozen pure-base model with the positive caption supplies the live target
velocity at the same batch shape and CFG as the student. Preservation uses an
identical source and target caption. Native paired-error RpGAN and token-unit
KA2 train the new adapter; DV12, stationarity control, learned noise and R1 use
the public E22 policy lifecycle. Branch and router rates are the original
application rate, `5e-5`; critic/table rates follow the native recipe.

Preservation multiplies the native generator and discriminator game terms by
0.1. KA2 penalty units and coefficient remain unchanged. This carries the
original preservation schedule and relative game weight into the adversarial
formulation; it does not make the objectives equivalent.

Structural probes run every 100 updates using full-model reruns, current clean
learned critic features, zero feature-harm allowance and separate training-only
midpoint guard contexts. No validation contexts enter the structural guards.
Feature-based decisions do not establish persistent game repair by themselves.
Text-site usage includes padded tokens; the native downstream attention mask
still governs their actual effect in complete candidate reruns.

Output MSE, endpoint error and renders are evaluation only. They do not drive
optimization, structural guards, learning rates, stopping or checkpoint choice.
The final checkpoint is fixed before observing quality metrics.

## Verification and artifacts

The GPU preflight checks gradients at all 71 sites and all 128 bank rows,
exact disk-checkpoint resume, unchanged frozen parameters and exact native
strength-zero inference. The training trace retains every update, input
indices, game losses, noise-stream digests, structural decisions and R1 events.

`outputs/e22-full-f459cb6d/` contains the original-input provenance in
`run.json`, packed data in `data.pt`, live progress in `status.json`, the
complete `train.jsonl`, fixed intermediate checkpoints and `final.pt`.
The public clean served model can be exported separately without frozen model
weights or optimizer state.

Final evaluation compares both trained systems against live frozen targets
on the same independent validation trajectories, reports preservation drift
and full 50-step trajectory endpoint errors, and repeats the original trained
knight, held-out guardian and fruit probes. All original showcase prompts and
latent draws are retained; images do not select the checkpoint.

From this checkout with the existing caches:

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=/ml2/hypergan/ParticleGAN-pr155-merge:.
supra_python=/ml2/ntc-image-studio/.venv-anima/bin/python
study_output=outputs/e22-full-f459cb6d
CUDA_VISIBLE_DEVICES=1 $supra_python -u scripts/train_e22_supra_full.py --output "$study_output" > "$study_output.log" 2>&1
tail -f "$study_output.log"
CUDA_VISIBLE_DEVICES=1 $supra_python -u scripts/evaluate_e22_supra_full.py --data "$study_output/data.pt" --checkpoint "$study_output/final.pt" --output "$study_output/evaluation" > "$study_output-evaluation.log" 2>&1
```

Use a fresh output directory to retain completed evidence. Training can resume
with the same arguments and `--resume` pointing to a saved checkpoint.

The evaluator exports `final-boss-particlegan.safetensors` and checks its loaded
clean velocity and strength-zero behavior against the native serving snapshot.
For another prompt using the exported adapter:

```bash
CUDA_VISIBLE_DEVICES=1 $supra_python scripts/infer_e22_supra_full.py \
  --adapter "$study_output/final-boss-particlegan.safetensors" \
  --prompt "An armored knight holding a sword in a ruined cathedral, full body, game concept art." \
  --seed 42 --scale 1 --out "$study_output/knight.png"
```

The clean inference export contains the nonlinear branches, router and bank,
with model pins and native serving provenance. It requires this loader;
ordinary-LoRA loaders do not implement the shared particle routing.

## Saved-checkpoint progress diagnostic

A subsequent read-only diagnostic evaluated all four existing fixed checkpoints
on the same 240 validation and 60 preservation contexts. It also held the final
critic and four private paired-noise draws fixed across checkpoints, so the
game trend does not merely reflect changing critic parameters. No training,
structural decision or checkpoint selection occurred.

| Stored update | Validation velocity RMSE | Preservation RMSE | G loss under fixed final critic |
| --- | ---: | ---: | ---: |
| 400 | 0.193242 | 0.095537 | 1.669925 |
| 800 | 0.133983 | 0.072640 | 1.288898 |
| 1,200 | 0.120212 | 0.065905 | 1.192289 |
| 1,600 | 0.111964 | 0.063144 | 1.144743 |

Over the last 400 updates, validation RMSE fell 6.86% and the fixed-critic
generator loss fell 3.99%. The last checkpoint reproduced the completed
evaluation exactly. Counter and RNG checks passed at every checkpoint. These
contexts cover six training subjects on fresh cached trajectories, rather
than 24 unseen subjects or independent optimizer replications.

The generator, router and bank stayed at their full learning-rate scales and
never received a native stationary verdict. Their parameters still changed
by 4.61%, 8.23% and 10.31% of their previous norms from 1,200 to 1,600. These
measurements support continued training as the next diagnostic; they do not
prove eventual catch-up or rule out weaker critic/game choices. The original
recipe also continued improving beyond 1,600, reaching its selected checkpoint
at 28,000.

The receipts are `progress-diagnostic.json`, `progress-diagnostic.png` and
`optimizer-progress.json` under the run directory. A longer comparison should
predeclare its horizon, retain the same native game and data stream, and use
the original checkpoint at that same horizon as its reference. Output metrics
remain reporting only.

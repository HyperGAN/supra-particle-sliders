# Particle-gated V3 at 1,600 updates

The qualified V3 run improves editing over ordinary LoRA at the same 1,600
updates, under both common learned critics. Ordinary LoRA still preserves
better, and its declared 6,400-update target still leads every pool. V3
improves every complete pool over the matched V2 particle control.

This is the fixed continuation of the [qualified 400-update architecture
experiment](e22_supra_particle_formulation_audit.md). The particle law is
`up(h + tanh(Hh+b) + h*tanh(Cz))`: native routed particle codes modulate the
input-dependent basis. The shared 128×4 particle bank, all 71 routing sites,
native per-site DV12, game loss, optimizer law and structural guards remain.
Output error is an endpoint diagnostic; it never chooses an update,
structural move, hyperparameter, horizon or checkpoint.

## Complete endpoint comparisons

All rows use the same 570 cached contexts, BF16 host, CFG3, batches of four,
and four paired Gaussian panels per context at output sigma 0.125. Two frozen
V2 critics judge **every** endpoint: the existing D1,856 and the qualified
shared-profile control's D1,600. Neither arm substitutes its own differing
critic in this comparison. Lower generator game loss is better.

The complete D1,856 leaderboard is:

| Fixed endpoint | Fit (240) | Test (240) | Training preservation (30) | Held-out preservation (60) |
| --- | ---: | ---: | ---: | ---: |
| V2 particles, 1,600 | 0.974589 | 1.055327 | 0.743877 | 0.747481 |
| Gated V3 particles, 1,600 | 0.908232 | 1.012091 | 0.732075 | 0.737606 |
| Ordinary LoRA, 1,600 | 0.952320 | 1.037191 | 0.727606 | 0.731798 |
| Declared ordinary LoRA target, 6,400 | 0.773533 | 0.912823 | 0.702649 | 0.710419 |

The two judges agree on the comparison directions:

| V3 minus reference | Test, D1,856 | Test, D1,600 | Held-out preservation, D1,856 | Held-out preservation, D1,600 |
| --- | ---: | ---: | ---: | ---: |
| Matched V2 particles at 1,600 | −0.043236 | −0.047694 | −0.009875 | −0.009287 |
| Ordinary LoRA at 1,600 | −0.025100 | −0.029369 | +0.005808 | +0.005307 |
| Ordinary LoRA target at 6,400 | +0.099268 | +0.109015 | +0.027187 | +0.024641 |

Against V2, all six editing subject means and all three preservation subject
means improve under both critics. Test improves on 238/240 contexts under
each; held-out preservation improves on 57/60 and 58/60. Against ordinary
LoRA at 1,600, all six editing subject means improve, with 200/240 and
208/240 test contexts improved. All three preservation subject means worsen;
50/60 and 49/60 held-out preservation contexts worsen. The larger-budget
ordinary target wins every recorded context under both critics.

The output diagnostics agree with this editing/preservation tradeoff:

| Fixed endpoint | Test RMSE, evaluation only | Held-out preservation RMSE, evaluation only |
| --- | ---: | ---: |
| V2 particles, 1,600 | 0.102711 | 0.055222 |
| Gated V3 particles, 1,600 | 0.094100 | 0.050709 |
| Ordinary LoRA, 1,600 | 0.096037 | 0.046628 |
| Ordinary LoRA target, 6,400 | 0.073333 | 0.034355 |

## The three declared ordinary artifacts

The endpoint evaluates all three previously declared files. CPU review finds
that the two 1,600-update files contain **bitwise identical values in all 142
adapter tensors**. Their tensor digest is
`b2f5a0b7e1d0c3cdc554cd719d0e358dc367a7be36c20c15e07eb6a0ba4be766`.
Their distinct file hashes reflect metadata: the converged-path artifact
adds a stored `selection_score`. The files do not provide two independent
trained-model controls; their complete evaluations are exactly identical.
The stored selection score is not used in this experiment.

| Artifact label | Declared path under `supra-concept-sliders/outputs/` | File SHA-256 |
| --- | --- | --- |
| Converged-path 1,600 | `final-boss-supra-converged/checkpoint-01600/final-boss-supra.safetensors` | `0c97f2f2fd05bf17be99c901a87a05447f0448ad6edae06ecba4e5d623fed997` |
| Previously published 1,600, identical tensors | `final-boss-supra-1600/final-boss-supra.safetensors` | `134f5a9d12206b74f39a2c3f9c90fdb38477a1c3204b5deea5152b1cbe49a47f` |
| Declared converged 6,400 target | `final-boss-supra-converged/checkpoint-06400/final-boss-supra.safetensors` | `24a98ba055d0b5298136d67dc7491a339684da03416c510d1d443fe730eb7069` |

Each original loads into a separate pure-base host, with all 142 tensor
names/shapes/dtypes and task/model metadata checked. The particle teacher
and training state stay unchanged. As an additional baseline-glue check,
all 570 new original-6,400 records under D1,856, teacher diagnostics and
pool aggregates exactly reproduce the previous original-6,400 evaluator.
That crosscheck excludes the intentionally different second judge.

## What the particles contribute

Code and routing ablations cover all 570 contexts with the same paired
panels and common critics. The table gives ablated minus normal V3 game:

| Evaluation-only ablation | Test, D1,856 | Test, D1,600 | Held-out preservation, D1,856 | Held-out preservation, D1,600 |
| --- | ---: | ---: | ---: | ---: |
| Zero particle codes | +0.147010 | +0.162166 | −0.007631 | −0.006845 |
| Mass-only routing | +0.096606 | +0.105861 | −0.004079 | −0.003611 |

The particle path now provides a substantial editing benefit. It still
adds a preservation cost within V3, consistent with ordinary LoRA's better
preservation endpoint. These ablations diagnose the trained formulation;
they do not alter training or select a model. All 71 native site
interventions have nonzero final-output effects at 1,600; actual clean and
DV12 game gradients reach all 128 bank rows, 142 router tensors and 284
generator tensors.

## Native control activity and qualification

The native policy observes one surprise fire at update 822
(`surprise.log = [[822, 2.905]]`) and one epoch rebase. Each LR-settling owner
records one reopen and zero monitor restarts. There are zero controller
reopens, anchor-release events or accepted structural moves. Generator and
router raw settling scales finish at 0.5, table/noise at 1.0 and critic at
0.0078125. Raw scales are distinct from the applied learning rates, which
also retain native support floors and payoff damping.

“Same game” here means the same algorithm and lifecycle. Changing the
architecture changes learned observations and therefore native control
decisions. Native FAST-plus-averaged, per-context zero learned-feature-harm
guards remain enabled and raw-output guards remain disabled. No structural
move is accepted; the structural guard does not promise zero harm from
continuous parameter updates or an overall preservation win.

GPU qualification passes 9,250 checks. Independent CPU artifact review
verifies 40 application/native source snapshots, 17 immutable inputs,
all 800/1,200/1,600 checkpoint frozen owners, the unchanged 400-update prefix,
all 1,600 sampled/paired/DV12 records, all endpoint/ablation reductions and
all 428 explicitly versioned clean-export tensors. The CPU-mapped boundary
restores strictly and replays updates 401–402 exactly in rows and complete
native state. Evaluation, baseline loading and clean export/reload preserve
the complete native state and RNGs. Actual CUDA forwards, gradients and
update replay are executed source/runtime witnesses; CPU review does not
rerun them.

The continuation's measured training loop takes 680.91 seconds; the complete
continuation/diagnostic job takes 868.64 seconds on an RTX A6000. The earlier
400-update job is separate. These times are not a matched wall-clock
benchmark against ordinary LoRA or V2.

Artifacts are in `outputs/e22-particle-gated-v3-1600/`: `receipt.json`,
`independent-review.json`, `original-6400-history-crosscheck.json`, all five
`evaluation-*.json` files, `particle-contribution.json`, `audit-final.json`,
native checkpoints and `final.safetensors`. There are five endpoint files
because the two declared 1,600 ordinary artifacts are both evaluated.

## Reproduction and scope

ParticleGAN is pinned to `bdf05d1be0f68cfdb0c71e81e7e0d3cce477572f`; both
particle arms use `pr223_shared_routed_v1`. V3 continues its qualified
fresh-from-zero 400-update state. No old trained V2 bank/model/optimizer is
migrated into V3. The initial experiment proves an exact fresh V2 prefix
and identical native V2/V3 step-zero tensors before the new formula learns.

```bash
CUDA_VISIBLE_DEVICES=1 /ml2/ntc-image-studio/.venv-anima/bin/python \
  scripts/continue_e22_supra_particle_gated.py \
  --particlegan-root /ml2/hypergan/ParticleGAN-pr223-supra \
  --initial outputs/e22-particle-gated-v3-400 \
  --steps 1600 --control outputs/e22-pr223-shared-1600 \
  --output outputs/e22-particle-gated-v3-1600 \
  > outputs/e22-particle-gated-v3-1600.log 2>&1

CUDA_VISIBLE_DEVICES='' /ml2/ntc-image-studio/.venv-anima/bin/python \
  scripts/review_e22_supra_particle_gated_continuation.py \
  --run outputs/e22-particle-gated-v3-1600 \
  --particlegan-root /ml2/hypergan/ParticleGAN-pr223-supra
```

Choose fresh output paths for reproduction. The reviewer preserves an
existing review; `--output` can name a fresh second review artifact.

This is one task and one matched particle stream. V3 versus V2 isolates
the architecture intervention within the same native algorithm, including
its resulting adaptive responses. Ordinary LoRA uses its historical paired
velocity-MSE/AdamW formulation, so the ordinary comparison measures complete
formulations rather than isolating the optimizer. The 6,400 ordinary model
is an explicit larger-budget target, not a matched 1,600-update control.
The user subsequently made editing the primary goal and requested
editing-only training for the next segment. This completed run retains its
original schedule and evidence. The next versioned continuation will report
its changed schedule, the predeclared 5,440-total-update checkpoint matching
the original's 5,120 editing updates, and the fixed 6,400-total-update
endpoint. Preservation remains a secondary diagnostic rather than a
requirement for an editing win.

# PR223 shared controls: matched Supra comparison

The PR223 shared routed profile **ties the previous ParticleGAN V2 exactly
after 1,600 updates on the real Supra task**. Both common-critic game scores
and diagnostic output RMSE are identical on every complete evaluation pool.
The saved generator, critic, router, particle bank, averages, optimizers,
DV12 controller, routed evidence, stationarity tests and native RNG streams
are also identical. This run shows compatibility and active guard handling;
it provides no convergence or accuracy improvement over that old endpoint.

The comparator is the qualified `linear_modulated_v2` ParticleGAN run at
`f459cb6d6aaaabeb1af076ec53ad7a963618de90`, in
`outputs/e22-particle-v2-1600`. The new run uses
`bdf05d1be0f68cfdb0c71e81e7e0d3cce477572f` and
`pr223_shared_routed_v1`. The original ordinary LoRA is a separate benchmark;
this comparison does not establish a new win against it.

## Endpoint leaderboard

The native paired RpGAN generator game is evaluated under two critics common
to both endpoints: the frozen V2 `cabe` critic from update 1,856, and the new
PR223 final critic from update 1,600. The final critic's parameters match the
old final critic exactly. Lower game scores are reported under each fixed
judge. RMSE is a diagnostic and never enters optimization or selection.

| Pool | Contexts | Old G game, D1856 | Shared G game, D1856 | Old G game, D1600 | Shared G game, D1600 | Old RMSE | Shared RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fit | 240 | 0.974589 | 0.974589 | 1.000809 | 1.000809 | 0.087904 | 0.087904 |
| Test | 240 | 1.055327 | 1.055327 | 1.090568 | 1.090568 | 0.102711 | 0.102711 |
| Training holds | 30 | 0.743877 | 0.743877 | 0.738754 | 0.738754 | 0.053258 | 0.053258 |
| Preservation evaluation | 60 | 0.747481 | 0.747481 | 0.742317 | 0.742317 | 0.055222 | 0.055222 |

Every unrounded new-minus-old difference is exactly zero. Test game scores
are `1.0553267267843087` and `1.0905679933726788`; test RMSE is
`0.10271051636240713`. Evaluation uses the complete fixed pools, clean native
BF16/CFG3 forwards in batches of four, and common four-panel paired Gaussian
noise at sigma `.125`. Progress curves additionally report privately replayed
DV12 forwards. Neither evaluation path advances native training state.

## What the shared controls actually did

`birth_death_backend="auto"` selects `actual_backend="routed"`, with reason
`routed_rows_owns_controls`. Its generator/noise rate factor remains 1. The
128×4 bank, all 71 sequential routing sites, native per-site DV12, whole-model
candidate reruns, learned critic-feature structural criterion, and zero
per-context feature-harm guard stay active. Atlas's independent feature-cell
counting, placement, birth and rate law do not activate in this adaptation.

The settled re-open guard was exercised:

| Actual training update | Observed event |
| ---: | --- |
| 343 | A full-rate excursion is observed without a contracted network witness. |
| 344 | Surprise reference/ratio telemetry first differs from the old trace; native loss, gradient and stream fields stay exact. |
| 800 | The actual KA2 anchor starts its blended phase. The guard rebases the detector once without resetting model or optimizer state. |
| 803 | A new excursion starts without a contraction witness; its native completed-update clock is 802. |
| 835 | That excursion reaches ratio `3.273568547797319`, while all network ladders are at full scale. |
| 976 | The detector returns below the calm threshold `1.25`. |
| 1531 | The critic's first stationary verdict contracts its native tester scale from 1 to `.5`. |
| 1532 | A calm observation acquires the critic `.5` contraction witness. |
| 1600 | One epoch rebase, zero optimizer re-opens, zero anchor-release events, zero accepted row moves, and no generator/router restarts. |

There are 77 recorded updates with ratio above 2 while the guard lacks a
contracted witness. The detector follows its slow reference instead of
allowing those signals to reopen the optimizers. This is active observation,
reference rebasing and gating, even though no optimizer re-open is accepted.
The old run also had zero optimizer re-opens and zero accepted moves.

The critic's raw tester scale is the guard witness. Its applied learning rate
still uses the native `.75 × table-scale` floor before payoff damping; the
tester reaching `.5` does not mean the applied critic rate halves. G, router
and table remain at scale 1. The saved final applied critic LR/base-rate ratio
is `0.5323137224`, including the floor and payoff damping. The learned-noise
floor therefore remains at `.125`; the previous noise-floor fix has no
numerical effect on this path.

All 1,600 native numerical trace fields and paired/DV12 stream fields match
the old run. Only 343 complete rows match because the remaining rows contain
changed detector telemetry. Endpoint equality was checked from saved native
owners, rather than inferred from matching scalar losses. Full checkpoint
digests differ because the new recipe, backend/guard owners and surprise
reference are intentionally different.

`guard-event-timeline.json` reconstructs the guard from actual recorded ratios,
the measured ladder decision clocks and the actual KA2 epoch. Its guard state
matches the saved update-400, 800, 1,200 and 1,600 owners exactly. It does not
invent optimizer observations or an alternate training trajectory. The epoch
change occurs on a scheduled `.1`-weighted preservation batch; that coincidence
may influence its new reference, but this run does not isolate that influence.
The larger detector ratio is not evidence of a model regression.

## Matched execution and qualification

The new arm starts from the separately qualified fresh shared-profile
initialization after two updates. No trained historical bank or branch is
migrated into that profile. Every common native owner and both application
streams equal the old update-two state, and its two logged updates match.
The remaining 1,598 updates use the same cached source/time contexts, CPU data
stream, paired Gaussian stream, preservation every fifth update with game
weight `.1`, branch/table rates, architecture and structural interval 100.
Old checkpoints at 400/800/1,200/1,600 are reused after strict native restore.
Private fixed-game probes use common panels and the same starting DV12 stream.

The preceding real-model smoke passed 47 checks, including CPU-loaded exact
recovery through preservation, all 71 sites, dense 128-row gradients after
initial zero-output acquisition, frozen weights and exact base/zero-strength
behavior. Its CPU artifact review passed 97 checks. The full comparison
records 8,010 passing checks, including the per-update pairing checks, strict
historical restores, unchanged source/input hashes, frozen weights and
read-only endpoint evaluations. The independent CPU endpoint review is
qualified in `outputs/e22-pr223-shared-1600/independent-review.json`. It verifies
47 captured source files and both ParticleGAN git trees, 11 immutable inputs,
the common owners and frozen tensors/buffers at all four matched checkpoints,
the application/DV12 streams across all 1,600 updates, and all 570 raw evaluation
contexts under both common judges. Every paired game-score and RMSE difference
is zero.

The new A6000 run records **791.43 seconds** in its training loop and
**911.36 seconds** total. The training timer includes progress probes and
checkpoint writes. The old log records 816.55 seconds through its final
training row and 905.45 seconds total, under a different reporting workload.
These are provenance timings, not an isolated speed benchmark.

Native final digest:
`d9900423c3c0084f69a25ae2d460ffd3bcfc01a577ef0d41ded7c658dc273ec3`.
Historical final digest:
`73aa27c43e2258e50118caf70906e3b716c9e10f64ea000e092c29a490c1d0fe`.
The common frozen critic digest is
`e248f16ada2cdb6137f670e9c1dece3740db9175dee9eb324702bfab4b759d2d`.

Receipts and raw records are under `outputs/e22-pr223-shared-1600/`:
`receipt.json`, `plan.json`, `train.jsonl`, `final.pt`, `evaluation-old.json`,
`evaluation-new.json`, `comparison-control-progress.json`, `progress.jsonl`,
`guard-event-timeline.json`, and the captured application/native source trees.
The tail log is `outputs/e22-pr223-shared-1600.log`. An initial attempt stopped
at update 25 when undefined diagnostic test statistics could not be serialized
as strict JSON. It is preserved under
`outputs/e22-pr223-shared-1600-aborted-status-json`; the identical fixed-budget
run was restarted after a reporting-only fix. Native checkpoint values and
training finite checks were unchanged.

Reproduce into a fresh output directory:

```bash
CUDA_VISIBLE_DEVICES=1 /ml2/ntc-image-studio/.venv-anima/bin/python -u \
  scripts/experiment_e22_supra_pr223.py \
  --particlegan-root /ml2/hypergan/ParticleGAN-pr223-supra \
  --output outputs/e22-pr223-shared-1600
```

Recheck the completed artifacts without a GPU:

```bash
CUDA_VISIBLE_DEVICES="" /ml2/ntc-image-studio/.venv-anima/bin/python \
  scripts/review_e22_supra_pr223.py \
  --run outputs/e22-pr223-shared-1600
```

The budget was fixed before training. Output metrics never set the loss,
structural criterion, guards, stopping point, hyperparameters or checkpoint.
There was one deterministic task/stream and no seed sweep. This is evidence
about the shared routed profile on this task, rather than a general verdict
on PR223's independent Atlas optimizer. Seeking Atlas-style convergence
benefits here would require a full population-law port that accounts for
coupled token routing and preserves native full-model, fast/averaged,
zero-feature-harm guards. This comparison does not show that such a port would
close the Supra gap.

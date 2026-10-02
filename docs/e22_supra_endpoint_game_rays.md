# Prepared post-run endpoint game-shape diagnostic

This diagnostic waits for the fixed6400 two-arm run to finish and its CPU review
to qualify full completion. Preparation has not evaluated any Supra model,
loaded a full checkpoint or launched GPU work. It performs zero training or
optimizer updates and does not change the held trainer, its card or its results.

The [fixed card](e22_supra_endpoint_game_rays_v1.json) declares both FAST exports
at5120 and6400, historical ordinary6400, both original frozen judges, all570
contexts, and a separate1200-second posthoc wall budget on GPU0. Capture reuses
`FrozenSliderContexts` and the public export/backend loaders: source-caption
student minus live target-caption teacher, same B4→B8 BF16 CFG3 pairing. Cached
velocities are never substituted for that live target. Raw physical residuals
and the original continuous CPU72 panel law are retained in ignored outputs.

After independent full qualification, root must freeze the actual completed
`independent-review.json` SHA before running this command:

```sh
CUDA_VISIBLE_DEVICES=0 timeout --signal=TERM 1200 /ml2/ntc-image-studio/.venv-anima/bin/python scripts/diagnose_e22_supra_endpoint_game_rays.py \
  --run outputs/e22-supra-neutral-initialization-6400 \
  --particlegan-root ../ParticleGAN-supra-neutral-develop \
  --qualified-review-sha256 THE_FROZEN_COMPLETED_REVIEW_SHA \
  --output outputs/e22-supra-endpoint-game-rays-v1
```

`--preflight-only` checks qualification and exact source/artifact bindings
using the frozen CPU data, without full-checkpoint/model loading or CUDA
initialization. Fresh export hashes
must match the independently reviewed FAST exports. The ordinary reference
must match the original held input SHA. Both endpoints and judges remain in the
denominator for either sign of the results. Existing output directories are
refused, preserving any failure or incomplete diagnostic.

The command's outer timeout enforces the wall budget through imports and I/O;
the script also checks its deadline after preflight, throughout capture, scoring
and file proofs, and after final report serialization/write. Qualification
requires `completion.json` with `complete=true` and the matching `report.json`
SHA. It records the wall clock after the main report write; the deadline is
also checked after the completion receipt write. Detected overruns retain
results with both receipts marked incomplete. Missing completion receipts remain
incomplete diagnostic evidence. Actual execution requires physical GPU0 via
`CUDA_VISIBLE_DEVICES=0`; CPU preflight has no GPU environment requirement.

The signed alpha grid is `[-1,-.25,-.05,-.01,0,.01,.05,.25,.5,1]`. Reports retain
per-context native games, signed asymmetries, origin andalpha1 radial derivatives,
positive-ray reversals, and all six editing subject means. Atalpha1, replay must
match original endpoint records within the fixed numerical tolerance
`atol=2e-6, rtol=1e-5`; differences are reported explicitly. This is a bounded
numerical comparison, not a claim of exact arithmetic across kernels.

Negative origin radial derivatives mean outward drift along the actual student
residual direction under that finite frozen judge and Gaussian panel. A restoring
endpoint derivative or monotone sampled ray cannot rank different residual
directions or establish semantic editing quality. The two judges share an
architecture; agreement does not make them independent accuracy authorities.
This posthoc observation introduces no output-MSE optimizer or new selection,
stopping, serving-head, training or promotion gate.

Execution completed on the qualified 6,400-update cohort. The
[compact results](e22_supra_endpoint_game_rays_results.json) bind the full
report, raw residuals and completion receipt. All 870 checks passed in
98.20 seconds, with zero training updates and exact replay of the original
endpoint scores. Every fit and test context, for all five adapters under
both common critics, has a restoring radial derivative at its actual
residual. All sampled positive rays from amplitude 0.05 through 1 have no
editing-context reversals. Small-amplitude finite-panel drift remains.
These observations support useful residual-space correction under the two
common critics; they do not establish the behavior of each run's actual
training critic or explain its parameter-space convergence.

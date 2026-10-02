# Fixed paired Supra continuation

This prepares one continuation of the two independently qualified particle models
at update6400. It keeps the128×4 live bank, all71 routed sites, the original model
formula, shared native E22 controls, FAST/EMA owners, optimizer moments, and every
native/caller RNG. The held ParticleGAN revision remains
`6ec7e5788e14ea15ddc3e16ac71110458108b6a6`. Only the external training horizon
extends to12800, with editing batches and weight1 throughout.

The score to beat remains the fixed historical ordinary LoRA6400 artifact.
Ordinary LoRA12800 is a predeclared secondary comparison for training budget.
Both9600 and12800 must be evaluated and reported under both frozen learned
critics on all240 held-out editing contexts, all six subject means, and the other
inherited pools. Neither score changes the optimizer, structural guards, horizon,
or stopping. Output RMSE is an offline diagnostic. There is no seed or head scan.

The historical ordinary models use MSE/AdamW, different initialization, and one
preservation update per five steps. They received5120 editing+1280 preservation
updates at6400, and10240 editing+2560 preservation at12800. Their historical
validation-based plateau controller reports zero LR reductions through12800.
These clock artifacts were fixed before the new continuation. A win after extra
training over ordinary6400 would be a budget comparison; matched optimizer
superiority does not follow. The two particle arms form the matched comparison.

## Restore and implementation

The new orchestration binds the exact held `fresh`, `all_edit`, immutable-owner,
reference, and evaluation code objects into isolated namespaces with explicit
closure owners. It does not modify held module globals, copy the native optimizer,
or edit the parent trainer/card. The card binds full source and code identities.
Public `initialize_` runs only in a transient fresh constructor; strict public
native restore immediately replaces those values. No initializer, zero-up, or
H/b neutralization touches the trained state after restore.

Before quality updates, each arm must pass two independent native6400 restores
and two editing updates per restore: eight additional native GPU replay updates
across the two arms. Complete checkpoint states and raw rows must agree exactly.
The authoritative first6401/6402 updates must then match the witnessed rows and
native6402 state. These actual GPU checks have not run during preparation.

The inherited private CPU72 panels,12-context observational cohort, and initial
native DV12 stream stay fixed. Training curves report every200 updates, and live
status/rolling100-update losses every25. Plateau flags are observational. The new
dashboard combines parent history with this continuation without copying it.

## Cost and artifacts

Each arm gets7200 charged seconds, and the paired launch gets14400 seconds from
before Torch import through all qualification/training/checkpoint/probe/reference/
endpoint/export/final writes. Shared costs are charged equally. Independent CPU
artifact qualification gets a separate1200 seconds; combined allowance is15600.
The prior paired6400 campaign cost11199.21 seconds, the workload estimate basis.

The128 GiB free-disk gate covers34 new full checkpoints:6402 and6800..12800 every
400, seventeen per arm, approximately60 GB. The two qualified6400 inputs remain
external references, so the final state denominator is36. Inputs are not copied.
FAST exports are checked tensor-for-tensor against native checkpoints. All raw
traces, checkpoints, source capsules, and detailed receipts stay under ignored
`outputs/`.

Completion requires actual launcher exit0 within14400 seconds, complete execution
SHA companion, all36 states, both endpoints/references/ablations, and an independent
review with a complete SHA companion within1200 seconds. CPU review reports GPU
recovery and forward checks as archived-source/runtime witnesses and makes zero
model forwards or native updates. Overruns or interrupted writes stay unqualified.

## Commands after root peer review and authorization

The [authorization receipt](e22_supra_neutral_continuation_authorization.json)
records the source/software/peer review and freezes execution authorization.
The preparation-only card is archived; only its status and authorization flag
changed. Actual GPU recovery remains mandatory before quality training. CPU
preflight does not load CUDA or construct a model:

```bash
/ml2/ntc-image-studio/.venv-anima/bin/python \
  scripts/continue_e22_supra_neutral_initialization.py --preflight-only \
  --preflight-output outputs/e22-supra-neutral-continuation-source-preflight.json
```

The single launch wrapper enforces physicalGPU0 with `CUDA_VISIBLE_DEVICES=0`,
`cuda:0`, a fresh fixed output directory, and an external deadline:

```bash
/ml2/ntc-image-studio/.venv-anima/bin/python \
  scripts/launch_e22_supra_neutral_continuation.py
```

After actual launcher completion, run independent qualification once:

```bash
timeout --signal=INT --kill-after=30s 1200s \
  /ml2/ntc-image-studio/.venv-anima/bin/python \
  scripts/review_e22_supra_neutral_continuation.py \
  --run outputs/e22-supra-neutral-continuation-12800
```

The launcher and reviewer output/log/exit status must be preserved. A review JSON
alone is insufficient; its `.completion.json` must be complete, match the actual
report SHA, and meet both1200/15600-second limits. No continuation quality results
are claimed by this preparation document.

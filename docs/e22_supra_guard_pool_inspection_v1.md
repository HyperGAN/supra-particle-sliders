# Saved structural guard composition

The editing-only continuation retains seven preservation contexts in its
structural guard pool. This is a verified task-scope observation, not evidence
that those contexts caused rejection or the convergence gap.

One root-owned CPU-only saved-state inspection qualified PASS in 38.476193
seconds under a startup-through-final-write 60-second envelope. All 19 frozen
input/source bindings passed before and after inspection. It instantiated no
models, called no ParticleGAN API, performed no native or optimizer updates,
and never initialized CUDA. Scientific status is null.

| Saved pool | Active rows | Editing | Preservation |
| --- | ---: | ---: | ---: |
| Original data guard | 64 | 57 | 7 |
| Neutral checkpoint, 12800 | 64 | 57 | 7 |
| BF16-trained checkpoint, 16000 | 64 | 57 | 7 |
| FP32-final-trained checkpoint, 16000 | 64 | 57 | 7 |

Every checkpoint's active guard context and paired target values match the
original data guard exactly, including order. Only the saved active prefix
`[:guard_fill]` was counted; allocated tails were excluded and cursor values
were not used to reorder rows.

Preservation rows have equal source and target caption IDs: two rows each for
IDs 7 and 8, and three for ID 9. Their slider strength is 1. They therefore
protect the active edited student on same-caption pure-base teacher pairs.
There are no preservation training updates in the current continuation.

All three native checkpoints retain `max_context_harm=0` and
`output_error_guard=False`. The structural guard compares learned critic
features. No output loss, stopping rule, checkpoint-selection rule, or relaxed
harm tolerance was introduced by this inspection.

The separate fixed 12801–16000 event audit found 55 guard-tested proposals and
zero accepted moves. All 55 failed the maximum context-harm check, but only 12
failed that check alone; others also failed aggregate FAST or EMA feature-gain
requirements. These aggregate records do not identify the harmful contexts.
Removing preservation guards is consequently not a demonstrated convergence
fix, and the native proposal mechanism is not proven defective by these counts.

Local evidence, retained unchanged:

- Inspection: `outputs/e22-supra-guard-pool-inspection-v1/report.json`, SHA256
  `68cb913ccd775be290b9c43a1b594498c2ae10c09a7db36f160d5408e5a799b0`.
- Child completion: `outputs/e22-supra-guard-pool-inspection-v1/completion.json`.
- External receipt: `outputs/e22-supra-guard-pool-inspection-v1-root-envelope/completion.json`.
- Frozen card and root plan: `/tmp/supra-guard-pool-inspection-v1-preparation/`.
- Inspector source: `/tmp/e22_supra_guard_pool_inspection_v1.py`, SHA256
  `ea8cb5f23e7c748f82bb858cbd08b3a67c8178795a8894cea2e99c2904f71322`.
- Fixed event reduction: `/tmp/e22-supra-16000-structural-events-descriptive-v1.json`,
  SHA256 `33f1217fca3aee02d371f660e2e244a04a526d56c5fb7b5db65b94c19dcfbcd6`.

This metadata inspection is not a portable convergence toy. A future toy must
run through the public ParticleGAN API with a numerical pass/fail criterion
and actual-training GIF, and any proposed improvement must be verified on Supra.

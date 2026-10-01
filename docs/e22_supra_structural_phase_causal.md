# Structural direction test at the final V2 boundary

Changing the split direction from preservation to edit-game gradients does not
repair the native step-6,400 structural rejection. All three matched direction
arms select the same duplicate, and every evaluated candidate violates the
unchanged zero context-feature-harm guard. This is a bounded negative result;
it does not prove that proposal timing never matters at another boundary.

The qualified V2 checkpoint is
`outputs/e22-particle-v2-6400/final.pt`, SHA256
`54b243580206beb772db77132ff367fca9389b209149dcef9573449aeb8b7fcf`,
with ParticleGAN `cabe2084284db923d525918cbf3e18de6f20faac`.
The artifact receipt is
`outputs/e22-structural-phase-causal-6400/result.json`; the independent CPU
review is `outputs/e22-structural-phase-causal-6400/independent-review.json`.

## Matched intervention

The existing 100-update probe interval always lands on the every-fifth
preservation update. Native antisymmetric splits use the latest generator
backward, so this scheduling coupling is real. The new diagnostic changes
only that displacement direction at one fixed post-update state.

The three arms use:

1. The actual cached preservation gradient from update 6,400.
2. A contemporary preservation-game gradient on that same update's four
   contexts, at the current weights.
3. A contemporary edit-game gradient on update 6,399's four contexts, at the
   same current weights.

The second arm controls the cached gradient's timing before the final generator
optimizer step. Both contemporary gradients use the same private paired
Gaussian and DV12 streams, copied from the saved outgoing state. The diagnostic
never advances training or global random streams, duplicates diagnostics, or
changes any optimizer-owned gradient buffer.

All arms retain native candidate eligibility, budget, ordering, split radii,
mass transport, and full-model candidate reruns. The four native pairs are
child 54 with parents 100, 46, 121 and 91. There are four shared duplicates and
four antisymmetric candidates per direction, totaling 16 distinct candidates.
The fit reservoir selects the winner independently for each arm, with the
native antisymmetric tie preference. Protected-context scores do not select
the winner. All candidate guards are recorded solely to show whether a rejected
fit winner hides a passing alternative within this bounded candidate set.

The criterion remains learned critic-feature distance, with the original FAST
and averaged-model checks and zero per-context feature harm. No output error
metric is calculated or used in this diagnostic.

## Result

| Direction arm | Fit-improving candidates, of 8 | Guard-passing candidates | Native fit-selected variant |
| --- | ---: | ---: | --- |
| Actual cached preservation | 5 | 0 | Duplicate, 54 from 91 |
| Contemporary preservation | 4 | 0 | Duplicate, 54 from 91 |
| Contemporary edit | 5 | 0 | Duplicate, 54 from 91 |

Every arm's selected duplicate has the identical measurements:

| Native feature check | Value |
| --- | ---: |
| Fit mean gain | +0.0000653890 |
| FAST protected mean gain | +0.0001045646 |
| Averaged protected mean gain | +0.0000432356 |
| FAST maximum protected harm | 0.0012663733 |
| Averaged maximum protected harm | 0.0021293420 |
| Harmed FAST contexts, of 64 | 29 |
| Harmed averaged contexts, of 64 | 32 |

Thus the winner improves fit and both protected mean feature errors, but fails
the pointwise guard by substantial amounts. Its direction is zero because it
is a duplicate; changing the gradient cannot change that particular proposal.
Even the other candidates fail the unchanged guard, so trying the second-best
fit candidate within this set would not accept a move either.

The contemporary edit and preservation directions differ materially: their
parent-row cosines are −0.922, −0.193, +0.536 and −0.545 for rows 46, 91,
100 and 121. The cached and contemporary preservation directions have cosines
0.904–0.979. The negative result therefore reflects changed candidate
directions, not an ineffective intervention.

The 64 protected contexts comprise 57 edits and seven preservation examples.
The selected duplicate harms 25 edit and four preservation contexts on FAST,
and 28 edit and four preservation contexts on the averaged model. Both models'
worst harm occurs on an edit context. This rejection is not explained solely
by a preservation context blocking an otherwise uniformly beneficial edit
move.

## Qualification and implication

`scripts/diagnose_e22_supra_structural_phase_causal.py` restores the complete
native state and requires the original fit baseline, fit-selected candidate,
and native protected-guard metrics to replay **bit exactly**. That witness
passed. Full checkpoint state, native and global random streams, model state,
optimizer-owned gradients, and executing sources remain unchanged. Capture
completed in 31.4 seconds on the A6000 used by the training runs.

`scripts/review_e22_supra_structural_phase_causal.py` independently recomputes
83 CPU qualification checks. All pass. Its per-context feature-gain and harm
recomputation differs by at most `3.47e-18` from CUDA scalar reductions. It
checks native eligibility/order/budget, cached-gradient provenance, unchanged
split radii and direction, all candidate decisions, and fit-only winner
selection. CUDA BF16 full-model inference and gradient values remain bounded
GPU capture witnesses; the CPU review does not rerun GPU training.

This test rules out changing the current proposal gradient's task phase as a
repair for this event. It gives no evidence to weaken the guard or add output
metrics. The remaining structural question concerns producing genuinely safe
feature-game candidates; changing timing alone has not demonstrated that.
Continuous learning of the existing particles remains useful independently of
whether any structural move is accepted.

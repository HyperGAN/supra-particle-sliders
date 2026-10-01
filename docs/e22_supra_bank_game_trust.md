# Same-batch game checks on the particle-bank update

Projecting unsafe bank proposals changed learning, but did not close the gap
to ordinary LoRA. After a fixed additional 256 updates, held-out editing is
slightly worse and preservation is slightly better under both common critics.
This experimental law is not promoted to the default trainer.

The baseline to beat remains ordinary rank-16 LoRA trained for 6,400 updates.
The best qualified particle result remains the V2 model trained for 6,400.
PR223's shared routed profile exactly ties old V2 at 1,600; that result does
not mean it matches ordinary LoRA or the longer 6,400 particle run.

## What changed

Both arms resume the complete qualified particle V2 step 6,400 checkpoint,
including native models, Adam moments, EMA, conditioning buffers, controller,
row evidence and random streams. Both use its archived ParticleGAN source,
`cabe2084284db923d525918cbf3e18de6f20faac`, avoiding a simultaneous optimizer
upgrade. The native control's first five updates match the earlier qualified
five-update checkpoint digest exactly.

The experimental arm lets native Adam propose its ordinary generator, router,
bank and learned-noise update. It then keeps the proposed generator/router
values and evaluates bank fractions 1, 1/2, 1/4, 1/8, 0 against the zero-bank-step
baseline. It applies the largest fraction that does not worsen either the
clean or DV12-perturbed paired generator game on that same training batch.
Both comparisons retain the actual pre-step output sigma and G-pass Gaussian
panel. "Clean" here means no latent DV12 perturbation; paired output Gaussian
noise remains in the critic game.

The private noisy rerun recomputes native support-dependent perturbations,
using the captured actual G-pass controller and DV12 stream. Before each
proposal, its original native G prediction must replay bit exactly. Candidate
evaluations advance no owned random stream and duplicate no diagnostics.
Projection happens before native `after_generator_step`, so native motion
observers see the actual applied bank step. Native Adam moments are retained;
this is explicitly a projected-Adam experiment, not an upstream API fix.

The selector receives learned RpGAN payoffs from the current training batch.
It receives no held-out context or output-error metric. Native structural
decisions retain their learned-feature criterion, zero per-context feature
harm, FAST/EMA checks and disabled output guard. Both arms run to 6,656 updates
without metric-dependent stopping or checkpoint selection.

## Local decisions

| Applied bank fraction | Updates |
| --- | ---: |
| 1 | 55 |
| 1/2 | 41 |
| 1/4 | 24 |
| 1/8 | 16 |
| 0 | 120 |

Of 256 full proposals, 201 harm at least one same-batch game after the proposed
G/router update: 133 harm clean game and 134 harm noisy game, with overlap.
Every applied fraction passes both zero-harm checks. 136 updates retain
nonzero bank motion; mean applied fraction is 0.32617. All 128 bank rows retain
dense native gradients. Both arms accept zero structural moves.

These counts describe the bank's incremental effect after G/router move.
They do not mean 201 complete native updates worsen their own game.
Earlier actual-step attribution showed all five complete native updates
improving their own noisy training game despite some harmful bank increments.

## Full held-out outcome

Both endpoints and the original ordinary-LoRA reference use the same two
critics: frozen particle V2 D6400 and the native control's final D6656.
The final evaluation covers all 570 fit/test/training-preservation/held-out
preservation contexts, BF16/CFG3, native batches of four, and identical four
private paired-Gaussian panels. Scores below are trust minus native; lower
is better.

| Pool | Common D6400 | Common control D6656 |
| --- | ---: | ---: |
| Fit (240) | −0.00000752 | −0.00001572 |
| Test (240) | +0.00012498 | +0.00007482 |
| Training preservation (30) | −0.00012664 | −0.00011412 |
| Held-out preservation (60) | −0.00029153 | −0.00028643 |

Ordinary LoRA still has lower game loss in every pool under both judges.
For held-out editing, the trust endpoint's gap to the ordinary-LoRA reference
is +0.03001405 under D6400 and +0.03221970 under D6656. Held-out preservation
gaps are +0.01800974 and +0.01797932. The historical reference was trained for
6,400 updates; the two causal continuations were trained for 6,656. This is a
constant quality target, not an equal-budget 6,656 LoRA comparison.

The small fixed training probes favored trust during the run, but complete
held-out editing does not. Endpoint RMSE is recorded only as a diagnostic.
The additional full-model reruns are substantial diagnostic overhead; receipt
`training_seconds` includes endpoint evaluation and is unsuitable for a speed
claim. Last-training log events separately record each arm's training duration.

## Interpretation and qualification

Unsafe finite bank increments are a real local effect. Earlier exact-offset
replays isolated support-dependent detached DV12 changes in two local sign
reversals; other reversals persisted with fixed offsets and clean profiles
were nonmonotonic through the BF16 host. Checking the realized game directly
addresses both effects locally. Its failure to improve held-out editing means
that local repair is insufficient to explain the broader formulation gap.
The next separate experiment changes how particles modulate the generator
basis, while retaining the game and native population controls.

The standalone CPU reviewer qualifies 31 source snapshots against the archived
native git tree, six immutable inputs, both explicitly tagged final native
checkpoints, frozen FAST/EMA weights and conditioning, all 256 matched sampled
contexts/Gaussian/DV12 stream records, every trust decision, and raw per-context
aggregate scores. GPU exact replay and private-owner immutability are runtime
witnesses; CPU artifact review does not re-execute BF16 training.

Artifacts: `outputs/e22-bank-game-trust-256/`, including `plan.json`,
`receipt.json`, `independent-review.json`, per-arm training/progress logs and
final checkpoints. Run log: `outputs/e22-bank-game-trust-256.log`.
Live completed comparison: <http://pop-os:8767>.

Runner: `scripts/experiment_e22_supra_bank_game_trust.py`.
Experimental helper: `scripts/experimental_e22_bank_game_trust.py`.
Reviewer: `scripts/review_e22_supra_bank_game_trust.py`.

# Independent particle adapter hookup audit

The adapter remains a particle model in both formulas. Every declared site
computes a query from its actual current host activation, uses the shared bank
as routing keys, mixes the same bank as values with represented row masses,
and passes the mixed code through the particle-conditioned bridge. The V2
formula adds an unsquashed input path inside that same adapter; it does not
replace the bank, router, native game, or structural controller.

The fixed 1,600-update V2 GPU run retains useful particle information for the
held-out edit task: removing particle codes or activation-dependent routing
worsens the game under both fixed critics, with positive mean degradation for
all six edit subjects. The same interventions slightly improve the preservation
game, so beneficial particle contribution is established for edits, not for
preservation. The complete V2 architecture nevertheless improves preservation
over V1. These are distinct comparisons.

## Source audit

* `supra/particle_adapter.py:137`: queries, key logits and mixed values use the
  explicit candidate bank, in FP32, at every actual native projection. Later
  queries are recalculated after preceding edits. Codes remain differentiated
  through both keys and values; the native DV12 support-radius calculation is
  detached while its returned `latent + displacement` retains that gradient.
* `supra/particle_adapter.py:142`: CFG represents the two correlated branches
  as `[contexts, 2, tokens, rows]`, preserving the original context count for
  usage attribution. Conditional/unconditional branch order is retained when
  codes rejoin the native host, then guidance combines complete model outputs.
* `supra/particle_adapter.py:150`: V1 uses
  `up(tanh(bridge(concat(down(input), code))))`; V2 uses
  `up(down(input) + tanh(bridge(concat(down(input), code))))`. Both contain the
  same trainable tensors, shared particle bank, and per-site router queries.
* `supra/particle_training.py:37`: fresh networks use the public
  `particlegan.init.deterministic_orthogonal_`, with role keys G/D/E/R
  `0/1/2/3`. Zero output matrices preserve the native base starting point.
  Frozen host and teacher parameters are excluded from optimizer ownership.
* `supra/particle_training.py:42`: the native optimizer owns distinct generator,
  router and bank groups. The public policy adds its separate learned-noise
  group. The audit verifies every trainable tensor has exactly one intended
  owner and that no frozen parameter enters those groups.
* `supra/particle_game.py:134`: D and G each receive a complete perturbed native
  model rerun. The teacher is frozen and recomputes the target for the same
  live batch shape. The G backward uses the paired native adversarial game;
  raw reconstruction error is not a training objective.
* `supra/particle_training.py:47` and `supra/particle_game.py:95`: structural
  evidence and proposals rerun the full clean model and compare learned
  critic features. Feature context harm is zero and the output-error guard
  is disabled. This feature distance is a game surrogate, not a direct
  certificate that an accepted proposal improves RpGAN payoff.
* `supra/particle_export.py:29`: shapes cannot distinguish V1 and V2, so clean
  export metadata and format versions must agree. Legacy files dispatch to
  V1; V2 exports explicitly identify their formula. Checkpoint configuration
  must likewise reject a formula change on native resume. The plain Python
  architecture attribute is retained by deepcopy and public serving copies.

An integration compatibility risk was identified during the audit: changing
the training-loop factory's default formula would cause existing V1 recovery
entrypoints to construct V2 and fail strict restore. Recovery must select the
saved configuration's mode with V1 as the legacy fallback, while experiments
pass their chosen architecture explicitly. The export/plumbing agent fixed
the supported evaluator, full-progress, training-controls, and linear-path
recovery entrypoints before the experiment sources were frozen.

## Executed checks

`scripts/audit_e22_supra_particle_hookup.py` is an independent diagnostic. It
performs no optimizer update and checks the complete native checkpoint, all
training RNGs, and existing parameter gradients remain unchanged. Counterfactual
output RMS is an intervention observation only; game payoff supplies the
reported effect on the game.

The CPU receipt `outputs/e22-particle-hookup-audit/cpu-native-two-step.json`
contains a miniature native Supra model with 11 real sites and all 128 bank
rows. Each formula received two complete native game updates using identical
initialization, data and stream controls. Both formulas passed:

* nonzero clean and DV12 game gradients to all 128 bank rows, all 22 router
  tensors, and every trainable down/bridge/up tensor;
* ordered complete-model routing, site-specific code perturbation, bank
  translation and row-deletion replay;
* exact strength-zero pure-base behavior and native forward parity;
* unchanged native state, training streams and preexisting gradients.

First-update zero gradients to the bank/router are expected because the fresh
output matrices start at zero. The second-update test qualifies connectivity
after those output matrices begin learning. Zero learned-noise gradients at
the physical clamp do not indicate a missing optimizer owner; the noise-floor
release issue is documented separately in the convergence investigation.

The native GPU receipt
`outputs/e22-particle-hookup-audit/native-v1-6400.json` audits the completed
6,400-update model under its archived `f459cb6d` ParticleGAN source. It visits
all 71 actual sites with native CUDA BF16 host layers and FP32 particle routing.
All 71 separate code perturbations changed the final output. All 128 bank
rows and every trainable generator/router tensor had nonzero clean and DV12
native G-game gradients. Exact ownership, native parity and full state/RNG
immutability passed.

| Intervention on the same fixed edit batch | Fixed-D G payoff | Change from clean | Output-change RMS |
| --- | ---: | ---: | ---: |
| Clean trained particle model | 0.811842 | — | — |
| Set mixed particle codes to zero | 1.084821 | +0.272979 | 0.084827 |
| Delete most-used bank row 100 | 0.943241 | +0.131399 | 0.057985 |
| Set the entire bounded modulation to zero | 1.950974 | +1.139131 | 0.219136 |
| Translate the bank by 0.1 | 0.811461 | −0.000381 | 0.015087 |

Higher G payoff is worse in this paired game. Removing codes or the important
bank row worsens the measured game, establishing useful particle content in
the existing trained model. A bank translation slightly helps this batch;
that is an observation, not a selected training intervention.

## Completed native V2 qualification

`outputs/e22-particle-v2-1600/audit-step-two.json` and `audit-final.json` audit
the real 71-site host after updates 2 and 1,600. Both checkpoints have native
clean and DV12 G-game gradients to all 128 bank rows, all 142 router tensors,
and all 284 trainable generator tensors. All 71 individually perturbed sites
change the final native output. Exact role ownership, native forward parity,
strength-zero pure-base parity, released forward frames, and complete native
checkpoint/RNG/existing-gradient immutability pass at both boundaries.

The complete held-out panels are in
`outputs/e22-particle-v2-1600/particle-contribution.json`. Each intervention
uses four identical paired Gaussian panels under both the frozen archived
V1-1600 critic and the frozen V2-final critic. The independent review
recomputed the stored means and per-draw deltas from their per-context records.
All values agree. Its persisted check receipt is
`outputs/e22-particle-v2-1600/particle-contribution-review.json`; checkpoint and
critic hashes agree with the architecture qualification receipt. Complete
native state, served model/bank, and both critics remain unchanged.

Positive G-game change below means that the intervention makes the game worse.

| Pool and intervention | Archived V1 D: game change | V2-final D: game change | Output-change RMS, diagnostic only |
| --- | ---: | ---: | ---: |
| Test 240: zero particle codes | +0.031929 | +0.031230 | 0.032424 |
| Test 240: mass-only routing | +0.024319 | +0.026733 | 0.028467 |
| Preservation 60: zero particle codes | −0.001594 | −0.002690 | 0.025476 |
| Preservation 60: mass-only routing | −0.002495 | −0.002526 | 0.016751 |

All four Gaussian panels agree with the aggregate direction in every row.
Zeroing codes worsens 185/240 test contexts under the archived D and 182/240
under V2-final D. Mass-only routing worsens 175/240 and 178/240 respectively.
Every edit subject has a positive mean game degradation under both critics
for both interventions; individual contexts can still improve.

| Held-out edit subject | Zero codes: archived / V2-final D | Mass-only routing: archived / V2-final D |
| --- | ---: | ---: |
| Robot | +0.055971 / +0.056838 | +0.026877 / +0.031205 |
| Wolf warrior | +0.021761 / +0.017815 | +0.024275 / +0.024743 |
| Comic knight | +0.024396 / +0.025315 | +0.016367 / +0.020614 |
| Cave warrior | +0.037390 / +0.036675 | +0.034210 / +0.034899 |
| Sorceress | +0.032570 / +0.032364 | +0.036581 / +0.039691 |
| Cathedral knight | +0.019486 / +0.018374 | +0.007602 / +0.009247 |

Preservation shows the opposite aggregate effect. Zero codes improves 47/60
contexts under both critics; mass-only routing improves 55/60 and 54/60.
Mass-only routing improves all three preservation subjects under both critics.
Zero codes improves the lake and bicycle subject means under both critics;
the cat mean worsens slightly under the archived critic (+0.000259) and improves
under the V2-final critic (−0.001747). Thus the audit must not claim that particle
codes or activation-dependent selection improve preservation. The useful edit
path has a small preservation cost in these counterfactual game measurements.

This does not negate the overall V2-versus-V1 improvement. The matched
architecture qualification receipt reports lower preservation G payoff by
0.018327 under the archived critic and 0.017770 under V2-final D, and lower
mean game payoff for all three preservation subjects. Removing particles from
V2 and comparing the complete V2 architecture against V1 answer different
questions. The ablated model was not retrained.

The input path is substantial but does not bypass all useful particle use.
On the fixed final audit edit batch, the median per-site code-effect RMS divided
by total adapter-delta RMS is 0.2371, versus 0.0710 after two updates. This
measures the bridge's code effect at fixed local input, not a percentage of
final-output improvement. The final range is 0.00825–0.66446, showing varied
site dependence. Zeroing codes changes the complete model and worsens the
held-out edit game, while constant mass-only routing also worsens it; the
bank supplies useful information and activation-dependent selection matters.

The run accepted zero structural moves. These measurements qualify continuous
particle learning and its measured edit contribution on one task and training
stream; they do not qualify the benefit of birth/death. Held-out trajectory
contexts are correlated and the two critics are endogenous game witnesses.
Unseen-task generalization and independent image quality remain outside this
audit's evidence.

## Reproducing the qualification

Run the same diagnostic at update 2 and the fixed final horizon of each
matched native architecture experiment. An architecture claim requires more than
faster aggregate learning: retain measurable code/bank dependence, useful
fixed-critic game contribution under particle ablation, routed usage and
gradient evidence, and exact serving/export parity. Report direct input-path
and code-effect magnitudes together so a dominant linear path is visible.

For example, with the same pinned ParticleGAN source as the experiment:

```bash
python scripts/audit_e22_supra_particle_hookup.py \
  --run outputs/CHOSEN_RUN \
  --checkpoint outputs/CHOSEN_RUN/checkpoint-00002.pt \
  --output outputs/CHOSEN_RUN/particle-hookup-step2.json \
  --all-site-interventions
```

The script accepts an explicit checkpoint path and reads `data.pt` from the
run directory. It can alternatively be imported into the fixed-horizon
runner at a completed update boundary. The same check should be repeated on
the final checkpoint without selecting a checkpoint, mode, or training budget
from raw output errors.

`scripts/evaluate_e22_supra_particle_contribution.py` broadens the same question
to every test context and preservation context, rather than extrapolating from
one fit batch. It evaluates three predeclared clean-forward interventions:

* the selected native served particle model;
* zero mixed particle codes at all native sites, retaining real router calls
  and propagation through the complete model;
* mass-only routing: zero site queries while retaining active bank values and
  represented masses, removing activation-dependent selection.

Every arm uses the same four private Gaussian panels under both the archived
V1-1600 critic and the chosen checkpoint's final critic. It reports paired
game deltas per draw, context, and subject, plus output-change RMS as a
diagnostic observation. Complete checkpoint/RNG, served model/bank, and fixed
critic immutability are checked. Neither critic is an independent image-quality
judge; agreement provides an additional game witness.

```bash
python scripts/evaluate_e22_supra_particle_contribution.py \
  --run outputs/e22-particle-v2-1600 \
  --output outputs/e22-particle-v2-1600/particle-contribution.json
```

`--checkpoint` selects an explicit checkpoint and `--include-fit` additionally
reports all fit and training-preservation contexts. The new evaluator passed
a two-update native CPU fixture with eight contexts, three arms, four panels,
and unchanged native/served state. Its CPU receipt is
`outputs/e22-particle-hookup-audit/cpu-contribution-panel.json`. GPU whole-pool
results for the fixed V2 run are complete in
`outputs/e22-particle-v2-1600/particle-contribution.json`. The earlier
6,400-update intervention table describes the legacy V1 model separately.

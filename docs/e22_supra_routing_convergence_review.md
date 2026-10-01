# Particle routing and structural convergence review

This review examines the fixed V2 step-1,856 checkpoint at
`outputs/e22-particle-noise-update-256/latest`, with matched step-1,600 V1
and V2 references. It performs no training update, seed sweep, output-loss
optimization, guard change, or checkpoint selection. Current long-run training
sources and completed artifacts remain unchanged. A separate, explicitly tagged
routing continuation is prepared below to test the measured remaining issue.

## What is already established

The V2 particle path is connected and useful on held-out edits. Its native GPU
audits at updates 2 and 1,600 have gradients to every generator/router tensor
and all 128 bank rows, and every one of the 71 independently perturbed sites
changes the final output. Whole-pool code ablation worsens mean edit-game
payoff under both frozen critics for all six edit subjects. Activation-dependent
routing likewise improves the measured edit game relative to constant mass-only
routing. Those measurements do not establish beneficial particle contribution
on preservation: removing codes/routing slightly improves its game, although
the complete V2 architecture preserves better than V1.

Evidence and exact intervention values are in
`docs/e22_supra_particle_hookup_audit.md` and
`outputs/e22-particle-v2-1600/particle-contribution-review.json`.

V2 also substantially reduces the earlier representation saturation. On the
stored native audit batch, V2-1600 has mean bridge saturation 0.00240, compared
with 0.10271 for V1-6400. Mean site maximum routing weight is 0.14362 versus
0.39574, and mean entropy is 4.1591 versus 3.0033. These are different training
horizons, so they illustrate the improved geometry rather than providing a
matched-budget causal comparison. The matched 1,600-point geometry diagnostic
below addresses that distinction.

## Remaining key/value norm coupling

The routing rule still uses the same bank coordinate as both key and value:
`logit_i = query dot bank_i / sqrt(4) + log_mass_i`. Increasing a particle's
magnitude can therefore improve its selection score as well as changing its
decoded value. Directional matching and value amplitude are coupled.

Saved-bank measurements, with no GPU work, give:

| Checkpoint | Median bank norm | Maximum bank norm | Maximum / median | Largest row |
| --- | ---: | ---: | ---: | ---: |
| V1-1600 | 3.1397 | 13.7823 | 4.3896 | 100 |
| V2-1600 | 2.8452 | 10.8823 | 3.8248 | 100 |
| V2-1856 | 2.9528 | 11.6452 | 3.9438 | 100 |

These medians use PyTorch's lower-middle convention for the 128-row bank.

Magnitude dispersion alone does not prove harmful routing. The existing V2-1600
native audit nevertheless retains one sharply concentrated site:
`blocks.8.cross_attn.q` puts mean weight 0.99817 on row 100, with entropy
0.01825. Most other sites are much softer. The earlier V1-6400 investigation
found a nearly frozen query Jacobian at this site and large norm-dependent
selection changes elsewhere; current V2 Jacobians must be measured rather
than inferred from that older model.

`scripts/diagnose_e22_supra_current_particle_geometry.py` performs that
read-only comparison on the same latest-selected next edit batch, next
preservation batch, and first four test contexts at all three checkpoints.
It captures per-site selection, entropy, native code/query Jacobians,
equal-norm-key selection/Jacobian counterfactuals, and both modulation and
actual up-input feature ranks. Host routing remains unchanged. The analytical
native Jacobian calculation was checked against autograd on a CPU fixture.
The same script emits current clean and four private DV12 residual draws for
independent critic analysis, without duplicating full-model GPU work. The GPU
capture completed; all model/encoder state digests remained equal. Its raw
receipts are `outputs/e22-particle-current-geometry/geometry.json` and
`routing-review.json`.

On the identical first four test contexts:

| Measurement, mean across 71 sites | V1-1600 | V2-1600 | V2-1856 |
| --- | ---: | ---: | ---: |
| Maximum routing weight | 0.27877 | 0.14612 | 0.16013 |
| Routing entropy | 3.54991 | 4.15731 | 4.09052 |
| Bridge saturation fraction, abs(preactivation) > 3 | 0.02931 | 0.00185 | 0.00286 |
| Centered modulation effective rank | 1.93544 | 2.52193 | 2.41661 |
| Centered actual up-input effective rank | 1.93544 | 2.61964 | 2.58476 |

The same pattern holds on the matched next edit/preservation batches: V2
reduces the original bridge saturation and raises measured feature rank.
Latest V2 is not globally stuck in saturated modulation or zero query
Jacobians. These ranks describe the captured contexts and token covariance,
not a global capacity or condition-number certificate.

One local routing bottleneck remains. At `blocks.8.cross_attn.q`, row 100 is
selected for every captured test token. Its mean weight is 0.999037 and median
code/query Jacobian Frobenius norm is 0.011364 at step 1,856. The same query
with equal-norm keys gives weight 0.054531 and Jacobian 1.222085, about 108
times larger. It usually retains that row's best-aligned direction: the
argmax changes for only 4.1% of those tokens. This is sharp concentration
around a useful direction, rather than proof that the selected particle is
wrong. The query Jacobian worsens from 0.027216 at V2-1600 to 0.011364 at
1,856 as the largest bank norm grows, even though the query norm stays near
2.67. Fit and preservation probes show the same local tendency.

Other sites exhibit selection/magnitude coupling. At
`blocks.13.cross_attn.q`, latest test weights put 0.972982 on row 100; the
selected directional cosine is 0.77484 while the best is 0.90685. Equal-norm
keys change its argmax for 97.6% of tokens. Across sites, the average changed
argmax fraction is 89.2%, though many soft sites have nearly tied low logits,
so that percentage alone is not a harm measure. Normalization increases the
median query Jacobian at only two of 71 test sites and decreases it elsewhere:
the median across sites falls from 3.16394 to 1.48688. It therefore cannot be
presented as a general gradient amplification fix.

The one architectural trial justified for qualification is **fixed-RMS normalized
keys with the original bank values retained**:

`key_i = sqrt(4) * bank_i / max(norm(bank_i), epsilon)`

Routing would use `query dot key_i / sqrt(4) + log_mass_i`, while decoded codes
remain `weights @ bank`. Keys are derived from the same trainable particle,
so there is still one native bank, its value geometry, its represented masses,
DV12, and complete candidate reruns. This separates selection direction from
value magnitude without substituting an ordinary LoRA or deleting particles.
It requires a versioned routing formula for resume/export.

The frozen argmax/Jacobian counterfactual does not prove better convergence;
actual routing changes alter later activations. The separate
`scripts/experiment_e22_supra_fixed_rms_keys.py` tests a fixed 256-update
continuation from the qualified step-1,856 checkpoint against the existing
native continuation. It records both native and modified zero-update game
probes under the frozen step-1,856 critic, then repeats the same 12 edit and
12 preservation probes after training. Thus an immediate routing-switch
shock is distinguishable from subsequent game convergence. It retains the
native data/private streams, bank values, masses, DV12, optimizers, structural
feature criterion, and zero feature-harm guard. Output errors are reporting
only and never select a variant or stopping point.

`scripts/experimental_e22_fixed_rms_keys.py` implements the new law solely in
the research runner. Every FAST/averaged projection derives keys from its
current full-model candidate; native `routing.mix` still decodes that original
candidate's physical values and additive masses. Four focused CPU tests
verified both normalized-key and physical-value gradient paths, all bank/router
gradients, sequential particle perturbation effects, CFG/deepcopy behavior,
zero/tiny-key finiteness, and rejection of unsupported checkpoint metadata.
Before changing routing, the runner requires exact native four-update replay
and equality with the native control trace; it then requires exact experimental
replay across a preservation update.

The warm-start forward changes immediately; this is a continuation test, not
an identical-output fresh-initialization claim. The final full checkpoint is
wrapped as `e22_supra_experimental_fixed_rms_keys_v1`, requires dedicated
restore, and carries `experimental_routing` in its native config. No ordinary
V2 clean adapter is emitted. Long-run native source files are unchanged.

## Fixed 256-update routing trial result

The continuation completed at step 2,112. It did not improve the measured
native game relative to the matched native-routing control. Under the same
frozen step-1,856 critic, lower generator game loss is better:

| Complete pool | Contexts | Native routing | Fixed-RMS keys | Change |
| --- | ---: | ---: | ---: | ---: |
| Fit edits | 240 | 0.932508 | 0.943799 | +0.011291 |
| Held-out edits | 240 | 1.026345 | 1.032796 | +0.006451 |
| Training preservation | 30 | 0.738950 | 0.740755 | +0.001805 |
| Held-out preservation | 60 | 0.743862 | 0.745087 | +0.001224 |

All six edit-subject mean game losses worsen; 200 of 240 held-out edit
contexts worsen and 40 improve. All three preservation-subject means worsen;
35 of 60 held-out preservation contexts worsen and 25 improve. This supports
retaining native routing for this task and fixed continuation budget. The
per-run final critics differ, so their scores are not used for this comparison.
No change is promoted to the core architecture.

The routing switch itself imposed an initial cost on the 12 fixed edit probes:
clean frozen-critic game rose from 0.872652 to 0.895846; DV12 game rose from
0.875637 to 0.895987. The subsequent 256 updates reduced those modified-law
values to 0.863491 and 0.865704. Thus it learns after the switch and recovers
the initial edit-probe shock, yet does not win the complete-pool matched
comparison. Preservation probes initially improve slightly, from clean
0.736680 to 0.736004, then regress to 0.736936. These small training probes
describe switch/recovery behavior rather than substituting for the held-out
pool comparison.

`scripts/review_e22_supra_fixed_rms_keys.py` independently recomputed all
32 artifact qualification checks on CPU; all passed. The result is saved in
`outputs/e22-particle-fixed-rms-keys-256/independent-review.json`. It verifies
actual input/current/archive source hashes; tagged outer and inner native
checkpoint digests; explicit routing metadata and absence of a clean export;
unchanged FAST/averaged frozen hosts and encoders; the exact 256-update horizon;
CPU-recomputed data sampling; all per-step data, paired-Gaussian and DV12
stream matches against the native control; identical final private stream
states; gradients to all 128 bank rows at every update; unchanged native
recipe, optimizer roles and feature guards; and complete-pool paired context,
teacher and aggregate recomputation. Native and experimental prefix replay,
dedicated restored-forward equality, and monitor state/RNG immutability are
qualified GPU witnesses backed by those source/state checks; the CPU review
does not rerun CUDA BF16 inference. No structural move was accepted.

The measured local frozen-query Jacobian improvement is therefore not a
convergence win at this budget. This one warm continuation does not settle a
fresh-initialization experiment or wider tasks, and does not imply another
trial or weaker guard. The native long continuation remains unchanged.

## Why structural proposals do not move

The CPU audit in
`outputs/e22-particle-current-geometry-cpu/structure-and-table.json` recomputes
every recorded acceptance condition from the traces. All flags match the
native conditions. The controller is observing, proposing, and rejecting;
there is no missing probe clock or disconnected candidate callback.

| Checkpoint | Evaluations | Deletion probes | Proposed variants | Guard rejections | Accepted moves |
| --- | ---: | ---: | ---: | ---: | ---: |
| V1-1600 | 16 | 128 | 120 | 12 | 0 |
| V2-1600 | 16 | 128 | 128 | 14 | 0 |
| V2-1856, cumulative | 18 | 144 | 144 | 15 | 0 |

The native guard requires sufficient positive mean feature gain on FAST,
nonnegative mean gain on the averaged model, and no increase for any protected
context in either model. It uses learned critic-feature distance, with raw
output guards disabled. All 14 V2-1600 guarded proposals violate the zero
context-harm constraint; only two pass both mean-gain checks. Their worst
context harms range from 0.001350 to 0.006936, with median 0.002331, far above
the decision tolerance of 1e-12. These are recorded feature-harm violations,
not an accidental acceptance boolean or a nearly-zero tolerance comparison.

At update 1,700, the FAST mean gain is +0.00003430, but the averaged gain is
−0.00000705 and maximum protected harm is 0.00175967. Update 1,800 is rejected
before guard evaluation because its selected proposal fails the fit-improvement
check. Thus the current no-move outcome follows the configured conservative
contract. The audit does not isolate physical candidate effects from BF16
propagation, or prove that unexamined candidate directions could not pass.

There is also an integration scheduling coupling. The probe interval is 100,
and every fifth update is preservation. All 16 proposal events in each
1,600-update run, plus events 1,700 and 1,800, consequently occur on preservation
updates. The native antisymmetric displacement uses `-latest_gradient[parent]`,
captured from that same backward pass, so every split direction comes from
the preservation game while its fit reservoir mostly contains edit contexts.
The direction is normalized, so multiplying the preservation payoff by 0.1
does not reduce its split radius. This is a task/phase coupling to investigate,
not evidence that the guard is implemented incorrectly. Of V2-1600's 14
guarded candidates, seven are antisymmetric and seven are duplicate variants
with no displacement; gradient-phase coupling cannot explain the duplicate
rejections.

The native CPU witness in
`outputs/e22-particle-current-geometry-cpu/structural-phase-witness.json`
confirms the cached gradient exactly equals the table Adam first moment at
all three saved boundaries; beta1 is zero. Calling native `_delta` with the
actual table/averaged table and either `g` or `0.1*g` changes the result by
less than 1.83e-7 relative across all 128 rows. Saved tensors remain unchanged.
The native `_delta` AST is identical in the f459 and cabe archives. The
1,856 boundary itself is an edit step; this witness confirms current gradient
capture and normalization, while the recorded 100-update clock determines
the historical preservation-only proposal phase.

A future predeclared probe schedule that does not align exclusively with
preservation, or explicit reservoir-aligned game-gradient directions, can
qualify that coupling separately. Keep the zero feature-harm guard intact.
The present continuous bank learning remains meaningful even without accepted
birth/death moves; their additional convergence benefit has not been shown.

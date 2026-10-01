# Particle architecture experiment

The experiment keeps the shared 128×4 particle bank, the ordered routing at all
71 native Supra projection sites, per-site DV12 perturbations, native E22
optimizers and structural controllers. It changes the generator's internal
input path, not the objective or the particle mechanism.

For `hidden = down(input)` and the routed `code`, the formulas are:

```
nonlinear_v1:        up(tanh(bridge([hidden, code])))
linear_modulated_v2: up(hidden + tanh(bridge([hidden, code])))
```

The added input path gives `down` a gradient path outside the saturated tanh.
The same bank codes still condition the bounded modulation. This also changes
the reachable generator functions, so the experiment cannot isolate tanh
saturation as the sole cause of the earlier performance gap.

Particle connectivity is insufficient by itself: the new path could learn to
ignore the particle modulation. The evaluation therefore measures the game
loss when particle codes are removed or routing is replaced with mass-only
mixing, in addition to testing gradients and full-model candidate replay.
These interventions are evaluation only, not alternative training losses.

## Fixed experiment

`scripts/experiment_e22_supra_particle_architecture.py` trains V2 from the
original frozen Supra base for exactly 1,600 updates on the original
final-boss slider task. It uses the archived ParticleGAN source at
`f459cb6d6aaaabeb1af076ec53ad7a963618de90` for both architectures. This isolates
the architecture change from subsequent ParticleGAN updates.

The V1 comparator is the completed 1,600-update particle run. Before accepting
that historical comparator, the experiment constructs V1 afresh and requires
bit-exact reproduction of its complete archived step-two native checkpoint
and both trace rows. V2 must then have an identical initialized native policy
state, tensor values, data RNG and paired-noise RNG. The sole configuration
addition is the explicit V2 architecture field.

Both architectures use public `particlegan.init.deterministic_orthogonal_`,
zero-initialized output projections, rank 16, batch size four, and preservation
every fifth update with the original game weight 0.1. Every V2 data batch and
paired Gaussian stream is compared against the original V1 trace. Native DV12
and row controls stay enabled; changes in accepted structures can legitimately
change their later draw programs.

The final checkpoint is selected by the fixed update horizon. Output RMSE is
reported only after training; it never affects gradients, guards, structural
acceptance, checkpoint selection, or stopping. Structural candidates use the
native learned critic-feature criterion with zero feature-context harm and no
output-error guard. A feature improvement does not itself certify improved
RpGAN payoff.

Full fit, held-out test, training preservation, and held-out preservation pools
are evaluated with four private paired Gaussian panels and two common judges:
the archived V1 final step-1,600 critic and the new V2 final critic. The helper's
`fixed_start_D` field names the former; it is not an untrained initial critic.
Both judges are learned and endogenous. This is one task and one matched
training stream, not a seed sweep or a general superiority claim.

## Compatibility and checks

The host constructor retains the exact V1 formula by default. Full training
supports explicit V1/V2 selection; legacy resumes infer V1 from a missing
architecture field and retain strict configuration equality. Legacy exports
retain their original V1 format and metadata. V2 exports declare both the new
format and architecture because their tensor names and shapes match V1.
Loading rejects conflicting or unknown versions.

The runner checks exact native continuation, unchanged frozen host/teacher
weights, bit-exact clean export reload, unchanged training state after
evaluation, and source hashes before completion. The separate particle audit
checks all routing sites, optimizer ownership, game gradients, intervention
effects, and checkpoint/RNG/gradient immutability.

Initial validation: 68 focused CPU tests passed. The real GPU step-two audit
reached every one of the 128 bank rows and all 142 router tensors through both
clean and DV12 game gradients; all 71 individual site perturbations changed
the final output. Exact V1 step-two replay and V2 native resume also passed.

## Evidence

- Fixed plan, native checkpoints, trace and final evaluations:
  `outputs/e22-particle-v2-1600/`.
- Full-model two-update particle audit:
  `outputs/e22-particle-v2-1600/audit-step-two.json`.
- Earlier V1 step-6,400 audit:
  `outputs/e22-particle-hookup-audit/native-v1-6400.json`.

On the predeclared four-context V1 audit batch, removing particle codes worsened
native generator game loss from 0.811842 to 1.084821. Deleting the most-used
row worsened it to 0.943241. That establishes useful particle content for that
batch; complete held-out contribution checks are required for the new formula.

## Qualified 1,600-update results

The fixed run completed and passed independent CPU qualification of sources,
native state, frozen tensors, complete traces, paired streams, export tensors
and full-pool aggregates. All 1,600 DV12 RNG digests also matched V1. No
structural moves were accepted, so these results show an architecture benefit,
not a measured benefit from birth/death.

| Held-out measure, lower is better | Particle V1 | Particle V2 |
| --- | ---: | ---: |
| Test G loss, common V1 final critic | 1.144665 | 1.089320 |
| Test G loss, common V2 final critic | 1.148233 | 1.090568 |
| Preservation G loss, common V1 final critic | 0.767654 | 0.749327 |
| Preservation G loss, common V2 final critic | 0.760087 | 0.742317 |
| Test velocity RMSE, evaluation only | 0.111964 | 0.102711 |
| Preservation relative RMS, evaluation only | 0.055676 | 0.048691 |

V2 improves both common game judges, test RMSE by 8.26%, and preservation drift
by 12.55% versus V1. Under the common V1 judge, its test game loss improves on
238/240 contexts and all six subject means; preservation improves on 54/60
contexts and all three subject means. The original ordinary-LoRA example's
test RMSE is 0.096037, still better than V2 by about 6.95% at this budget. A
separate read-only evaluation under these same paired panels and judges gives
the original example test G losses 1.068160 / 1.072243 and preservation G
losses 0.732916 / 0.727723. The original still wins both common game judges
as well. The architecture change narrows the gap; it does not establish
superiority over the original training recipe. The original adapter's
142 tensors, provenance and output RMSE parity were checked, with no
teacher contamination or evaluation-state changes. Its receipt is
`evaluation-original-game/receipt.json`.

The final full-model audit still reaches all 128 bank rows and all 142 router
tensors through clean and DV12 game gradients. Individual code shifts at all
71 sites change final output. On the complete 240-context test pool, removing
particle codes worsens G loss by 0.031929 under the V1 judge and 0.031230 under
the V2 judge. Replacing activation-dependent routing with mass-only mixing
worsens it by 0.024319 and 0.026733, respectively. Both effects hold across all
four matched Gaussian panels. The direct path has not rendered the particles
or learned routing irrelevant on held-out edits.

There is a preservation qualification: code removal slightly improves G loss
there, by 0.001594 and 0.002690 under the two judges; mass-only routing also
slightly improves it. Particle content is useful for edits but still adds some
preservation cost. These interventions are diagnostics, not training choices.

Receipts: `qualification-review.json`, `audit-final.json`, and
`particle-contribution.json` inside the experiment directory. Image rendering
and endpoint evaluation were not part of this fixed experiment.

## Learned-noise update

PR155 subsequently added commit
`cabe2084284db923d525918cbf3e18de6f20faac`, excluding the noise tester from its
own floor-release eligibility. This fixes the reproduced self-veto: a raw
sigma of 0.0625 now remains 0.0625 with a nonzero learning gradient once the
other required groups settle. Nineteen relevant CPU regressions and nine
CUDA noise-floor cases passed, including release and exact recovery.

At the completed V2 checkpoint the generator, router and table scales are all
1, so they independently keep applied sigma at 0.125. The fix is necessary
for eventual release but does not change this checkpoint's floor. The above
architecture comparison deliberately used the same archived f459 source for
both models; a separate matched continuation checks the newer native source.

That continuation completed: both sources started from the identical full V2
step-1,600 checkpoint and ran 256 unchanged native updates to step 1,856.
Independent qualification verified identical final full native states, every
training row, every full-pool evaluation record, and every exported tensor.
The generator/router/table scales stayed at 1 and applied sigma stayed 0.125
under both versions. The fix has no behavioral effect within this still
unsettled budget. The matched final digest is
`f234982bdbfea66d97710514ebaf3598c968f4c281b9e03bc9a14d6039835e17`.

Both continuations improve held-out game loss under the fixed step-1,600 V2
critic from 1.090568 to 1.072027. Evaluation-only test RMSE falls from 0.102711
to 0.099363. This is progress from additional training, not a gain caused by
the noise fix. There were no accepted structural moves in these continuations.
The two GPU processes ran concurrently, so their timings cannot establish a
speed comparison. Results and qualification are in
`outputs/e22-particle-noise-update-256/qualification-comparison.json`.

`requirements-e22.txt` now pins the tested #155 head at cabe2084. All 68
focused Supra adapter/export/resume/game/evaluator tests also pass against
that updated native source. Older experimental source archives and receipts
remain intact.

## Live fixed-6,400 continuation and reference dashboard

`scripts/train_e22_supra_particle_convergence.py` continues the qualified
latest-source V2 state at update 1,856 to a predeclared 6,400 updates, preserving
the native configuration, owned streams, bank, routing, DV12 and structural
guards. It reports edit and preservation losses separately. Diagnostic probes
do not stop training or select a checkpoint.

The dashboard is served by `scripts/serve_e22_supra_progress.py` on
`0.0.0.0:8765`; on this machine it is available at `http://pop-os:8765`.
It refreshes every five seconds and provides a recent-update range for
spotting plateaus. Its blue curves score the current clean and privately
perturbed generator with a frozen step-1,856 critic. The gold curve scores
all 16 existing original ordinary-LoRA snapshots, every 400 updates through
6,400, with exactly the same judge, contexts, four paired Gaussian panels,
BF16 host and batch-four CFG3. Dotted gold lines mark the original final
6,400-update scores. These are training probes, not full held-out evaluation.

The probe panel contains 12 fit contexts and 12 training preservation
contexts. It uses a private CPU generator initialized at 72, distinct from
the full-pool evaluator's private CUDA panels. Each live probe requires an
unchanged complete native checkpoint; archived reference evaluation also
checks frozen weights, inputs and global RNGs. Reference receipts and an
independent reconstruction of the panels are in
`outputs/e22-dashboard-baselines/`. The original reference follows the
`final-boss-supra-converged` lineage throughout: its update-1,600 adapter is
different from the separate `final-boss-supra-1600` artifact used in the earlier
fixed-1,600 comparison. The dashboard does not splice those runs.

The archived original LoRA MSE training curve appears in a separate panel,
in its own units. New particle optimization continues to use the native game.
A possible plateau means less than 0.2% absolute net change in clean frozen-
critic loss across the last four probes. Regression is reported separately;
neither indicator changes training. Live files are
`outputs/e22-particle-v2-6400/{status.json,progress-latest.json,progress.jsonl}`.
The server exposes only explicit progress routes, not native checkpoints.

## Qualified 6,400-update results

The continuation completed all 4,544 additional native updates. Independent
CPU review checked the actual pinned source tree, complete states, unchanged
FAST and averaged frozen weights, every CPU data draw, the qualified native
control prefix, saved stream endpoints, 24 fixed probes, bounded native
histories, all 428 export tensors and complete evaluation aggregates. Exact
GPU replay, paired CUDA draws, state-preserving probes and export-forward
parity remain explicitly bounded runtime witnesses. No structural moves were
accepted. Applied output noise remained 0.125 throughout the continuation.

The original fixed-6,400 LoRA was evaluated on the identical complete pools,
four private paired panels and both common critics. It still wins both game
judges on every edit subject mean and every preservation subject mean.

| Held-out measure, lower is better | Original LoRA | Particle V2 |
| --- | ---: | ---: |
| Test G loss, common frozen step-1,856 critic | 0.912823 | 0.946709 |
| Test G loss, common final step-6,400 critic | 0.924374 | 0.956669 |
| Preservation G loss, common frozen critic | 0.710419 | 0.729414 |
| Preservation G loss, common final critic | 0.708120 | 0.727027 |
| Test velocity RMSE, evaluation only | 0.073333 | 0.080942 |
| Preservation velocity RMSE, evaluation only | 0.034355 | 0.047450 |

The new particle model improves diagnostic test RMSE by about 5.90% versus
the historical V1 particle model at the same 6,400-update budget (0.086016).
It narrows the diagnostic RMSE gap to original LoRA from 17.29% to 10.38%.
This compares complete formulations and cannot attribute improvement to the
optimizer update alone. It also does not establish a common-critic V1/V2
game comparison at 6,400; the common-critic comparison above is V2 versus
the original LoRA. Output error never selected this configuration or checkpoint.

At the fixed horizon, the preservation training probe shows only 0.19%
net improvement across its last four measurements, triggering the diagnostic
possible-plateau flag. The edit probe shows 0.79% improvement across that
window, with a small late reversal visible in the curve. These observations
do not retrospectively alter the fixed stopping point.

The separate fixed-RMS-key continuation repaired a local routing Jacobian
but worsened complete held-out edit and preservation games versus its matched
native control. Changing preservation critic/controller units improved
preservation at a small edit-game cost. Neither intervention was promoted.
Their bounded results are in `e22_supra_routing_convergence_review.md` and
`e22_supra_v2_critic_review.md`. Passing correctness tests does not establish
that a proposed intervention converges faster.

Receipts: `outputs/e22-particle-v2-6400/qualification-review.json` and
`evaluation-original-6400-game/receipt.json`. The dashboard's final table uses
complete held-out evaluation, separately from its small training probes.

The final particle-contribution audit matches every clean context's game
scores exactly with this full-pool evaluation, then removes codes at all 71
sites or replaces queries with constant mass-only routing. Under the final
critic, code removal worsens mean test game loss by 0.049189 and preservation
by 0.014721; mass-only routing worsens them by 0.047750 and 0.002690.
Both interventions worsen all six edit and all three preservation subject
means under both critics and every aggregate paired panel. Particles remain
useful after the longer linear-path training. Preservation effects are mixed
per context: code removal improves 27/60 and mass-only routing improves 35/60
under the final critic, despite worse means. This is average contribution on
the fixture, not universal protection.

Independent CPU qualification recomputed all 600 clean judge-context matches,
7,200 paired deltas and 990 pool/subject aggregates; source and input hashes
match the runtime witnesses. Complete native state, served model, gradients,
critics and RNGs stayed unchanged. Artifacts are
`particle-contribution-final6400.json` and
`particle-contribution-final6400-review.json` within the final run directory.

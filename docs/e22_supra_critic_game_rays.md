# Frozen critic game shape at native V2 step 6,400

The saved conditional critic consistently favors shrinking the actual particle
residuals in this fixed diagnostic. No positive-ray rank reversal, large-scale
flat region, or incorrect pooling/coordinate unit was found. Its uneven patch
gradient allocation remains measurable, but first hidden saturation does not
explain that allocation by itself. This supports continuing the particle-gated
architecture experiment before changing the critic blindly.

## Controls and evidence

The inputs are the qualified `outputs/e22-particle-v2-6400/final.pt` checkpoint,
its cached dataset, and the previously qualified native BF16/CFG3 capture
`outputs/e22-particle-final-critic-causal/residuals.pt`. ParticleGAN is pinned to
`cabe2084284db923d525918cbf3e18de6f20faac`; production source, particles,
training state, optimizers and structural guards are unchanged.

The capture contains four upcoming edit contexts, four upcoming preservation
contexts, four test contexts, and four native per-site DV12 residual samples
per context. A single private CPU72 generator supplies four paired Gaussian
panels per pool at the saved output sigma 0.125. These are a diagnostic
quadrature, not repeated training runs or alternative training seeds.

For every clean/DV12 residual, the predeclared multipliers are
`{-1, -0.5, 0, 0.25, 0.5, 0.75, 1, 1.5, 2, 4}`. Both the saved D6400 and the
saved D1856 are evaluated on these **same final6400 residuals and panels**;
old1856 residual measurements are not reused. The critic runs in CPU FP32.

The native game is
`mean_panels softplus(D(noise, condition) - D(noise + alpha*residual, condition))`.
The condition is the captured frozen source embedding plus flow time. At
alpha0, identical real/fake inputs make the game exactly log2. Positive
`dG/dalpha` means native game descent favors shrinking the current residual.

Evidence is in `outputs/e22-particle-final-critic-causal/game-rays/`:
`plan.json`, `receipt.json`, `raw-rays.pt`, `progress.jsonl`, and
`independent-review.json`. The independent CPU review qualified 24 source
files, six inputs, all 120 ray cases and 1,200 context-ray points. Critics,
input files and global CPU RNG remain unchanged; no optimizer or native
controller is constructed or advanced. Output errors never enter an update,
guard, stopping decision or checkpoint selection.

## Native ranking and derivatives

All 840 positive-ray adjacent context comparisons across the two critics are
monotonic. There are no outward radial derivatives at any sampled alpha>0,
including all 48 correlated DV12 variants per critic. The D6400 clean means
are:

| Selected pool | Game at alpha0 | At alpha0.25 | At alpha1 | At alpha4 | dG/dalpha at alpha1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Edit fit | 0.693147 | 0.698913 | 0.792498 | 1.674282 | 0.188706 |
| Preservation | 0.693147 | 0.694457 | 0.717816 | 1.170095 | 0.058532 |
| Edit test | 0.693147 | 0.705851 | 0.875800 | 2.056435 | 0.304711 |

At alpha1, only two of 1,024 fit patches have an outward radial contribution;
none of the clean test or preservation patches do. Native input gradients
therefore remain mostly restorative even before summing over patches. This
does not establish that a finite Adam parameter step improves another batch.

At exact origin, the fixed quadrature has a small outward derivative in all
eight clean edit contexts, and 44/60 D6400 clean/DV12 cases overall. Mean
origin derivatives are -0.005106 for fit and -0.015134 for test. Thus alpha0 is
the best **sampled** point, but it is not an exact stationary minimum along
every ray. Finite Gaussian sampling and an imperfect transient critic can
both produce this effect; this diagnostic does not separate them. It cannot
explain the much larger current residual gap by itself, because all sampled
positive multipliers from0.25 onward have positive restoring derivatives.

## Patch allocation and pooling

The output head is linear after mean feature pooling. Its game can therefore
be written as `softplus(mean_patch local_paired_gap)`. The observational
comparison `mean_patch softplus(local_paired_gap)` applies the same learned
local critic and sigmoid link before pooling. It changes neither the saved
critic nor any training objective in this diagnostic. Jensen's inequality
holds in every recorded case.

At alpha1 on the four clean test contexts:

| Diagnostic patch group | Raw residual power share | Native game-gradient power share | Observational tokenwise share |
| --- | ---: | ---: | ---: |
| Lower half | 12.07% | 33.89% | 27.32% |
| Largest10% | 52.42% | 14.31% | 22.10% |

The observational tokenwise sigmoid raises the tail's share, but it does not
remove the imbalance or show a trained-model improvement. Each patch's
gradient direction has the same learned local-score derivative in both
forms; the difference is its positive sigmoid weight. The native pooled game
shares that weight across the context. Raw residual bins are used only to
describe the existing allocation, never as optimizer weights or selection.

The test-tail first-tanh mean derivative is0.309, versus0.305 in the lower
half. Second-tanh derivatives are0.292 versus0.349, respectively. The tail
has *less* second-tanh saturation above magnitude3 (24.53% versus33.31%).
These values do not support a simple claim that the largest patches lose
their signal because their first hidden activations saturate. Replacing that
activation with SiLU is not justified as a demonstrated fix by this probe.

The earlier bounded second-feature bypass also failed its matched128-update
D-only test. It learned negative gains under opposing game/KA2 gradients.
Combined with the negative bank-trust continuation, this rules out claiming
that either bypassing a tanh or suppressing locally harmful bank moves has
already repaired the particle formulation.

## Units and numerical qualification

The independent reviewer reconstructs native scores, all game summaries and
monotonic comparisons from the saved raw records. It also checks the full
analytic input-gradient chain, including the native `1/256` pooling factor,
four-panel mean, sigmoid derivative and raw-coordinate `1/scale` conversion.
Maximum radial discrepancy is5.96e-8; independent FP64 central finite
differences at alpha0/1 differ by at most9.74e-8.

Pooling the linear score before versus after features is equal in FP64 to
2.78e-15. FP32 paired score subtraction differs by at most1.18e-6 because
preservation scores have a common offset near6.6. The initial diagnostic
used an absolute1e-6 check, stopped on1.065e-6 roundoff, and was preserved in
`game-rays-aborted-pooling-roundoff/`. The completed run proves the algebraic
identity in FP64 and checks FP32 differences against an explicit score-scaled
roundoff bound. Native game values and monotonicity tolerance are unchanged.
This was a diagnostic arithmetic qualification issue, not a production bug.

## Conditional next step

Keep the active particle-gated V3 fixed-budget experiment first. A held
prototype in `scripts/experimental_e22_tokenwise_game.py` isolates the measured
pooling difference: only D/G payoffs use
`D.score(D.features(error, condition)).flatten(0, 1)` in the public RpGAN loss.
This changes the pairing granularity from context to context/patch. It is an
explicit experimental game objective, not a production optimizer bug fix.

The pooled `ConditionalTokenCritic.forward` is unchanged for native KA2,
including real R1, caps and16-coordinate input units. Critic parameters,
initialization, EMA, bounded structural features, source/time conditioning,
128x4 particles, routes, native lifecycle, controller observations, Gaussian
and DV12 streams, preservation payoff weight0.1, and unweighted penalty
remain native. No output-derived weights or features are introduced.

Five meaningful CPU checks pass on PR223, including native legacy and shared
auto/settled profiles. They check paired origin, homogeneous payoff/gradient
equivalence, heterogeneous Jensen/gradient/pairing identities, exact native
KA2 value/parameter-gradient/EMA/record equivalence on a shared initial pair,
matched five-update sampling/noise streams, and exact two-update full state
replay through an actual preservation update. A loop configuration tag and
versioned checkpoint wrapper reject legacy or different-objective resume.
Ordinary native checkpoints must be explicitly restored before opting in;
the experimental module's update/checkpoint/restore entry points are then
used together. No production entry point or source was edited.

This prototype has not trained a real Supra model. Its CPU checks exercise
small native routed fixtures; there is no GPU result or win claim. No GPU
experiment, checkpoint choice or promotion follows from those checks.
If the predeclared longer V3 game results justify the next causal experiment,
test this objective with a matched fixed budget and both shared critics.
The observational tail-share numbers do not establish a training benefit.

The isolated prototype qualification command was:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
  PYTHONPATH=/ml2/hypergan/ParticleGAN-pr223-supra \
  /ml2/ntc-image-studio/.venv-anima/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_experimental_e22_tokenwise_game.py
```

Result: **5 passed in2.56s**. Only game aggregation is experimental; there is
no output-MSE objective, saliency weight, structural output guard, early
stopping, seed sweep, or metric-driven checkpoint choice.

## Reproduction

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
  /ml2/ntc-image-studio/.venv-anima/bin/python \
  scripts/diagnose_e22_supra_critic_game_rays.py
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
  /ml2/ntc-image-studio/.venv-anima/bin/python \
  scripts/review_e22_supra_critic_game_rays.py
```

Both commands preserve existing artifacts; use a fresh `--output` diagnostic
directory, then pass it as `--run` to the reviewer when repeating.

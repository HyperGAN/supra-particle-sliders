# Native reopening replay

The editing-only V3 run regressed after a generator ladder release and a
later optimizer-surprise response. An isolated replay identifies both as
harmful on this fixed trajectory. It retains the particle architecture,
learned RpGAN/KA2 game, native guards and fixed update-6,400 endpoint.
Output metrics do not enter this diagnostic.

Starting from the qualified update-6,000 checkpoint, all 400 native update
rows reproduce exactly, including losses, contexts and Gaussian/DV12
streams. The final complete native digest and every raw final game record
match the original. Each counterfactual then forks that replay's complete
native state immediately before its specified transition. Hooks are scoped
to that owner and event in an isolated diagnostic process; no upstream
source or completed training artifact changes.

The earlier DRIFT decision is observed after update 6,222. It doubles the
generator LR from 2.5e-5 to 5e-5 starting at update 6,223. The cancellation
arm preserves the native decision, evidence and subsequent controls, but
holds that one scale increase. Later optimizer decisions may diverge as
consequences. In particular, the later surprise fire does not occur in
this arm, so its total causal effect includes avoiding that event.

The later detector records completed step 6,387; its actions occur before
update 6,388. Separate arms retain detector/guard firing bookkeeping while
omitting second-moment rescaling, anchor-latch creation, ladder reopening,
or all three. All arms retain the sampled input, Gaussian and DV12 streams
through their predeclared update-6,400 endpoint. Controller transformations
and learned parameters may subsequently differ; these are consequences,
not extra controlled interventions.

Held-out editing game over all 240 contexts, lower is better:

| Isolated replay law | Common D1,856 | Common historical V2 D6,400 |
| --- | ---: | ---: |
| Original native replay | 0.964124 | 0.977456 |
| Cancel only generator LR increase after 6,222 | 0.920431 | 0.931124 |
| Omit only moment rescaling before 6,388 | 0.946525 | 0.956402 |
| Omit only ladder reopening before 6,388 | 0.941899 | 0.954734 |
| Omit only anchor-latch creation before 6,388 | 0.964598 | 0.972666 |
| Omit all three surprise actions before 6,388 | 0.941309 | 0.951419 |

Cancelling the earlier generator increase improves 215 of 240 contexts
and all six subject means under both common critics. Later moment and
ladder actions also cause harm in this replay. Anchor omission alone
disagrees between judges, so anchor release is not a robust explanation.
The strongest arm still trails ordinary LoRA's 0.912823 / 0.924374.
These counterfactuals establish effects on one trajectory, rather than
select a production policy or prove every rate increase harmful.

An additional independent first-step comparison links this effect directly
to the generator update: its released/cancelled displacement norms are
0.19391216285 / 0.09695608071, a ratio of 2.000000015 within FP32 rounding.
All five roles' Adam moments remain bit-exact after that update, and every
other FAST owner's update is bit-exact. Only the generator step is doubled.
This check is recorded separately in `first-g-release-step-review.json`.

The action snapshots confirm that none of these actions directly writes
FAST parameters. Harm develops through subsequent optimizer steps.
The late response doubles router LR from 2.5e-5 to 5e-5 and increases the
applied critic LR from 0.002873305 to 0.003826028. Generator and table LRs
are already at their ceilings; their moments still shrink by their own
observed ratios squared. The critic stays active even when its raw ladder
scale is tiny because its applied LR includes table support and damping.
The qualifying excursion also has real router contraction, so the critic's
raw-versus-applied scale distinction does not alone explain permission.

Runtime checks pass all 1,806 assertions. Independent CPU review passes
26,087 checks covering replay provenance, all six fixed arm states, source
and immutable input ownership, all raw game aggregates, streams, actual
moment/first-moment/FAST changes, KA2 latch and five ladder effects. One
cross-fork scalar is explicitly excluded: `RobustCriticAnchor.decay` is
derived scratch state, absent from checkpoints and overwritten before each
EMA update. Within-action content checks remain strict. It is not a
learning-state intervention.

Artifacts:

- `outputs/e22-particle-gated-late-reopen-causal/receipt.json`
- `outputs/e22-particle-gated-late-reopen-causal/independent-review.json`
- `scripts/diagnose_e22_supra_late_reopen.py`
- `scripts/review_e22_supra_late_reopen.py`

Native source is pinned to bdf05d1b. PR223's checked head bc9d9aec has
identical native policy, continuous controls and recipes.

The ParticleGAN test-suite reproducer now lives in the separate worktree
`/ml2/hypergan/ParticleGAN-supra-reopen-toy`:

- `examples/e22_stiff_game_reopen.py`
- `tests/test_e22_stiff_game_reopen.py`
- `docs/e22_stiff_game_reopen.md`

It uses a two-coordinate, constructed stationary feature-score critic,
public RpGAN loss, native generator AMSGrad and native `SettleTest`. The
optimizer snapshot is specified in local spectral units; two tests reproduce
its moments through 1,000 actual Adam updates. A weak coherent direction
permits native DRIFT release while a stiff direction requires the contracted
rate. Native release drives peak paired game above 4,406; holding that one
increase leaves it near log(2). A second geometry permits a safe native rate
increase, so disabling all increases cannot satisfy the suite. Three exact
CPU recovery cases cover the transition.

Independent focused runs give seven passes and one strict expected failure
in about two seconds. `--runxfail` exposes exactly the game-stability failure.
This is an explicit native-controller counterexample linked to the proven
Supra action, rather than an extraction of Supra's Hessian or proof that
its particular stiff/soft geometry is identical. No native production fix
has been selected here.

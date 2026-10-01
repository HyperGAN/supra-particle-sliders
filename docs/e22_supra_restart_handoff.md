# Restart handoff — 2026-10-01, 23:06 UTC

The user requested a restart to repair sandbox access. Training is still
running outside this shell's process namespace. **Do not start another run.**
The latest observed status is 4,300 editing updates, with no failure artifact.
Read the current files again after restarting; this is a dated observation.

Run directory: `/ml2/hypergan/supra-e22-neutral-init/outputs/e22-supra-neutral-initialization-6400`.
Log: the same path with `.log` appended. Dashboard: `http://pop-os:8784`.
The run has two fresh interleaved native particle arms, sampled control and
H/b-only neutral initialization. Both must finish all 6,400 updates. The full
held-out endpoints are fixed at 5,120 and 6,400; do not select an earlier peak.

At the latest 4,200-update fixed 12-context probe, control scored
0.8110781858364741 and neutral scored 0.8058221737543741. Historical ordinary
LoRA scores 0.791677887241 on this probe. These are progress observations,
not the full endpoint verdict. The historical full 240-context reference
scores are ordinary 0.9128064962724844 / 0.924385150273641, particle V2
0.9466044135391712 / 0.956588652729988, and particle V3
0.9643095632394155 / 0.9776970190306504 under D1856 / D6400 respectively.

## Held identities and pending verification

Training source was committed and pushed as `69996d6`. The application branch
is `codex/supra-neutral-init`; local review/diagnostic commits `1b30a18` and
`3fd22be` remain unpushed. Do not edit held training source or the protocol card.
Native checkout is `/ml2/hypergan/ParticleGAN-supra-neutral-develop`, revision
`6ec7e5788e14ea15ddc3e16ac71110458108b6a6`, Python source digest
`d0be475a055aae953c22dbd19f6d2357db041437e0679d2f2f2cfa356517f65b`.
Protocol SHA is `31c1f596f9d3ba2ff600115311029ec3fa3b895a0925d9bcd35413049f5195cf`.
The public `particlegan.init.initialize_` API is used. Particles remain trainable;
training is editing-only. Output error never drives optimization, structural
guards, stopping or checkpoint/head selection.

After both arms reach 6,400 and `status.json` reports `complete`, run:

```bash
/ml2/ntc-image-studio/.venv-anima/bin/python /ml2/hypergan/supra-e22-neutral-init/scripts/review_e22_supra_neutral_initialization.py --run /ml2/hypergan/supra-e22-neutral-init/outputs/e22-supra-neutral-initialization-6400 --particlegan-root /ml2/hypergan/ParticleGAN-supra-neutral-develop
```

No review process is currently running. This CPU artifact reviewer writes
`independent-review.json`, checks all 38 states, matched streams, recorded
scores and exact exported tensors; it performs no model forwards or training.
GPU recovery/full-output checks are source-bound runtime witnesses, rather
than fresh CPU full-model replays. Eight corruption tests passed. An earlier
partial review was interrupted after 617.26 seconds; it grants no qualification.
Do not repeat bulk partial reviews during training. External CPU review cost
is reported separately from the GPU run's 14,400-second budget.

Report neutral-minus-control and neutral-minus-ordinary under BOTH fixed
critics at BOTH endpoints, including all six subject means and signed particle
ablations. Historical ordinary used a different initialization/schedule and
5,120 edit plus 1,280 preservation updates. Only the new 5,120 endpoint matches
its editing count. Output RMSE is a separately labeled diagnostic.

## Conditional ParticleGAN toy

Existing PR227 targets develop: https://github.com/255BITS/ParticleGAN/pull/227.
Worktree `/ml2/hypergan/ParticleGAN-convergence-toy-develop`, branch
`codex/routed-convergence-toy`, local HEAD `1cce9072`. Three new preparation
commits remain unpushed. Native package and original qualified PR227 sources
are unchanged.

If independently qualified neutral Supra at 6,400 does not beat historical
ordinary under BOTH mandatory critics, enable exactly the prepared rotated-
teacher task once. First bind the qualified Supra receipt and review SHAs into
the card, freeze the resulting identities, then set execution authorization.
The current card is `docs/e22_routed_convergence_rotated_teacher_v1.json`, SHA
`36e8694e48ea02bf05854dad6dab8d519c58f9d7db5776da6fc0b14327791b8a`;
`execution_authorized` is false and no quality toy run has occurred.

Its three CPU native-game arms each train 6,400 updates; all four actual
baseline critics, both endpoints, 105 states, 102 curves, particle participation
and exact recovery are mandatory. Budget is 900 seconds per arm / 2,700 total,
including independent review. Use the runner and independent reviewer in
`examples/`; no fresh MSE arm. Preserve earlier PR227 evidence. The task changes
teacher basis, targets and derived coordinate scales together. Its rho near
0.0276 is approximately chance overlap in Supra but more severe than toy
rank2/width16 chance 0.125; a gap is not unique proof of angle causality.

## Additional read-only audit findings

No additional integration bug was established. Whitening scales are
0.2714–0.3959, none at the floor; source RMS 0.0453–0.0496 is close to toy 0.052.
Actual source embedding cosines 0.734–0.915 exceed the toy's -0.100–0.375.
Full Supra also includes 71-site attention/LayerNorm coupling and the CFG
tangent `3 J_cond^T v - 2 J_uncond^T v`, absent from the two-site toy.
Independent DV12 perturbations affect both CFG halves, but observed clean-to-
DV12 game differences at 4,000 were small (+0.000229 / +0.000771).
These are hypotheses and configuration differences, not demonstrated causes.

A bounded future diagnostic can decompose the unchanged learned-game gradient
at fixed saved 800/6,400 states and existing balanced fit contexts, reporting
conditional/unconditional cosines and cancellation ratios for each site,
bank and router with zero updates and immutable state/RNG.

Frozen-critic scores need an inferential limit: zero residual has game log2,
but saved critics have nonzero origin gradients. A lower frozen game can
therefore accompany a nonzero residual. Final additive score bias cancels in
paired RpGAN. Current full endpoint files contain scalar game/MSE observations
and exports, but no endpoint residual-ray calibration; do not equate every
game reduction with target-accuracy improvement. PR226 changed examples/tests/
docs, not native Python at its inspected head. Saved parity tests show that
evenizing alone leaves single-draw force; evenizing plus antithetic draws
cancels origin force. No such intervention was added to this held run.

The earlier `docs/e22_supra_critic_game_rays.md` V2 audit found outward origin
derivatives in all eight clean editing contexts (fit mean -0.005106, test
-0.015134), while the sampled rays at alpha >=0.25 were restorative. First
PR227's initial-residual reference rays increase under all four judges, yet
neutral's diagnostic RMSE 0.034320 remains above the historical MSE reference's
0.031274 despite winning the learned games. Neither observation uniquely
explains the current full convergence gap. A separate post-run signed ray
audit of actual endpoint residuals would require immutable full-model output
capture, since the current evaluator does not retain those residual tensors.

## Environment restrictions to retain

`nvidia-smi` fails in this shell, yet shared run artifacts continue advancing.
GitHub retry fails because `github.com` cannot resolve. Do not claim local
commits were pushed. Browser security rejected opening `http://pop-os:8784`
because permission was denied; do not work around that denied browser action
through another browser, raw HTTP or CDP. The live page has not been visually
verified in this restarted sandbox.

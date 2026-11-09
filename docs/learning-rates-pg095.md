# Measured PG0.9.5 learning rates

**Historical response-error diagnostic.** Current finished-output guidance is [G0.036 with its retained effect evidence](finished-output-learning-rates.md); use that saved result without another search or confirmation. All numerical SSE tables, checkpoints and original source identities below remain unchanged.

The historical Final boss native256 response-error study selected generator rate **`0.024`**
and critic rate **`0.018`** in bounded sequential screens, with no particle bank. Its
original options were `lr=0.024` and **`d_lr_mult=0.7499999999999999`**; that multiplier
preserves the exact tested binary floating-point critic value. This is not the current effect selector.

The [options overlay](../configs/supra/final-boss-pg095-native256-research-fp32-128.options.json)
selects 128 updates, batch4, rank16 and FP32 base precision. Apply it to verified
options for the matching source-bound research profile. The qualified backend is
`supra095-native256-raw-response-research-v1`, with native response tensors
`[1,4,32,32]`, CFG3, conditional-only adaptation and response scale
`0.32837046474230014`. Public inference replay used `open_inference` with the
explicit research backend. The public `supra` backend keeps its own geometry and
native admission contract.

## Original generator screen and held-out confirmation

All candidates used the same fresh initialization, teachers, data and numerical
profile, a common128 schedule horizon and exact critic rate0.018.

| Generator rate | RAW error at32 updates | RAW error at64 updates |
| --- | --- | --- |
| 0.012 | 0.6602348990234086 | — |
| 0.024 | 0.5267898271233262 | **0.40612875052157144** |
| 0.036 | **0.4996069120597182** | 0.4206344450578483 |
| 0.072 | 0.7335815485814126 | — |

The matched64 comparison selected0.024. Exact continuation from its64 checkpoint
completed128 updates, scoring **0.4766090536445644** on development. The frozen
RAW export then scored **0.5118560854877398** on six separately sealed held-out
examples, pooling four native response positions within each example (24fields).
All six teacher examples had positive edit energy. An independent saved-tensor
audit reproduced the score exactly. OFF scores1.0.

That historical response-error metric casts student, teacher and neutral tensors to FP64 before
subtraction, pools squared error and teacher-edit energy within each example,
divides per example, then averages examples equally. Teacher-only zero-energy
examples are undefined and excluded without an epsilon; every student response
must be finite. Preservation, late/weak-field and media diagnostics do not gate
selection or confirmation.

## Critic screen at fixed generator rate

A later development screen fixed G at `0.024` and tested the critic rates below,
using the same initialization, teachers, numerical profile and horizon128.
The original D `0.018` endpoints keep their genuine source and were reused
without replaying any completed training prefix.

| Critic rate | RAW error at32 updates | RAW error at64 updates |
| --- | --- | --- |
| 0.002 | 1.1456837587274709 | — |
| 0.006 | 0.6952588251088505 | — |
| 0.018 | 0.5267898271233262 | **0.40612875052157144** |
| 0.054 | **0.4520688916433381** | 0.47413260807143265 |

The matched64 comparison retained D `0.018`, an interior rate. Its genuine
128-update checkpoint remains the development winner, scoring
`0.4766090536445644`; no additional128 training or proof replay was needed.
Three fresh32 runs and one32→64 continuation added128 actual updates.

The winner was frozen before evaluating a **new** six-example held-out cohort:
three new subjects and two seeds, with four response positions on predeclared
four-step native trajectories. RAW error was **`0.5730644137290212`**, below
OFF `1.0`, across all24 saved fields. An independent CPU FP64 reduction reproduced
that score exactly. All six teacher examples had positive energy; every student
was finite. Confirmation performed zero training updates, using public RAW
inference and exact OFF/base checks with training constructors forbidden.

This new confirmation and the earlier `0.5118560854877398` confirmation use
different cohorts and trajectory positions, so their scores are not direct
comparisons. No confirmation score was used to choose or retune a rate. The
[machine-readable critic evidence](lr-evidence/supra-pg095-native256-critic-screen-v1.json)
retains the original producer identities and the new source/measurement hashes.

For fresh runs, the options overlay now states `ema_decay: null` explicitly;
export and inference select RAW weights. Exact old checkpoints keep their original
serialized source and configuration. These are bounded sequential G and D
measurements, not a claim of a global or jointly optimized rate.

## Reproducible scope

The [machine-readable record](lr-evidence/supra-pg095-native256-fp32-128.json)
contains exact rates, all per-example confirmation sums, source and numerical
identities, checkpoints, RAW export, native continuation/replay proof and the
independent audit digests. The measured source is product `e0d4814`, core0.13.0
`3dd672a` and official ParticleGAN0.9.5 `release-0.9.5`, using Torch2.13.0+cu126.
Training and confirmation used the same frozen base and fresh PG0.9.5 cache.

These sequential G and D measurements apply to this native256 research
concept, precision and128-update budget. Use fresh evidence for other concepts,
geometries or budgets. The [historical PG0.9.4 measurement](learning-rates.md)
retains its original options and source qualification.

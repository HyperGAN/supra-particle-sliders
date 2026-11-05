# Measured PG0.9.5 learning rates

The Final boss native256 research recipe has a held-out-confirmed generator rate
of **`0.024`** at fixed critic rate **`0.018`**, with no particle bank. Use
`lr=0.024` and **`d_lr_mult=0.7499999999999999`**; that multiplier preserves the
exact tested binary floating-point critic value.

The [options overlay](../configs/supra/final-boss-pg095-native256-research-fp32-128.options.json)
selects 128 updates, batch4, rank16 and FP32 base precision. Apply it to verified
options for the matching source-bound research profile. The qualified backend is
`supra095-native256-raw-response-research-v1`, with native response tensors
`[1,4,32,32]`, CFG3, conditional-only adaptation and response scale
`0.32837046474230014`. Public inference replay used `open_inference` with the
explicit research backend. The public `supra` backend keeps its own geometry and
native admission contract.

## Selection and held-out confirmation

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

The sole metric casts student, teacher and neutral tensors to FP64 before
subtraction, pools squared error and teacher-edit energy within each example,
divides per example, then averages examples equally. Teacher-only zero-energy
examples are undefined and excluded without an epsilon; every student response
must be finite. Preservation, late/weak-field and media diagnostics do not gate
selection or confirmation.

## Reproducible scope

The [machine-readable record](lr-evidence/supra-pg095-native256-fp32-128.json)
contains exact rates, all per-example confirmation sums, source and numerical
identities, checkpoints, RAW export, native continuation/replay proof and the
independent audit digests. The measured source is product `e0d4814`, core0.13.0
`3dd672a` and official ParticleGAN0.9.5 `release-0.9.5`, using Torch2.13.0+cu126.
Training and confirmation used the same frozen base and fresh PG0.9.5 cache.

This is a measured G recommendation at fixed D for this native256 research
concept, precision and128-update budget. Use fresh evidence for other concepts,
geometries or budgets. The [historical PG0.9.4 measurement](learning-rates.md)
retains its original options and source qualification.

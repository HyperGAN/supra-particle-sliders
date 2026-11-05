# Learning rates

Current training accepts optional `lr` (generator/base rate), `d_lr_mult` (critic multiplier). Omitted values retain the existing recipe defaults. Inputs must be finite positive numbers; booleans are rejected. The shared ParticleGAN recipe constructs every optimizer: D = `lr * d_lr_mult`.

For the default `bispf_optimized` preset, this product keeps G = **0.012**, D multiplier = **1.5**. These are defaults, not a claim of native LR calibration. Selecting `bispf` uses that official preset's existing values (base 0.00425, D multiplier 1, particle multiplier 2). This is an adapter-only game; there is no particle bank and `prior_lr_mult` is rejected.

Common interface: put the same keys in session `options`, for example `"lr": 0.00002, "d_lr_mult": 1.5`. The Supra CLI dispatches a session configuration; set the keys in its JSON options. It has no separate native LR parser. 

Rates become part of the exact shared recipe and checkpoint configuration. Start a fresh cohort/output for a changed rate; resume requires the original rate recipe. Do not relabel a historical packet. The separate closed HNDL build retains its pre-LR borrowed modules through checksum-verified `cohorts/pre-lr-hndl` archival build routing; this change does not retune that cohort or rewrite its source hashes.

Native calibration compares identical initialized state, data, stochastic streams, precision, declared schedule horizons and completed-update budgets on the actual pretrained model. Select and independently confirm rates using only held-out raw teacher/student relative SSE: cast student, teacher and neutral to FP64 before subtraction, pool positions within each example, divide by that example's teacher-edit energy, then average examples equally. Keep the teacher-only valid/excluded set fixed across candidates; zero-energy examples remain undefined, and student failures never remove examples. See the [common LR protocol](https://github.com/HyperGAN/particle-sliders/blob/main/packages/particle-sliders-core/LEARNING-RATE-CONTRACT.md).

Preservation/no-edit drift, late/weak-field scores and native media outputs are optional diagnostics; they cannot select, reject or delay a rate. After the native mechanical proof, candidate endpoint net displacement is diagnostic too. Retain exact source/model/data/numerical identity, actual optimizer-rate, finite native-update, checkpoint-continuation and raw-response replay checks. Tiny CPU software checks are not native LR evidence.

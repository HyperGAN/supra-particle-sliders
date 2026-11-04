# Learning rates

Current training accepts optional `lr` (generator/base rate), `d_lr_mult` (critic multiplier). Omitted values retain the existing recipe defaults. Inputs must be finite positive numbers; booleans are rejected. The shared ParticleGAN recipe constructs every optimizer: D = `lr * d_lr_mult`.

For the default `bispf_optimized` preset, this product keeps G = **0.012**, D multiplier = **1.5**. These are defaults, not a claim of native LR calibration. Selecting `bispf` uses that official preset's existing values (base 0.00425, D multiplier 1, particle multiplier 2). This is an adapter-only game; there is no particle bank and `prior_lr_mult` is rejected.

Common interface: put the same keys in session `options`, for example `"lr": 0.00002, "d_lr_mult": 1.5`. The Supra CLI dispatches a session configuration; set the keys in its JSON options. It has no separate native LR parser. 

Rates become part of the exact shared recipe and checkpoint configuration. Start a fresh cohort/output for a changed rate; resume requires the original rate recipe. Do not relabel a historical packet. The separate closed HNDL build retains its pre-LR borrowed modules through checksum-verified `cohorts/pre-lr-hndl` archival build routing; this change does not retune that cohort or rewrite its source hashes.

Native calibration must compare identical data, seeds, fixed update budgets and held-out edit/preservation evaluation, record the resolved recipe and exact source identity, and qualify any selected values on the actual pretrained model. Tiny CPU software checks are not native LR evidence.

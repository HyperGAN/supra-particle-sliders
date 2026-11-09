# Learning rates

Current measured guidance: [finished-output recommendation](docs/finished-output-learning-rates.md) and [portable values](docs/finished-output-rate.json). These scoped rates do not change package defaults.

Current training accepts optional `lr` (generator/base rate), `d_lr_mult` (critic multiplier). Omitted values retain the existing recipe defaults. Inputs must be finite positive numbers; booleans are rejected. The shared ParticleGAN recipe constructs every optimizer: D = `lr * d_lr_mult`.

For the default `bispf_optimized` preset, this product keeps G = **0.012**, D multiplier = **1.5**. These are defaults, not a claim of native LR calibration. Selecting `bispf` uses that official preset's existing values (base 0.00425, D multiplier 1, particle multiplier 2). This is an adapter-only game; there is no particle bank and `prior_lr_mult` is rejected.

Common interface: put the same keys in session `options`, for example `"lr": 0.036, "d_lr_mult": 0.5`. The Supra CLI dispatches a session configuration; set the keys in its JSON options. It has no separate native LR parser.

Rates become part of the exact shared recipe and checkpoint configuration. Start a fresh cohort/output for a changed rate; resume requires the original rate recipe. Do not relabel a historical packet. The separate closed HNDL build retains its pre-LR borrowed modules through checksum-verified `cohorts/pre-lr-hndl` archival build routing; this change does not retune that cohort or rewrite its source hashes.

## Current finished-output selection

Rank rates by one equal-case arithmetic mean of intended-effect intensity ON
minus exact matched OFF in finished outputs; higher wins. Retain negatives,
native zeros and exact ties. Technical failures remain missing evidence.
Teacher/student SSE is diagnostic only; preservation, beauty, confidence and
completion never rank or break ties. Reuse completed searches, saved RAWs,
proofs, teachers, outputs and first scores. Preserve separate fresh results
without pooling; no additional confirmation or worse-arm completion is required.
Keep each comparison's native profile, matched budget and original horizon.
Use RAW current weights with no model-weight EMA; scalar optimizer statistics
are unchanged. Follow the exact product-supported no-EMA configuration.

## Historical response-error protocol (diagnostic only)

Earlier cohorts selected and confirmed rates by held-out RAW teacher/student
relative SSE. Their archived reduction cast student, teacher and neutral to FP64
before subtraction, pooled positions within an example, divided by that example's
teacher-edit energy and averaged examples equally. Teacher-only valid/excluded
membership was fixed; zero-energy relative scores were undefined and student
failures did not remove examples. That historical protocol and all recorded
values retain their original scope; they are not the current finished-output
selector or a requirement for new confirmation. Native source/model/data,
finite-update, continuation and RAW replay evidence stays with each cohort;
reuse valid completed proofs rather than repeat them.


The old response-error studies treated media/preservation observations as optional diagnostics. Current LR selection instead measures the requested effect in finished outputs as defined above.

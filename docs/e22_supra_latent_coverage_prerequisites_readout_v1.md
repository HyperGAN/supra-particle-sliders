# Actual Supra latent coverage: qualified prerequisites

The eight-seed training data and its attachment to the genuine full Supra
12,800-update particle checkpoint are qualified. No additional quality training
has run yet, and the original LoRA remains the full-task score to beat.

| Prerequisite | Result | Whole-process seconds / cap | Stable input/source bindings |
| --- | --- | ---: | ---: |
| Asset-free CPU public wrapper/replay | PASS | 7.207 / 60 | 43 |
| Actual full-teacher data generation | PASS | 101.897 / 600 | 65 |
| Independent CPU saved-data integrity | PASS | 15.776 / 120 | 72 |
| Genuine full12800 attachment/reload | PASS | 186.446 / 300 | 76 |

The original frozen fourteen-block teacher generated six added globally shared
noise seeds, 7065–7070. There were 900 B4 Euler trajectory forwards and 180 B4
opposite-caption forwards, each with internal CFG batch8. The result has 960
ordered FIT rows, 912 distinct packed contexts and 872 distinct latent tensors.
These are 96 source/seed/path trajectories, rather than 960 independent noise
draws. The two paths duplicate their initial time-zero input; all six caption
pairs reuse the same initial tensor for each seed.

The independent CPU reader passed 7,070 assertions. All original240 FIT rows,
cached velocities, prompts and all fields outside FIT are byte-exact. TEST240,
guard64, coordinate scale, text buffers and masks remain unchanged. No exact
packed-context or latent-row overlap exists between expanded FIT and TEST.
This verifies integrity and exact exclusion; it does not establish statistical
independence, teacher-output quality or convergence.

The full-model migration starts from the exact original trained12800 public
checkpoint, after public `init.initialize_` is used only for fresh owners before
optimizers/EMA. Both two-seed and eight-seed attachments preserve their common
original FIT predictions byte-exactly. Independent fresh owners restore their
own migrated checkpoints with identical predictions and complete native/caller
state. Wrong data tags reject before restore without changing state. Finally
rollback to original config, FIT, precision, models, optimizer/controller state
and streams is exact. Six actual public routed forwards were performed; native
updates and optimizer steps were zero.

The original CPU software prerequisite has three tests and three actual native
updates, including genuine clock0→1→2 versus fresh saved1→2 replay. Its synthetic
schema is rejected by production validation. The full migration proof adds
actual full14 ownership/data restoration, but does not independently replay
training with the expanded pool or qualify any accuracy improvement.

The next frozen comparison is full14, 71 shared-Up particle sites, rank16 and a
128×4 bank, from the same trained12800 state through13824: 1,024 editing updates
per arm, two-seed versus eight-seed support. Both retain BF16 training arithmetic,
native learned noise, KA2/DV12 and all learned optimizer/controller/RNG state.
The common960 sampler preserves uniform240 versus uniform960 marginals while
pairing source/time/path addresses; it does not reproduce the historical
`randint(240)` sequence. Guard and scale stay fixed. Accuracy never enters
optimization, structural decisions, scheduling, stopping or checkpoint selection.

The planned whole GPU allowance is2,400 seconds, with1,200 charged per arm.
Scores at13696/13728/13760/13792/13824 use unchanged TEST240, live original BF16
teacher, common FP32 student-final serving, D1856/D6400 and fixed CPU72 panels.
Terminal expanded scores must beat both original arithmetic references in all
three metrics and retain beneficial particles. The paired coverage contrast is
reported separately. Source/card/software/reader qualification remains required
before this comparison begins.

The last qualified full-task best particle RMSE is0.0679143351; the original
LoRA's common-FP32 score is0.0633609846. Neither prerequisite PASS changes that
quality result. The generated two-site public-API coverage toy improved23.09%
and is [draft ParticleGAN PR261](https://github.com/255BITS/ParticleGAN/pull/261);
its actual-Supra transfer remains untested.

Reproduction sources and root-frozen protocols:

- [Full teacher preparation](../scripts/prepare_e22_supra_latent_coverage.py),
  [protocol](e22_supra_latent_coverage_data_preparation_v1.json).
- [Independent saved-data reader](../scripts/review_e22_supra_latent_coverage_data.py),
  [preregistered law](e22_supra_latent_coverage_data_reader_preregistration_v1.json),
  [actual output descriptor](e22_supra_latent_coverage_saved_data_review_v1.json).
- [Actual12800 migration](../scripts/qualify_e22_supra_latent_coverage.py),
  [protocol](e22_supra_latent_coverage_full12800_recovery_v1.json).

Compact bound evidence is in
[`e22_supra_latent_coverage_prerequisites_v1/`](e22_supra_latent_coverage_prerequisites_v1/).
Raw tensor data, native checkpoints, stdout and JUnit remain local under
`outputs/`; they are not added to Git. Saved tensor equality/digests are
independent reader evidence. Actual seed realization,1,080 teacher calls,
pure-teacher execution and full public rollback remain authenticated
producer/source proofs, rather than an independent model replay.

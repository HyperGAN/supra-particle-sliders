# Late table-step game profiles with native and replayed DV12

The subsequent [exact-addition control](e22_supra_table_dv12_exact_offset.md)
removes the FP32 association confound below. Both local sign reversals survive,
with different magnitudes; use that result for displacement attribution.

Fixing the actual G-pass perturbation changes two locally worsening table
steps into improving steps. It does not explain every bad step, establish a
gradient bug, or demonstrate a better training recipe. The comparison also
retains a floating-point association caveat described below.

The starting point is the qualified V2 update-6,400 checkpoint on ParticleGAN
`cabe2084284db923d525918cbf3e18de6f20faac`. The native role autopsy found that
table-only displacements worsen the same sampled generator game on four of
updates 6,401–6,405, despite negative gradient/displacement dot products. This
diagnostic replays those exact five updates and profiles their actual table
displacements without changing training.

Artifacts are `outputs/e22-table-dv12-displacement-6400/profiles.json`,
`profile-evidence.pt`, and `independent-review.json`. The isolated GPU runner
is `scripts/diagnose_e22_supra_table_dv12_displacement.py`; the independent
CPU reviewer is `scripts/review_e22_supra_table_dv12_displacement.py`.

## Intervention and controls

Each update first performs the original native D/G backward and optimizer
step. The diagnostic retains that actual post-D critic and the pre-generator-
step G/router parameters, then temporarily applies six predetermined fractions
of the actual table displacement: 0, 0.01, 0.1, 0.25, 0.5 and 1.
The fractions are diagnostic coordinates, not selected learning rates.

Every fraction is evaluated under three modes:

- Native DV12 recomputes support geometry using the candidate table, with a
  private copy of the actual G-pass random stream.
- Anchored DV12 replays every one of the 71 actual G-pass perturbations as
  `(new_codes - actual_codes) + actual_perturbed_codes`.
- Clean disables the latent perturbation in this temporary diagnostic forward.

All modes use the same actual G-pass paired Gaussian, output sigma,
source/time conditioning and native RpGAN loss. The clean baseline is a
different forward from the noisy baseline; changes are compared to each
mode's own zero-displacement point. No output accuracy metric, structural
guard, optimizer setting or training checkpoint is selected or changed.

Both noisy zero-displacement forwards reproduce the actual G-pass prediction
and each other's score gaps bit exactly. Table fractions one and zero also
reproduce the earlier role autopsy's table-only and no-step game scores
exactly. The actual native state is restored after every temporary panel.

## Complete five-step endpoint result

These are unweighted generator-game changes relative to each mode's own
baseline; negative is improvement. The preservation update still trains with
its original 0.1 weight. Its gradient/displacement product is divided by that
weight below so the reported units agree.

| Update | Task | Unweighted gradient dot actual table displacement | Native DV12 game change | Anchored DV12 game change | Clean game change |
| --- | --- | ---: | ---: | ---: | ---: |
| 6,401 | Edit | −0.0000246239 | +0.0008785725 | +0.0009359717 | −0.0010444522 |
| 6,402 | Edit | −0.0003525034 | −0.0013337135 | −0.0000968575 | −0.0021132231 |
| 6,403 | Edit | −0.0000775778 | +0.0006526709 | +0.0007047653 | −0.0016062260 |
| 6,404 | Edit | −0.0000041703 | +0.0002161264 | −0.0000370741 | −0.0009205937 |
| 6,405 | Preservation | −0.0000006959 | +0.0001069903 | −0.0004873276 | +0.0000904202 |

Native versus anchored endpoint differences are −0.0000573993,
−0.0012368560, −0.0000520945, +0.0002532005 and +0.0005943179,
respectively. Recomputing the perturbation helps the 6,402 sampled game more
than anchored replay; anchoring improves 6,404 and 6,405 enough to reverse
their endpoint signs. The other two native worsening steps remain worsening
under anchored replay. There is no uniform benefit from either mode, and no
long-run convergence claim follows from these local profiles.

## Why this is not yet a derivative diagnosis

The native DV12 local-support calculation uses detached codes and particle
support to set the clipping radius. A table displacement can therefore change
the forward perturbation geometry without that dependence entering the table
backward. The matched comparison measures the resulting native versus replay
forward difference at actual optimizer displacements.

The full host also computes frozen layers in BF16. Its finite-displacement
forward can differ from the cast derivative used in autograd. All three modes
show nonmonotonic game profiles across the predetermined fractions. At only
one percent of the actual table displacement, 47.2–72.4% of native prediction
coordinates already change. These equality/change counts measure sensitivity;
they do not isolate BF16 as its cause, nor measure output accuracy. A negative
gradient dot product and a positive finite-step game change do not by themselves
prove an incorrect derivative or identify an optimizer failure.

There is an additional numerical control limit. Anchored replay uses
`(new_codes - old_codes) + old_perturbed_codes`; native DV12 adds its displacement
as `new_codes + new_displacement`. Anchored replay intentionally makes the
zero point bit exact, but at a nonzero fraction the different FP32 addition
association may contribute to downstream differences alongside changed support
geometry. Therefore the two sign reversals are a result of the matched replay
intervention, not proof that geometry alone causes them. A stricter geometry
control can capture each actual native displacement and replay
`new_codes + actual_displacement`, preserving the native addition order.

## Qualification

All five instrumented native update rows match an uninstrumented reference
replay exactly. Both complete final native digests are
`ede3822d70475b7dc51555d8ecece6d135cfb0253f872865e33c26fb860130e5`,
also identical to the earlier role autopsy. Every temporary profile preserves
active model/averaged-model/optimizer/controller/evidence/penalty state,
gradients, module modes and owned/global random streams. Frozen weights and
executing sources remain unchanged. The capture finished in 39.7 seconds.

The independent CPU review passes 432 checks and reconstructs all 90 raw
game-score panels. Its maximum cross-device FP32 score difference is
`8.94e-8`. It verifies source/input hashes, native data draws, all predetermined
coordinates, finite raw shapes, all 71 actual site captures, gradient and
displacement geometry, paired game deltas, prediction equality/change counts,
bit-exact noisy zero points, and the exact role-autopsy endpoint links.
CUDA BF16 full-model replay and active-state immutability are bounded GPU
runtime witnesses; this CPU review does not rerun GPU training.

The first capture attempt was aborted because an instance-method hook entered
the controller's checkpointed instance dictionary. Its partial output and log
are retained. The successful runner hooks the class method, captures only the
actual training controller's G pass, delegates the original method unchanged,
and restores it afterwards. The native strict checkpoint schema is unchanged.

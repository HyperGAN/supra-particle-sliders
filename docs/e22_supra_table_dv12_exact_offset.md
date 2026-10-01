# Exact-addition DV12 control of late table steps

Changing the DV12 displacement after the table optimizer step causes two of
the five sampled table-step outcomes to change from improving to worsening.
That local result survives a control that preserves the native addition order.
It is a concrete convergence lead, not an explanation of the complete original-
LoRA gap or proof of an incorrect optimizer derivative.

The primary receipt is
`outputs/e22-table-dv12-exact-offset-6400/profiles.json`; raw game panels,
predictions, table gradients/displacements and all actual per-site DV12
displacements are in `profile-evidence.pt`. The independent CPU qualification
is `independent-review.json`. New, isolated scripts are
`scripts/diagnose_e22_supra_table_dv12_exact_offset.py` and
`scripts/review_e22_supra_table_dv12_exact_offset.py`.

This is the same qualified V2 update-6,400 starting checkpoint, ParticleGAN
`cabe2084284db923d525918cbf3e18de6f20faac`, five actual native updates and six
predetermined table-step fractions as the earlier
[anchored replay diagnostic](e22_supra_table_dv12_displacement.md). The earlier
script, reviewer and receipts remain intact. This variant has the explicit
schema `supra_table_dv12_exact_offset_v1`.

## Stronger control

The actual native G pass records every DV12 application. The diagnostic reads
each original record's `displacement` immediately after that application;
it draws no additional training noise and adds no diagnostic record. At every
one of the 355 site applications across the five updates, the recorded
`codes + displacement` reproduces the original perturbed codes bit exactly.

Temporary table profiles keep G/router at their actual pre-generator-step
values and use the actual post-D critic, conditioning, paired Gaussian and
output sigma. They apply fractions 0, 0.01, 0.1, 0.25, 0.5 and 1 of the
actual table displacement under:

- Native DV12, which recomputes local support geometry using the candidate.
- Fixed-offset DV12, which adds `new_codes + actual_Gpass_displacement` with
  the same addition order as native DV12.
- A clean diagnostic forward, with no latent perturbation.

Both noisy zero-displacement predictions and score gaps reproduce the actual
G pass bit exactly. The native zero and full-table-step scores match the
previous role autopsy exactly. Thus the native versus fixed-offset comparison
changes the displacement recomputation while retaining the native FP32
addition, complete sequential routing and frozen BF16 host.

No profile selects an optimizer setting, structural move, checkpoint or output
metric. All actual training follows the original native path and is restored
after each temporary profile.

## All five full-step results

Negative generator-game change is improvement. Each change is paired with its
mode's own no-step baseline; both noisy baselines are identical. Preservation
game values are reported before the original 0.1 task weight.

| Update | Task | Native DV12 game change | Fixed-offset DV12 game change | Clean game change | Native minus fixed-offset |
| --- | --- | ---: | ---: | ---: | ---: |
| 6,401 | Edit | +0.0008785725 | +0.0004613400 | −0.0010444522 | +0.0004172325 |
| 6,402 | Edit | −0.0013337135 | −0.0009620190 | −0.0021132231 | −0.0003716946 |
| 6,403 | Edit | +0.0006526709 | +0.0011789799 | −0.0016062260 | −0.0005263090 |
| 6,404 | Edit | +0.0002161264 | −0.0009851456 | −0.0009205937 | +0.0012012720 |
| 6,405 | Preservation | +0.0001069903 | −0.0002020001 | +0.0000904202 | +0.0003089905 |

At updates 6,404 and 6,405, recomputing the displacement changes the sampled
game outcome from improvement to harm. At 6,401 it worsens a step that is
already harmful with the fixed offset. At 6,402 and 6,403 it helps relative
to fixed-offset replay; the latter step remains harmful either way. The
controller's displacement change is therefore a measured local effect with
mixed signs, not uniformly harmful noise.

The actual table gradient dot displacement is negative at all five updates.
After removing the preservation reporting weight its values are
−0.0000246239, −0.0003525034, −0.0000775778, −0.0000041703 and
−0.0000006959. Native DV12 derives local clipping geometry from detached codes
and particle support. Table changes can alter the applied displacement without
that dependence entering the table backward. This test measures that forward
effect at the actual finite optimizer step; it does not establish whether
differentiating the controller's geometry would improve the game.

## Remaining numerical sensitivity

The exact-offset control removes the earlier FP32 addition-association
confound, but finite optimizer displacements still pass through a BF16 host.
Profiles are nonmonotonic even under fixed offsets and in clean mode. Both
the numerical forward and the learned nonlinear game can contribute, so these
profiles alone do not isolate a BF16 cause or certify a local derivative.

The earlier anchored replay and this exact-offset replay implement the same
intended fixed perturbation under real arithmetic, but differ in how the
already rounded original addition is carried through FP32 calculations. Their
full-step game scores differ by up to 0.0009480715 at update 6,404. This is
material numerical propagation sensitivity in the complete model; it explains
why the exact-addition control was required. It does not independently identify
which later layer or operation amplifies the difference.

At one percent of the actual table displacement, 47.2–72.4% of native
prediction coordinates change. Those equality counts are sensitivity
diagnostics and do not measure output accuracy or isolate rounding as the
cause. No step fraction is promoted based on these profiles.

## Qualification and next question

All five instrumented native update rows and the complete final checkpoint
state match the uninstrumented reference replay exactly. The final native
digest is `ede3822d70475b7dc51555d8ecece6d135cfb0253f872865e33c26fb860130e5`,
also matching both previous local diagnostics. All temporary panels preserve
active model/average/optimizer/controller/evidence/penalty state, gradients,
module modes and owned/global random streams. Frozen weights and sources
remain unchanged.

The independent CPU review passes 437 checks, reconstructs all 90 score
panels, and verifies all 355 captured native offset additions bit exactly.
Its maximum cross-device FP32 score difference is `5.96e-8`. It also verifies
native sampling, finite tensor shapes, every predeclared fraction and mode,
game-weight units, gradient/displacement geometry, paired game changes,
prediction equality/counts, and the exact role-autopsy endpoint links.
Full CUDA/BF16 model replay and active-state immutability are bounded GPU
capture witnesses, not independently repeated CPU training.

The supported next question is whether a native-game training intervention
that controls displacement changes improves matched convergence while keeping
particles and structural safeguards. This diagnostic does not implement or
qualify such an intervention, and provides no reason to add output metrics to
the optimizer or weaken the zero feature-harm guard.

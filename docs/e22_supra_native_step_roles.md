# Native parameter-update attribution after update 6,400

The complete native update is still learning. Each of the five predetermined
updates 6,401–6,405 improves its own unweighted generator game payoff on the
exact Gaussian and DV12 draw used by its backward pass. The particle-bank
portion alone worsens that payoff in four of five updates, despite having a
negative gradient/displacement dot product in all five. This is a concrete
local discrepancy to investigate, not evidence to remove the particles.

The qualified starting checkpoint and source are unchanged:
`outputs/e22-particle-v2-6400/final.pt`, ParticleGAN
`cabe2084284db923d525918cbf3e18de6f20faac`. No output-error metric is used in
this diagnostic. Production code, optimization and structural guards retain
their native formulations.

## Measurement

`scripts/diagnose_e22_supra_native_step_roles.py` first runs five unmodified
native updates. It restores the full starting checkpoint and repeats exactly
those updates while measuring the displacement applied to generator, router
and bank parameters. For each update, all eight subsets of those three
displacements are applied temporarily. The critic is held at its actual
post-critic-step weights for the same-minibatch comparison. The G-pass
Gaussian panel, private DV12 stream, controller geometry and output sigma are
replayed; the no-displacement full-model prediction must match the actual
native G forward bit exactly.

Separate clean comparisons use the same frozen step-6,400 critic and fixed
four-panel CPU Gaussian draws. They cover 12 fit, 12 training-preservation and
12 held-out edit contexts spread across their subjects. These small diagnostic
probes are distinct from the full-pool benchmark panels and do not select a
checkpoint, parameter group or learning rate.

After every panel, parameters and modes are restored. All intermediate model,
optimizer, controller, routing, gradient and random-stream fingerprints remain
equal. Native checkpointing rejects unfinished updates, so the intermediate
fingerprint reads those active owners without pretending to checkpoint at a
completed boundary. The final complete native checkpoint digest and every
training row equal the uninstrumented reference replay exactly.

## Result

Game deltas are relative to applying none of the current G update; lower is
better. The preservation row reports unweighted game units although its actual
backward retains the native 0.1 task weight.

| Update | Task | Complete G/router/bank delta | Bank alone delta | Bank gradient dot displacement |
| --- | --- | ---: | ---: | ---: |
| 6,401 | Edit | −0.00669992 | +0.00087857 | −0.000024624 |
| 6,402 | Edit | −0.01647532 | −0.00133371 | −0.000352503 |
| 6,403 | Edit | −0.02904296 | +0.00065267 | −0.000077578 |
| 6,404 | Edit | −0.00435901 | +0.00021613 | −0.000004170 |
| 6,405 | Preservation | −0.00094694 | +0.00010699 | −0.000000070 |

The generator displacement alone improves its own minibatch in all five
updates, as does the router displacement alone. Their interaction is not
additive. For example, at 6,402 the generator alone improves the game by
0.02527279, while the combined displacement improves it by 0.01647532.
At other updates the combined displacement helps more than generator alone.

Across the four edit updates, the complete displacement improves the fixed
held-out edit probes by a mean 0.00395084. Generator alone improves them by
0.00418986; bank alone worsens them by 0.00013480. The complete update worsens
the fit probe by a mean 0.00161878 despite improving each actual training
minibatch. Across subjects and updates these small probes have mixed signs;
they support investigating local interference and sampling variance, not a
blanket conclusion that any parameter role is harmful. The preservation
update improves training-preservation probes by 0.00015737 and slightly
worsens held-out edit probes by 0.00022397.

## Remaining distinction

The bank mismatch has at least two plausible explanations requiring a matched
test. DV12 clipping depends on detached code/support geometry: an actual bank
step changes the local perturbation while its backward treats that geometry
as fixed. Separately, native BF16 projection boundaries make forward values
piecewise constant at small parameter changes, while casts pass gradients
through. A negative gradient dot displacement therefore does not guarantee
that a finite bank displacement reduces the measured forward game.

The appropriate follow-up replays the same actual bank displacement at
predeclared fractions, comparing native recomputed DV12 with fixed original
per-site perturbations and with the clean game. It must retain the unchanged
native optimizer, source checkpoint, paired draws and guards. Such a diagnostic
can distinguish causes without selecting a new rate from held-out outputs.

## Qualification

Artifacts are in `outputs/e22-native-step-roles-6400/`. The independent CPU
review, `scripts/review_e22_supra_native_step_roles.py`, passes all 16 checks and
reconstructs 160 raw game panels. Largest cross-device score error is
`8.94e-8`; clean FP64 aggregates are checked within `1e-12`. It verifies source
and archived input hashes, exact subset names, finite panel shapes, paired
deltas, native training-row links, predetermined horizon and independently
reconstructed CPU data sampling. Complete CUDA native replay and active-state
immutability are qualified runtime witnesses, not CPU re-executions of GPU
training.

This five-update local result does not explain the entire original-LoRA gap.
It does establish that the complete update has not stopped learning and that
testing actual bank motion is more informative than checking gradient
connectivity alone.

## Completed noise-offset follow-up

The [exact-offset control](e22_supra_table_dv12_exact_offset.md) reuses all
71 actual per-site native DV12 displacements, with the original addition order.
It restores and repeats the same five native updates, with both noisy
zero-displacement predictions and every final native state matching exactly.
At 6,404, the bank-only game delta changes from +0.00021613 with recomputed
offsets to −0.00098515 with fixed offsets. At 6,405 it changes from
+0.00010699 to −0.00020200. Changed offsets therefore cause those two local
worsening outcomes relative to the fixed-offset control.

The other two worsening steps, 6,401 and 6,403, still worsen with fixed
offsets; at 6,402 recomputing offsets helps. Both clean and fixed-offset
profiles remain nonmonotonic at the predeclared displacement fractions.
An earlier algebraically equivalent anchored replay differs materially from
the exact-addition replay, showing that floating-point arithmetic and its
propagation through the BF16 host also affect these measurements.

Neither result selects a new bank learning rate or promotes a change. The
earlier old-V1 DV12-off continuation did not close the original gap either.
These local tests support investigating update/noise/precision consistency,
not a conclusion that removing noise would beat original LoRA.

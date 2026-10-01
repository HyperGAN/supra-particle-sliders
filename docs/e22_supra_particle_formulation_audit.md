# Particle-conditioned input basis

The gated generator makes the actual mixed particle code gate the
rank-16 input features at each of the 71 projection sites. It preserves the
shared 128×4 bank, activation-dependent routing, represented masses, per-site
native DV12, complete candidate reruns, and native E22 game and optimizers.
It is an explicitly opt-in architecture, `gated_particle_v3`; fresh production
training continues to default to `linear_modulated_v2`.

The qualified 400-update comparison improves native game scores over matched
V2 particles. It does not establish that gating repairs the full convergence
gap or beats ordinary LoRA.

## What the completed investigations establish

V2 particles are useful on edits. At update 1,600, removing codes worsens the
complete test game under both common critics, on 182/240 contexts under the
V2 judge and all six edit-subject means. That removal improves 47/60
held-out preservation contexts, however. Mass-only routing improves 54/60 preservation
contexts under the V2 judge. The learned particle contribution has a small
preservation cost; it has not become an irrelevant decoration on the input
path. These are read-only ablations of a trained model, not retrained controls.

The existing matched geometry capture also rules out a global saturation
explanation for V2. Recomputing its 71-site test records gives:

| Measurement | V2-1600 | V2-1856 |
| --- | ---: | ---: |
| Mean saturation, abs(bridge preactivation) > 3 | 0.001849 | 0.002860 |
| Median mean tanh derivative | 0.902607 | 0.892845 |
| Median code / hidden bridge preactivation RMS | 1.357483 | 1.432662 |
| Sites where that code / hidden ratio exceeds one | 51/71 | 51/71 |
| Median centered actual up-input effective rank | 2.355623 | 2.299303 |

These ranks describe covariance on four fixed contexts and correlated tokens.
They do not prove a global capacity limit. RMS ratios compare the two inputs
to the bridge, not their contributions to useful output accuracy.

The raw capture is `outputs/e22-particle-current-geometry/geometry.json`.
The useful edit contribution and preservation qualification are independently
recomputed in `outputs/e22-particle-v2-1600/particle-contribution-review.json`.
Full interpretations are in `e22_supra_particle_hookup_audit.md` and
`e22_supra_routing_convergence_review.md`.

Native key/value norm coupling creates a local concentrated-routing bottleneck.
Its fixed-RMS-key continuation already failed to improve any complete pool in
256 matched updates. Removing DV12 application, correcting preservation D units,
releasing the critic anchor, and removing G-owner AMSGrad also failed to close
the earlier gap in the completed fixed late-stage controls. The critic's
output-space game gradient remains restorative, while large edit residuals
receive relatively weak correction. A bounded critic-bypass D-only experiment
did not repair that allocation. None of these results justify repeating the
same intervention or weakening the zero feature-harm guard.

The exact-offset table diagnostic does isolate a different local mechanism:
changed native DV12 support/offset turns two of five actual late table updates
from helpful to harmful on their same G-pass game. Two other harmful updates
remain harmful under fixed offsets. This is a real local interaction, with
BF16 and finite-step caveats; it does not establish the full-gap cause.

## Formulation hypothesis

Write `h = down(x)` and split the existing bridge weight into `H` for its first
16 hidden columns and `C` for its four code columns. V2 uses

```text
a = Hh + Cz + b
delta_v2 = up(h + tanh(a))
```

Its direct, unsquashed input path is independent of particle codes. In a
linear approximation to the bounded modulation, codes add `up(Cz)` to that
path. The full nonlinear law already contains hidden/code interactions; it
would be incorrect to call it an exactly additive model or claim all of its
particle effects are rank four.

V3 gives particles an explicit multiplicative role:

```text
gate = tanh(Cz)
delta_v3 = up(h + tanh(Hh + b) + h * gate)
```

The product is coordinatewise. Codes now scale the usable input basis with a
hidden gain between zero and two. All trainable matrices and biases have the
same names, shapes, values at initialization and optimizer owners as V2.
The code gate has no separate bias or extra trainable tensor. Zeroing its
mixed code removes precisely the particle-dependent multiplicative term.
Perturbed codes from the native routed execution drive this same gate.

The local derivative identities make the change concrete. Let
`s(v) = 1 - tanh(v)^2`, elementwise, and let `phi` denote the input to `up`:

```text
V2: dphi/dz = diag(s(Hh+Cz+b)) C
V3: dphi/dz = diag(h * s(Cz)) C

V2: d²phi_i/(dh_j dz_k)
      = -2*tanh(a_i)*s(a_i)*H_ij*C_ik
V3: d²phi_i/(dh_j dz_k)
      = indicator(i=j)*s((Cz)_i)*C_ik
```

V3 supplies an explicit feature/code interaction even when its gate starts
near zero. Both pointwise code Jacobians still have rank at most four.
This is a changed conditional function family and conditioning, not an
increase in latent dimension or proof that larger derivatives help. It can
also amplify the output effect of native DV12 on large hidden features.
The matched test retains that noise and reports preservation game as well as
edit game, so this tradeoff is visible.

## Versioning and qualification

`supra/particle_adapter.py` adds the opt-in architecture enum and formula.
The native training factory writes its explicit architecture to configuration;
strict resume rejects interpreting a trained V2 checkpoint as V3. Clean V3
exports use `supra_particlegan_clean_v3`, require `gated_particle_v3`, and reject
conflicting format/architecture metadata. Existing V1/V2 defaults and tensor
schemas remain unchanged.

The focused CPU checks prove zero-start and strength-zero base equality,
complete native routing and later-site recomputation, dense bank/router
gradients, deepcopy/CFG behavior, versioned clean export reload, and that
changing fixed codes changes the projection's input Jacobian. A deterministic
native six-site CPU training/audit also reaches all 128 bank rows through clean
and DV12 game gradients while preserving full native state and existing
gradients. These qualify mechanics; they are not convergence measurements.

The previous hookup helper hooks the unsplit `bridge.forward`; V3 deliberately
uses its two parameter slices directly. Its old whole-modulation hook therefore
cannot qualify V3. The new runner instead captures the actual mixed codes,
`down` outputs and `up` inputs, requires exact reconstruction of the declared
V3 feature law at every site, and separately checks complete-model code and
row interventions and native game gradients.

## Fixed matched training experiment

`scripts/experiment_e22_supra_particle_gated.py` declares exactly 400 fresh
updates using ParticleGAN PR223 `bdf05d1be0f68cfdb0c71e81e7e0d3cce477572f`
and `pr223_shared_routed_v1` for both architectures. It uses only frozen teacher
tensors and cached task data from the qualified historical checkpoint.

Before admitting the cached V2 control at update 400, it constructs V2 afresh
and requires exact reproduction of the qualified shared step-two native
checkpoint and both actual update rows. It then constructs V3 afresh at step
zero and requires identical initial native tensors/owners/streams, with only
the architecture tag changed. The initial host outputs are identical because
all `up` weights are zero. The first G gradient intentionally changes; trained
V2 step-two tensors are never substituted for fresh V3 initialization.

V3 saves its native checkpoint after update two and requires exact CPU-mapped
recovery of updates three through five, including preservation. It continues
to the fixed 400 horizon, checking every sampled context and paired Gaussian
stream against the qualified V2 trajectory. It records DV12 stream equality,
native row decisions and guard activity. Native FAST plus averaged per-context
zero learned-feature harm stays enabled; raw output guards stay disabled.

During training, fixed clean and private-DV12 game probes are written every
100 updates, with the common qualified V2 step-two perturbation stream and
frozen V2 step-1,856 critic. These report progress without selecting a stopping
point. At the endpoint, all 570 fit/test/training-preservation/held-out-
preservation contexts use identical Gaussian panels and two shared judges:
that frozen step-1,856 critic and the qualified V2 control's step-400 critic.
The two arms' own different critics are not substituted as a shared comparison.
Full four-pool code removal and mass-only ablations assess whether the
new particle path is useful. Output RMSE is an endpoint diagnostic only.

The runner checks frozen tensor equality, complete native state/RNG immutability
after evaluation, explicit V3 export/reload parity, and all source/input hashes.
This is one task and one matched training stream. A win at 400 would qualify an
early architecture benefit over current particles; comparison with ordinary
LoRA must use the same budget and evaluation protocol before claiming that the
original formulation has been beaten.

```bash
CUDA_VISIBLE_DEVICES=1 /ml2/ntc-image-studio/.venv-anima/bin/python \
  scripts/experiment_e22_supra_particle_gated.py \
  --particlegan-root /ml2/hypergan/ParticleGAN-pr223-supra \
  --output outputs/e22-particle-gated-v3-400 \
  > outputs/e22-particle-gated-v3-400.log 2>&1
```

## Qualified 400-update outcome

The completed run and independent CPU artifact review both qualify. V3
improves every complete pool under both shared critics. The fixed D1,856
comparison is:

| Pool | V2 game | V3 game | V3 minus V2 |
| --- | ---: | ---: | ---: |
| Fit (240) | 1.349213 | 1.332187 | −0.017026 |
| Test (240) | 1.356538 | 1.334031 | −0.022508 |
| Training preservation (30) | 0.786259 | 0.768321 | −0.017938 |
| Held-out preservation (60) | 0.788366 | 0.770486 | −0.017879 |

The shared D400 judge agrees on all four directions. Test improves on 212/240
contexts under D1,856 and 202/240 under D400; held-out preservation improves
on 56/60 and 55/60 respectively. All three preservation subject means improve
under both judges. Five of six edit subjects improve; subject 5's mean game
worsens slightly (+0.001376 / +0.002761). This is an early improvement over
matched particles, not a claim of superiority to ordinary LoRA or SOTA.

The particles contribute to editing. Removing codes worsens test game by
+0.051854 / +0.110504; mass-only routing worsens it by +0.027502 / +0.053637.
Removing codes still helps held-out preservation by −0.007842 / −0.019708.
Thus the new basis improves the overall preservation comparison with V2,
while its own learned particle contribution retains an early preservation
cost. Connectivity and code dependence alone would not establish usefulness.

All 400 context/Gaussian/DV12 records match the V2 sampled program. The run
accepts zero structural moves or optimizer reopens. All 71 split-basis formulas,
128 bank rows, 142 router tensors and 284 generator tensors are observed in
clean and DV12 game gradients. Frozen owners, full native replay, all 570
paired ablation records and all 428 explicitly versioned export tensors pass
review. Training takes 208.85 seconds; the complete diagnostic job takes
397.79 seconds. These are not a matched wall-clock speed benchmark against V2.

Artifacts: `outputs/e22-particle-gated-v3-400/`, especially `receipt.json`,
`independent-review.json`, `evaluation-v2.json`, `evaluation-v3.json`,
`particle-contribution.json`, `audit-step-two.json` and `audit-final.json`.
The next fixed horizon is 1,600 updates, preserving V3's complete game state
and comparing directly with declared ordinary-LoRA reference checkpoints.

That continuation has now qualified. At 1,600 updates, V3 improves all four
complete pools over matched V2 under both common critics. It improves editing
over ordinary LoRA at the same budget, while preservation still favors
ordinary LoRA. See [the verified 1,600-update comparison](e22_supra_particle_gated_1600.md)
for full paired results, particle ablations and native controller responses.
The ordinary 6,400-update result remains the editing target. The user has
made editing the primary goal and requested editing-only training for the
next segment. The completed 1,600-update evidence retains its original
four-edit/one-preservation schedule. The new continuation will identify its
changed schedule and report the fixed 5,440-total-update checkpoint, which
matches the original's 5,120 editing updates, as well as its fixed 6,400
total-update endpoint. No longer-run outcome or SOTA claim follows from the
early improvement.

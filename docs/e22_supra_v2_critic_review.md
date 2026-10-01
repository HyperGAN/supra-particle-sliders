# V2 critic review at native step 1,856

This review uses the complete V2 checkpoint in
`outputs/e22-particle-noise-update-256/latest/`, trained with ParticleGAN
`cabe2084284db923d525918cbf3e18de6f20faac`. Findings from the earlier nonlinear
V1 run are not substituted for measurements of this model. The ordinary-LoRA
examples differ in both architecture and objective; their convergence lead
does not by itself isolate a ParticleGAN optimizer failure.

## Observed native controls

Generator, router, bank and learned-noise stationarity scales are all 1.
Generator/router learning rates remain 0.00005, the bank rate 0.0085, and the
applied critic rate is 0.00239144. Applied output sigma remains 0.125 while
the learned unconstrained sigma is 0.12478075. The generator, router and bank
still independently retain that floor, so the recently repaired noise
self-veto has no effect at this boundary.

The KA2 anchor weight is 1 and its tracking alpha is approximately
4.15e-112. Its record contains 656 EMA updates, 401 skips, and no reseeds.
This establishes almost-frozen anchor tracking, not a causal failure: its
effect on the current game needs measured gradient decomposition. The native
critic still uses real R1 and the balanced A/B penalty after 799 applied calls.

Preservation D/G payoffs are weighted by 0.1 with unchanged penalty strength,
creating ten times the relative D regularization on preservation updates.
The separate fixed preservation-units experiment tests that integration
choice while retaining the generator's original 0.1 preservation weight.

## Local critic shape

A CPU probe of this saved critic at twelve training source/time conditions
(six edit and six preservation conditions) found only 0.044% of first-tanh
preactivations beyond magnitude 3, with a mean first-tanh derivative of 0.278.
About 40.6% of second-tanh preactivations crossed that threshold. This does
not establish a disappearing game gradient: saturated feature counts need
to be checked against derivatives on the actual student residuals.

At six selected conditions, all sixteen eigenvalues of the mean score
Hessian under a fixed antithetic Gaussian quadrature were negative. Sampled
preservation curvatures were weaker than the edit curvatures. No positive
mean score-curvature failure was found in those probes.

These are shared-coordinate score derivatives, not a full RpGAN or model
parameter Hessian. At paired origin, the generator Hessian includes both
terms `H_G = 1/4 grad(S) grad(S)^T - 1/2 H_S`. Positive score curvature alone
cannot establish an outward generator improvement because the logistic
term is positive. Balanced patch perturbations cancel linear drift in
expectation, rather than on every fixed noise panel.

Patch pooling discards spatial ordering, but this alone does not prove that
the paired-residual game has a different desired optimum. Matching the
real Gaussian distribution with an independently perturbed residual still
requires zero residual distribution. Finite critic capacity and transient
gradients remain separate empirical questions.

## Native residual qualification

`scripts/diagnose_e22_supra_v2_critic.py` consumes the latest native clean/DV12
payload from `scripts/diagnose_e22_supra_current_particle_geometry.py`. It
requires matching checkpoint hash, step, source, dataset, and complete fit,
preservation and test groups. It performs no optimizer update or structural
decision. Output residual norms and alignment are diagnostic only.

The diagnostic measures native G output-space gradients, noise sensitivity,
conditioning contributions, feature saturation, and allocation of game
gradient power across fixed residual-magnitude bins. It separates the KA2
real R1, real/fake caps, and anchor parameter gradients, validating their
sum against the native penalty value and full parameter gradient. This
decomposition passed both a synthetic plumbing check and all nine actual
clean/noisy/noisy-mean native payload cases. Results are in
`outputs/e22-particle-current-geometry/critic-geometry.json`.

Independent source review confirmed the RpGAN signs, clean/noisy replication,
conditioning order, raw/standardized coordinate conversions, and KA2 units.
The review corrected the interpretation of score curvature without changing
previous frozen sources.

## Actual student gradients

For the four selected contexts in each pool, clean game output-gradient
alignment with raw paired residuals is positive: means are 0.782 for fit,
0.741 for preservation, and 0.801 for test. All twelve contexts and all
forty-eight DV12 draw/context combinations have positive alignment. This
describes output-space descent under the frozen saved critic, not the actual
Adam parameter trajectory. DV12 barely changes the measured directions.

Removing the paired output noise diagnostically increases gradient norm by
only 4.2% on fit and 6.4% on test, while reducing it by 8.9% on preservation.
This does not support a broad output-noise starvation explanation. The first
tanh is rarely saturated on these native residuals (less than 1.2%). Second
tanh saturation is 25–27% on edit pools and 49% on preservation, yet the
observed game gradients remain useful.

The total KA2 parameter-gradient norm divided by the unweighted D game norm
is 0.496 on fit, 0.133 on preservation, and 0.394 on test. Under the actual
0.1 preservation payoff weight, the preservation ratio becomes 1.33. The
total penalty opposes the game parameter gradient, with cosine from -0.887
to -0.928. Its anchor component alone has cosines -0.082/-0.735/-0.341,
respectively. This is not evidence of an anchor dominating edit learning.
The separate preservation-units experiment directly tests the integration
weighting rather than modifying native KA2 coefficients.

One measured asymmetry remains: the largest 10% of edit residual patches
carry about 48–49% of raw residual power but only 14–15% of native G gradient
power. Their gradient-to-residual gain is roughly three times lower than
the lower half of patches. Preservation does not share that allocation:
its largest 10% carry about 50% of residual power and 56% of gradient power.
These are fixed diagnostic bins, never weights in a training loss or a
selection rule. They identify a possible finite-critic response limitation,
not a proved saturation cause or an incorrect minimax objective.

## Fixed preservation-units result

Both 256-update continuations from the exact step-1,856 state completed at
step 2,112. The intervention changes preservation D and controller payoff
units from 0.1 to 1 while retaining the generator's 0.1 preservation weight.
All other native configuration, particles, routing and structural controls
remain unchanged. This tests the combined D/controller integration change;
it does not isolate their individual effects.

The independent CPU reviewer passed source-pin, full-state, frozen-parameter,
data-stream, paired-noise, DV12 clock, export and complete evaluation checks.
It also verified that the control's first 256 trace rows exactly match the
unchanged long native run. Exact four-update restore, export-forward parity
and evaluation immutability are recorded GPU runtime witnesses, rather than
GPU checks repeated by the CPU reviewer. The qualification is saved in
`outputs/e22-particle-preservation-units-256/qualification-review.json`.

Against the common frozen step-1,856 critic, held-out edit G game is
1.026344919 for native and 1.026647260 for corrected units, a 0.029% worsening.
Only 113 of 240 edit contexts and three of six edit subjects improve. Fit
game also worsens slightly. Preservation game improves from 0.743862349 to
0.739654132, with 58 of 60 contexts and all three preservation subjects
improving. This establishes a preservation tradeoff, not an edit-convergence
win that justifies changing the current recipe.

Each arm's own final critic is different. The stronger preservation critic
in the intervention raises its own final preservation G game from the
control's 0.739 to 0.830; those two numbers cannot serve as a shared comparison.
Output RMSE remains diagnostic only. No output metric enters training,
structural decisions, stopping or checkpoint selection in either arm.

## One optional causal capacity probe

If a further critic intervention is needed, test a zero-initialized trainable
linear feature term: `features = tanh(a) + gain * a`, with a sixteen-component
gain initialized to zero. Initial scores, input gradients and structural
features would match the existing critic exactly. Existing parameter values
and optimizer states can be retained; only the new gain's optimizer state
starts fresh. Feature width, native RpGAN, KA2 input-gradient units and caps,
shared particles, routing and structural controls remain the same.

This tests whether giving the critic an additional learning route around its
second saturation narrows the native game gap. It must use a fixed budget,
paired application streams, and common critics for final reporting. Neither
the diagnostic raw residual bins nor output RMSE should enter its optimizer,
guards, stopping or selection. The current evidence does not justify changing
noise or penalty strength, and it does not establish a win for this proposed
capacity change.

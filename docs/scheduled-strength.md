# Scheduled strength for Supra

Open the RAW adapter through the documented scoped Supra inference owner.
Pass that owner to `supra.scheduled_inference.render` with the same prompt,
seed, guidance and step count as the native constant-strength comparison.
Use one nonnegative gain per forward integration step. The helper returns the
decoded image and its resolved strengths and time coordinates.

```python
from supra.scheduled_inference import render

# The open owner supplies the native model, conditioning, and saved RAW.
image, receipt = render(owner, prompt=prompt, seed=7, strength=1.0,
                        steps=50, cfg=3.0, step_gains=[1.5] * 25 + [0.5] * 25)
```

The default native trajectory uses equal Euler intervals from noise to data.
The adapter remains at zero on unconditional guidance forwards.

On the saved Final Boss RAW256 adapter, the example schedule increased the
four-case mean Final Boss effect to **+0.02318**, compared with **+0.01870** at
constant strength and **+0.01229** for the reverse schedule. All eight new
256×256 renders passed the separate catastrophic-output check. The comparison
used the same frozen native checkpoint, cached conditioning, seed per case,
50 Euler intervals and CFG 3 for all arms. Early was stronger than constant
in three of four cases. It is the measured choice for this configuration.
[Shared qualification guide](https://github.com/HyperGAN/particle-sliders/blob/main/docs/strength-schedules.md)

## Activation-dependent strength

Pass a fresh shared `ActivationGate` and a calibrated `alignment` readout to
`render`. The callback receives the live, read-only solver latent
`[1, channels, height, width]` and scalar rising flow timestep. It returns one
finite signed score before the DiT runs. CFP compares that score with its
calibrated threshold. CLABS uses the measured projection statistics and
controller settings to sample its gate. A fired gate applies the selected
strength to the saved RAW adapter on the conditional branch. The decision is
made once per Euler interval and the unconditional branch runs at zero.
[Shared equations and calibration](https://github.com/HyperGAN/particle-sliders/blob/b1f780decc516808cfd3019d5e5b674457161f86/docs/strength-schedules.md#activation-dependent-steering)

```python
from particle_sliders.adaptive_steering import ActivationGate
from supra.scheduled_inference import render

image, receipt = render(
    owner, prompt=prompt, seed=image_seed, steps=steps, cfg=guidance,
    strength=selected_strength, gate=ActivationGate("cfp", calibrated_threshold),
    alignment=calibrated_alignment,
)
```

This is an allocation adaptation of *Right Token, Right Time*. Supra gates
its learned LoRA residual in place of the paper's fixed-direction edit. To
construct a projection readout, fit a direction in the chosen Supra feature
space and define how those spatial activations reduce to one vector per
interval. Use the shared `activation_projection` to account for signed
strength. Record the feature definition, fit data and probe identity. For
CLABS, pass `ActivationGate("clabs", calibrated_bias, seed=gate_seed,
clabs=parameters)`. The matched random control uses
`ActivationGate("random", target_rate, seed=gate_seed)`.

Use disjoint data for fitting the readout, calibrating the gate and measuring
finished outputs. Run the shared `calibrate_gate_budget` on active calibration
rollouts with the same RAW, sampler and case seeds. The receipt records each
interval's alignment, probability, decision and effective strength, plus the
realized gate rate and `interval_weighted_dose`, the sum of
`abs(delta * effective_strength)`. For exact strength replay, use
`step_gains=[value / selected_strength for value in receipt["effective_strengths"]]`
with the same request. Compare held-out images against the matched OFF,
matched-budget constant and random controls. Rank by mean per-case Final Boss
effect ON minus OFF and apply the separate catastrophic-output check.

The retained Final Boss RAW256 panel supplies fixed-schedule reference scores.
Adaptive gains are unmeasured. The new path requires a fitted Supra readout
and measured active-rollout calibration. Reopen source-bound RAW and FULL
packets in their original environments. Earlier source cohorts retain their
captured loaders and evidence.

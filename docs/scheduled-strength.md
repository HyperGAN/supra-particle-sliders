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

Reopen source-bound RAW and FULL packets in their original environments.

### Final Boss RAW256 activation pilot

The pilot uses the original 256×256 Supra2-IMG sampler with 50 rising Euler
intervals, CFG 3 and the saved Final Boss RAW256 adapter at nominal strength 1.
The readout averages each latent channel and within-patch position over Supra’s
native 2×2 patch grid before the denoiser call. The signed top-ten probe was
fitted on four archived positive/neutral trajectory pairs from two prompt
families. CFP’s threshold and CLABS’s bias were calibrated on active rollouts of
four separate H3 Final Boss prompts encoded by Supra’s native T5. Both gates
fired on exactly 100 of 200 calibration intervals.

Each of the four held-out cases compares a gate with constant strength at the
gate’s own realized interval-weighted dose. Gains are the equal-case mean
finished-image Final Boss score over the exact matched OFF image.

| Method | Held-out firing | Gate gain | Dose-matched constant gain | Cases ahead |
| --- | ---: | ---: | ---: | ---: |
| CFP | 111/200 | +0.008638 | +0.005886 | 2/4 |
| CLABS | 93/200 | +0.006925 | +0.004532 | 1/4 |
| Random | 93/200 | +0.006500 | +0.005600 | 2/4 |

All four OFF PNGs replayed byte for byte. Those four and all 20 distinct new
images passed the independent categorical check. The bridge case stayed OFF
under both learned gates. The measured fixed early profile above remains the
documented choice for this configuration. The pilot’s fitted direction and
calibration apply to this adapter and sampler. [Study and per-case evidence](final-boss-raw256-adaptive-evidence.json)

### Reconstruct the measured Final Boss CFP setting

The measured RAW256 uses its archived training owner's packet format. Use
that runtime and its original conditioning for this study. The newer
`supra_scoped` owner requires its own RAW packet schema.

Read the calibrated strength and probe from the study. `raw_path` is the
checkpoint used by the archived owner. Pass the resulting `gate` and
`alignment` to the adaptive render with the study's Euler steps and CFG.

```python
from hashlib import file_digest
import json
from pathlib import Path

import torch
from particle_sliders.adaptive_steering import ActivationGate, activation_projection

study = json.loads(Path("docs/final-boss-raw256-adaptive-evidence.json").read_bytes())
profile = study["profile"]
with Path(raw_path).open("rb") as checkpoint:
    if file_digest(checkpoint, "sha256").hexdigest() != profile["RAW256_sha256"]:
        raise ValueError("Use the measured Final Boss RAW256 checkpoint")

selected_strength = profile["nominal_strength"]
direction = torch.tensor(study["probe"]["direction"], dtype=torch.float32)
patch = study["probe"]["native_patch"]

def alignment(latent, timestep):
    batch, channels, height, width = latent.shape
    if batch != 1 or height % patch or width % patch:
        raise ValueError("Expected the study's single-image native patch geometry")
    by_phase = latent.reshape(batch, channels, height // patch, patch, width // patch, patch)
    pooled = by_phase.mean(dim=(0, 2, 4), dtype=torch.float32).reshape(-1)
    return activation_projection(pooled, direction, strength=selected_strength)

gate = ActivationGate("cfp", study["calibration"]["parameters"]["cfp"]["parameter"])
```

Use a fresh gate for each request. The render receipt records the decision
and strength at every native Euler interval.

### Reconstruct the measured Final Boss CLABS setting

This replay uses the same archived RAW256 owner, projection, and native
sampler as the CFP recipe. CLABS averaged +0.002392 over its dose-matched
constant across the four held-out cases and led on one case. The calibrated
bias fired on 100 of 200 intervals on the separate calibration panel.

For an archived case, set `case_id` to its study identifier and use its
original cached conditioning and image seed. The original study also used
that case's seed to initialize an independent CLABS gate for each request.
Keep the `study`, `alignment` and `selected_strength` from the CFP recipe.

```python
from particle_sliders.adaptive_steering import ActivationGate, CLABSParameters

case, = (row for row in study["comparisons"]["clabs"]["cases"] if row["case"] == case_id)
image_seed = case["seed"]
calibration = study["calibration"]
gate = ActivationGate(
    "clabs", calibration["parameters"]["clabs"]["parameter"], seed=image_seed,
    clabs=CLABSParameters(**calibration["measured_normalization"]),
)
```

Use the gate and projection with that case's frozen request. Verify the
receipt's `effective_strengths` and `interval_weighted_dose` against
`resolved_gains` and `interval_weighted_dose` in the case record. The
case's `matched_constant_strength` gives the fixed control for that same
seed and conditioning.

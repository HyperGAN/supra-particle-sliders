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

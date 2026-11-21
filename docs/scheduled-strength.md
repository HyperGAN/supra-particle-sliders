# Scheduled strength for Supra

Open the RAW adapter through the documented scoped Supra inference owner.
Pass that owner to `supra.scheduled_inference.render` with the same prompt,
seed, guidance and step count as the native constant-strength comparison.
Use one nonnegative gain per forward integration step. The helper returns the
decoded image and its resolved strengths and time coordinates.

```python
from supra.scheduled_inference import render

# owner is the open Supra scoped RAW inference owner.
image, receipt = render(owner, prompt=prompt, seed=7, strength=1.0,
                        steps=4, step_gains=[1.5, 1.5, 0.5, 0.5])
```

The default native trajectory uses equal Euler intervals from noise to data.
Compare gains of 1 throughout with the early profile shown above and its
reverse. The adapter remains at zero on unconditional guidance forwards.
Select a native profile using matched finished-image effect and the separate
catastrophic-output check. [Shared qualification guide](https://github.com/HyperGAN/particle-sliders/blob/main/docs/strength-schedules.md)

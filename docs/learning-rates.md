# Measured Supra learning rates

For the tested Final boss FP32 128-update recipe, use generator rate `0.036`
and critic rate `0.018`: `lr=0.036`, `d_lr_mult=0.5`. Supra is adapter-only,
so there is no particle-bank rate and `prior_lr_mult` must be omitted.

The reusable [options overlay](../configs/supra/final-boss-fp32-128.options.json)
contains the measured rates and training settings. Merge it into a valid native
session's `options`; it is not a complete `--config` file. Supply that session's
reviewed native configuration, local cache and contexts, exact asset digests,
fresh-cache receipt and device as described in [PG0.9.4 setup](pg094.md).
This overlay selects a fresh 128-update run; it does not retarget an existing
checkpoint's serialized configuration.

```python
import json
from pathlib import Path
from particle_sliders.lifecycle import SessionConfig

overlay = json.loads(
    Path("configs/supra/final-boss-fp32-128.options.json").read_text()
)
config = SessionConfig(
    backend="supra",
    cohort="supra-final-boss-fp32-128",
    options={**verified_native_options, **overlay},
)
```

`verified_native_options` must describe the matching model/data/source cohort.
The measured fresh cache has SHA256
`e09d0c00a8fa20acc2fee74940a81f61d35b130b9fc3475479bc87c5d9e76bef`;
its contexts have SHA256
`de73df48081b3dca59663e630d197b3d4eb21865da4681e30b8c1cdfaed4fee3`.
Changing those inputs requires a new measured cohort. Product defaults remain
unchanged; this is a recommendation for the recorded 128-update recipe rather
than an optimum established at every training budget or concept.

## Selection and confirmation

The sole selection metric is raw teacher/student relative squared error. For
each example, subtract tensors in FP64, pool squared errors and teacher-edit
energy across its positions and coordinates, divide within the example, then
average examples equally. Lower is better. OFF, teacher and reversed-edit
controls score 1, 0 and 4. Zero-energy examples have undefined relative error
and are excluded rather than given an epsilon denominator.

The selected generator/critic rates scored `0.48553781061681967` on development
and `0.7827618615930451` on six held-out edit examples, versus OFF at `1.0`.
The other 128-update finalist used generator `0.036` and actual critic
`0.05399999999999999`, scoring `0.673095327830374` and
`0.7904014816459853` respectively. The same candidate wins development and
confirmation. Both candidates completed the same 128-update budget and horizon
with matching initial components, teacher fields and conditioning. Full-state
continuation, actual optimizer rates, frozen base and raw export replay were
verified.

Preservation drift, late/weak-field scores and media quality are optional
diagnostics. They do not select or reject a rate for this optimization task.
Earlier rejection on those grounds is superseded by this one-metric rule.

## Measured scope and evidence

The run used native response tensors `[1, 4, 32, 32]`, FP32 base precision,
conditional-only adaptation, batch 4, rank 16 and response scale
`0.32837046474230014`. It used official ParticleGAN 0.9.4, shared core
[`ddb4142`](https://github.com/HyperGAN/particle-sliders/commit/ddb41423bcefc1fe76cd94d0f9b367353b3ea137)
and product source
[`ec7ae60`](https://github.com/HyperGAN/supra-particle-sliders/commit/ec7ae600ada98eed4391b6244a550d007063c4b4).

The frozen base SHA256 is
`78ff6c20e47899d12e69d7f865aa2222d6445947e23c8001cc83b4e5ef17c2a4`.
The selected full training checkpoint SHA256 is
`bda11ced3df6d6e24a0d12911b9bb59d3abf4e8d9befce441db2cccf3824ca11`.
The saved-tensor FP64 reanalysis report SHA256 is
`b0a5223e859a572f3ef13b461188eb44eb6c91167e52c3e95d400561dfd0e8aa`.
Its full local evidence is retained at
`/mnt/nvme/sliders/experiments/lr-standardization-20261104/supra/primary-only-assessment-v1/`;
the two original training/confirmation runs remain under
`supra/confirmation128-traced-retry-v2/` in the same experiment directory.
Independent review checked the reanalysis, grouping, controls and actual rates.

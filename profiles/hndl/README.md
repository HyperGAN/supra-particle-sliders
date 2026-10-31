# Supra HyperGAN/HNDL prerelease candidate

This separately built `0.3.0.dev1` prerelease registers `supra_hndl`, using the
real shared ParticleGAN game, HyperGAN develop builder, HNDL LoRA branches and
shared HNDL spatial critic. The native DiT, CFG, sampler and ordinary linear
LoRA law are preserved. These branches are not routed particles.

## Installation and isolation

The supported root wheel remains `supra-particle-sliders==0.2.0`, with the exact
core0.8.1 dependency. Its wheel excludes all five candidate modules/assets.
The candidate retains the product distribution and `supra` namespace so its
native model source identity stays explicit. Its development version prevents
normal pip resolution from selecting it over the stable release; installing
this candidate wheel is an explicit opt-in. **Use a new virtual environment.**
The two profiles replace the same namespace and must never be coinstalled or
installed over a historical checkpoint environment. Never use `--pre` in a
stable environment or force the two core dependency pins to coexist.

Build from the source checkout with:

```bash
python scripts/build_hndl_profile.py --out artifacts/hndl
python -m venv .venv-supra-hndl
# Install the appropriate CPU or CUDA PyTorch build in this new environment.
.venv-supra-hndl/bin/python -m pip install --index-url https://pypi.org/simple artifacts/hndl/supra_particle_sliders-0.3.0.dev1-py3-none-any.whl
```

This command uses PyPI for declared dependencies. `--find-links` only adds wheel
candidates; it does not disable index access. This is not an offline install
recipe. A fully offline installation needs every dependency as a verified local
artifact, including the Git requirements; `--no-index` alone does not disable
Git network access.

The exact core commit `4abc0eb03a4e3a84fc3e6d02734c13dc7e8ea231` is publicly
reachable on `qualify/core012-cutpoints-20261031`. The candidate metadata pins
that Git commit with its `hypergan-hndl` extra, which pins HyperGAN develop
`5f368a3770cf5bdae0448995c9f76e29af8049f7` and HNDL0.10.0. Clean CPU installation
from these exact public sources is tested; there is no remaining source
publication blocker. This does not claim a core0.12 PyPI release.

`supra/core012-receipt.json` validates the actually imported 50 shared
Python/JSON/HNDL files, independently of `PARTICLE_SLIDERS_ROOT`. Core/PG/HNDL
versions and HyperGAN build provenance are enforced. No wheel hash is treated
as a runtime identity: independently built wheels can have different archive
bytes while containing the same admitted source files.

The native factory accepts local campaign/config/cache/context/base checksums,
backend `supra_hndl` and a fresh explicit cohort name. It uses shared training,
response, render, evaluation and full-state resume APIs. Raw/EMA native-only v2
exports use a distinct inference factory without training contexts/cache/game
construction. The fresh CPU inference worker is explicitly a test profile;
separate tests exercise the shipped CUDA/rank/precision/base validation.

Existing CDF source environments and locks retain their roles. Cleanup and
source-validation fixes change the stable lifecycle's source identity; a packet
bound to the pre-fix source must use its original checkout/environment, such as
PR6 head `5f6d2a521038388a930ff509f70710c98b908b3d`. No receipt or checkpoint is
rebound to the fixed source. The HNDL candidate always requires its new cohort.

Native CUDA/T5/VAE loading, pretrained cache parity, actual 256px decoding, GPU
fit, convergence and image quality remain unrun. Tiny native CPU fixtures and
synthetic caches do not qualify those operations. No production device gate is
relaxed and no new GAN loss is introduced.

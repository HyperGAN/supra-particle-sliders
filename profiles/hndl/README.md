Current upgrade: [PG0.9.4 canonical instructions](../../docs/pg094.md). The separate
HNDL candidate is 0.4.0.dev1; original dev1/dev2 previews remain historical.

# Supra HyperGAN/HNDL prerelease candidate

This separately built `0.4.0.dev1` prerelease registers `supra_hndl`, using the
real shared ParticleGAN game, HyperGAN develop builder, HNDL LoRA branches and
shared HNDL spatial critic. The native DiT, CFG, sampler and ordinary linear
LoRA law are preserved. These branches are not routed particles.

## Installation and isolation

The supported root wheel remains `supra-particle-sliders==0.3.0`, with the exact
current core 0.12.0 / ParticleGAN 0.9.4 dependency. Its wheel excludes candidate modules and data.
The candidate retains the product distribution and `supra` namespace so its
native model source identity stays explicit. Its development version prevents
normal pip resolution from selecting it over the stable release; installing
this candidate wheel is an explicit opt-in. **Use a new virtual environment.**
The two profiles replace the same namespace and must never be coinstalled or
installed over a historical checkpoint environment. Never use `--pre` in a
stable environment or force the two core dependency pins to coexist.

The resolver test verifies that both wheel files exist before testing selection.
If a wheelhouse contains only the candidate, pip can select that prerelease
without `--pre`; the development version is not an installation guard. Choosing
such a candidate-only source is an explicit opt-in and still requires a new
environment. No candidate is published to PyPI by this work.

Build from the source checkout with:

```bash
python -m pip install build==1.6.1 setuptools==84.2.1
python scripts/build_hndl_profile.py --out artifacts/hndl --sdist
python -m venv .venv-supra-hndl
# Install the appropriate CPU or CUDA PyTorch build in this new environment.
.venv-supra-hndl/bin/python -m pip install --index-url https://pypi.org/simple artifacts/hndl/supra_particle_sliders-0.4.0.dev1-py3-none-any.whl
```

This command uses PyPI for declared dependencies. `--find-links` only adds wheel
candidates; it does not disable index access. This is not an offline install
recipe. A fully offline installation needs every dependency as a verified local
artifact, including the Git requirements; `--no-index` alone does not disable
Git network access.

The exact core commit `ddb41423bcefc1fe76cd94d0f9b367353b3ea137` is publicly
reachable on the approved immutable source ref. The candidate metadata pins
that Git commit with its `hypergan-hndl` extra, which pins HyperGAN develop
`5f368a3770cf5bdae0448995c9f76e29af8049f7` and HNDL 0.10.0. Normal dependency installation from these exact public sources is tested
in a separate system-site environment; the canonical wheel also has a clean
CPU installation. There is no remaining source publication blocker. This does not claim a core 0.12 PyPI release.

`supra/core012-ddb4142-hndl-receipt.json` validates the 574 recursive source/data files
at the resolved imported core paths, independently of `PARTICLE_SLIDERS_ROOT`. Core, PG and HNDL
versions and HyperGAN build provenance are enforced. No wheel hash is treated
as a runtime identity: independently built wheels can have different archive
bytes while containing the same admitted source files.
These are source-file checks at resolved import paths, not an attestation of
already executed bytecode or protection against runtime monkeypatching or
concurrent file replacement. The negative tests cover changed on-disk source
under an imported root while a separate good checkout override is supplied.

The native factory accepts local campaign/config/cache/context/base checksums,
backend `supra_hndl` and a fresh explicit cohort name. It uses shared training,
response, render, evaluation and full-state resume APIs. Raw/EMA native-only v2
exports use a distinct inference factory without training contexts/cache/game
construction. The fresh CPU inference worker is explicitly a test profile;
separate tests exercise the shipped CUDA/rank/precision/base validation.
`ema_decay` belongs in `SessionConfig` options: shared `SliderSession` owns EMA
updates and checkpoint/export state. It is not forwarded to `ConditionalPG09`
as a game recipe option or an additional loss.

Existing CDF source environments and locks retain their roles. Cleanup and
source-validation fixes change the stable lifecycle's source identity; a packet
bound to the pre-fix source must use its original checkout/environment, such as
PR 6 head `5f6d2a521038388a930ff509f70710c98b908b3d`. No receipt or checkpoint is
rebound to the fixed source. The HNDL candidate always requires its new cohort.

Native CUDA/T5/VAE loading, pretrained cache parity, actual 256px decoding, GPU
fit, convergence and image quality remain unrun. Tiny native CPU fixtures and
synthetic caches do not qualify those operations. No production device gate is
relaxed and no new GAN loss is introduced.

## Coordinated final core pin

The active development release is now `0.4.0.dev1`, bound to core commit
`ddb41423bcefc1fe76cd94d0f9b367353b3ea137` and network/source profile
`supra-hypergan-hndl-ddb4142-v1`. The prior `0.3.0.dev1` preview and its
`core012-receipt.json` remain historical cohorts. The current source identity requires a fresh cohort; no conversion or
relabeling is provided. Genuine archived preview packets were unavailable for
this upgrade's refusal test. The canonical and HNDL factories both use the current core but retain distinct source identities. Historical 0.2.0 and preview environments remain separate.

The new closure is generated from every tracked file under the exact core
commit's package namespaces, including nested data and LICENSE. Validation
checks both the expected bytes and the complete non-cache file set at the
resolved imported package paths. This remains a source-file check, not an
attestation of previously executed bytecode or a race-free runtime guarantee.

For the prior dev2, wheel and sdist installations were separately tested in
fresh CPU environments with the product checkout temporarily absent. See the
[historical dev2 migration receipt](../../docs/hndl-a794-migration.md) for prior tests,
artifact identity and provenance. Current upgrade validation is in
[the PG0.9.4 receipt](../../docs/pg094-validation.md); native qualification limits remain.

The coordinated core owner reported a bounded synthetic CUDA software fixture.
That does not qualify Supra model loading, native cache/width behavior, image
quality or a new GPU training run. This product migration performs CPU tests
only and retains all existing production gates.

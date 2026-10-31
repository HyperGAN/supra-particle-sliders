# Supra inventory repair and HNDL candidate CPU validation

The default installer repair reuses commit `517dab3a11e8ab402631002c06eebbdbf890f133`.
It retains the packaged core0.8.1/PG0.9.1 cohort and the separate historical
CDF core0.6.0 environment. Their source locks, native model, checkpoint loaders
and runtime code are unchanged. See [installation cohorts](../INSTALL-COHORTS.md).

The distinct [0.3.0 HNDL profile](../profiles/hndl/README.md) executes the actual
shared core0.12 game, HyperGAN develop `5f368a3770cf5bdae0448995c9f76e29af8049f7`
and HNDL0.10.0. It registers training and native-only inference as `supra_hndl`.
The new LoRA graph preserves the native linear down/up computation. The spatial
critic is the shared `HNDLFieldCritic`; the native DiT/CFG/sampler and GAN law
retain their existing owners. Native responses remain `[B,4,32,32]`; no shape,
response-domain or width admission is changed.

## Exact shared candidate

The local shared commit is `4abc0eb03a4e3a84fc3e6d02734c13dc7e8ea231`.
Its wheel SHA256 is
`9afad3afc1f6c653a574074dc14f0da1f1c43975c31ca45c4eac049df0b0b4ad`.
The new `supra/core012-receipt.json` checks all 50 shared Python/JSON/HNDL
files before native preparation. The shared builder additionally verifies the
released HNDL source closure, HyperGAN builder closure and clean build provenance.
This candidate is local and unpublished. The profile requires its exact local
wheel; no publicly resolving core0.12 dependency is advertised.

## Checks

- Supported core0.8.1 installed wheel: **21 passed**, including original native
  source/AST equivalence, signed native forward/gradient parity, split resume,
  atomic rejection, native-only exports and the new installer/ownership guards.
- Historical CDF core0.6.0 source CLI: **29 passed**, including optimizer-history
  corruption and exact native replay. This is deliberately a source checkout
  test: the old product envelope hashes `scripts/train_pg09.py`, which is absent
  from the root wheel. An initial installed-only attempt failed at that missing
  source file; it is not counted as an installed-package pass.
- HNDL candidate installed wheel: **18 passed** on Torch2.14.1+cpu, including
  actual native HNDL forward/input/parameter gradient parity, spatial critic
  response/parameter second derivatives, real shared-game updates, exact EMA
  resume, strength0 stability, source/model/config/builder tamper rejection,
  observation/failure restoration and raw/EMA inference in fresh processes with
  training assets removed and training/optimizer constructors forbidden.
- The same **18 checks passed** with the declared minimum Torch2.6.0+cpu.
- An actual core0.8.1 Supra checkpoint is rejected by the HNDL session with live
  game state unchanged. Checkpoint/cohort labels and historical receipts are not
  rewritten. Old environments continue to use their original factories.

Installed tests run from `/tmp` with `python -I -m pytest --import-mode=importlib
-p no:cacheprovider -o pythonpath=`, empty PYTHONPATH, plugin autoload disabled,
offline model variables, no visible CUDA and one operator thread. Select
`tests/core081 tests/test_install_cohorts.py` for the stable environment (supply
the preserved fd89e1c source using `SUPRA_REFERENCE_CORE`); select
`tests/hypergan_profile` for the candidate environment. The legacy source tests
are `tests/test_pg09_integration.py tests/test_pg09_restore_history.py`, with the
repository explicitly on pytest's Python path in the legacy environment.

Each environment is task-local. Existing runtime libraries are borrowed read-only;
these checks do not establish clean resolution of all optional pretrained
dependencies. All fixtures are tiny native CPU models with synthetic cache
scales. No production CPU/CUDA device gate is relaxed.

Native CUDA/T5/VAE loading, pretrained cache parity, actual 256px decoding, GPU
fit, convergence and image quality remain unrun. There is no new model download,
training run, model publication or deployment. The repository has no GitHub
workflow or automatic release configured in its tracked tree; this change adds
none. A reviewed shared-core publication and separate native qualification are
required before promoting the default installer to the HNDL cohort.

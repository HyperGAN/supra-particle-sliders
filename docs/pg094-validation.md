# PG0.9.4 upgrade validation — 2026-11-04

This is software conformance for root `supra-particle-sliders==0.3.0` and the
separate `0.4.0.dev1` HNDL candidate, on exact shared core commit
`ddb41423bcefc1fe76cd94d0f9b367353b3ea137` / official ParticleGAN 0.9.4.
The former preview remains historical; this receipt does not inherit its
native or installed-artifact claims.

All runs use Python 3.11, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
`OPENBLAS_NUM_THREADS=1`, `CUDA_VISIBLE_DEVICES=''`, `HF_HUB_OFFLINE=1`
and `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`. No pretrained download or GPU job ran.

- Canonical CPU source suite: **84 passed**. Covers actual adapter updates,
  generator/critic ownership, exact split resume, metrics/EMA/RNG, zero-strength
  preservation, atomic corrupt-restore refusal, observation and failed-prepare
  cleanup, native LoRA/CFG/schedule/distillation, installer and train aliases.
- Canonical wheel installed in its own environment, tests/configuration copied
  outside the product checkout, Python `-I` and pytest `--import-mode=importlib`:
  **26 passed**. The selected tests are `tests/core081/test_lifecycle_closure.py`,
  `test_native_cleanup.py`, and `test_current_dispatcher.py`; the directory name
  is historical but their factory now uses the current PG0.9.4 receipt.
- Wheel contents, dependency metadata, entry points and pip resolution with both
  canonical and candidate artifacts: **4 passed**. Current wheels contain the
  exact expected namespace files; canonical excludes optional HNDL modules.
- Separate installed HNDL candidate with HyperGAN develop
  `5f368a3770cf5bdae0448995c9f76e29af8049f7` and HNDL 0.10.0: **65 passed**.
  Covers real shared game/HNDL graphs, exact native forward/gradients, split
  resume, raw/EMA native-only inference in fresh processes, source/data tamper
  refusal, native admission and cleanup. **Three archived-preview tests were
  deselected because genuine prior-preview packets were unavailable**; no
  historical packet-refusal execution is claimed for this upgrade.

The shared dispatcher is exercised with actual installed profile discovery,
plan, isolated `--check` and changed-input refusal before output allocation.
Its checks are allocation-free source/configuration/checksum admission. Native
training is tested through explicitly staged tiny CPU model fixtures; this does
not run production training through the dispatcher or qualify native assets.

The first installed environments deliberately inherited system-site packages,
with Torch 2.14.1+cu130 and CUDA hidden. Their `pip check` reports unrelated
pre-existing application dependency conflicts; those are not represented as a
clean environment. A separate clean CPU environment validation is recorded
below after normal dependency resolution.

Native CUDA loading, T5/VAE/cache parity, actual 256px output, GPU fit,
convergence and image quality remain unqualified. Original historical
locks/receipts, FinalBoss MSE recipes and published weights are unchanged.

Clean canonical wheel validation: **26 passed** in an independent virtual
environment without system-site packages, with Torch **2.15.0+cpu**, official
PG0.9.4, Pillow 13.0.0 and NumPy 2.4.6, after normal dependency resolution.
`pip check` reports **No broken requirements found**. The installed import
origin is the clean environment's `site-packages/supra`, and core
`direct_url.json` records exact VCS commit `ddb41423bcefc1fe76cd94d0f9b367353b3ea137`.
The same external copied tests and isolated import mode were used. No further
native qualification is implied.

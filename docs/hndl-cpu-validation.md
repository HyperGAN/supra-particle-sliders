# Supra inventory repair and reviewed HNDL candidate CPU validation

The default installer repair reuses commit `517dab3a11e8ab402631002c06eebbdbf890f133`.
The supported root0.2.0 wheel retains the exact core0.8.1/PG0.9.1 dependency;
`requirements-legacy.txt` retains the CDF core0.6.0 environment byte-for-byte.
Source locks and checkpoint labels are not retargeted. See
[installation cohorts](../INSTALL-COHORTS.md).

The separate [0.3.0.dev1 prerelease](../profiles/hndl/README.md) executes the real
shared core0.12 game, HyperGAN develop `5f368a3770cf5bdae0448995c9f76e29af8049f7`
and HNDL0.10.0. Native responses remain `[B,4,32,32]`; no response-domain or width
admission is changed. Native-only inference remains separate from training.

## Corrections after source review

The supported wheel now contains exactly21 product files, excluding the three
HNDL modules, graph and core012 receipt. The candidate contains those additional
five files and uses a prerelease version. A real pip dry-run with both wheels
available selects stable0.2.0 without `--pre`. The two profiles share a namespace
and require separate environments; an explicit candidate install is never a
supported upgrade of an existing historical environment.

Both source gates hash the actually imported core, ignoring checkout overrides.
A subprocess test imports a deliberately altered core while pointing
`PARTICLE_SLIDERS_ROOT` at a good copy and confirms rejection. The candidate
receipt enforces core/PG/HNDL versions and exact HyperGAN build provenance.
Its previously unused source-commit field was removed; archive hashes are build
receipts, not claimed runtime identity checks.

Native cleanup attempts every callback and native attribute restoration even
after an error, reports cleanup failures, and preserves an already active primary
exception. Partial LoRA attachment rolls back the native module tree and the
original list object even when a setter mutates before raising. Replacement
wrappers preserve parameter bytes, dtype/device, gradient flags, module modes,
base ownership, scale and multiplier.

Missing/malformed context receipts have an explicit ValueError contract, and
EMA decay is checked as a finite numeric value in `[0,1)`. Independent tests
exercise the shipped inference validator's CUDA, rank, precision, base and
training-input gates. The fresh-process CPU worker remains explicitly a test
profile; it does not qualify production CUDA execution.

These source fixes change the stable lifecycle source identity. Checkpoints
bound to pre-fix source must retain their original source environment (for
example PR6 head `5f6d2a521038388a930ff509f70710c98b908b3d`). No converter,
checkpoint relabeling or new source hashes are injected into old receipts.
Historical CDF loaders and source locks retain their original roles.

## Public dependency and clean installation evidence

The exact core commit `4abc0eb03a4e3a84fc3e6d02734c13dc7e8ea231` is publicly
reachable on branch `qualify/core012-cutpoints-20261031`. It was independently
fetched and checked. The prerelease metadata now pins this immutable Git source
with its real `hypergan-hndl` extra. No core0.12 PyPI release is assumed.

Two new virtual environments contain no borrowed third-party runtime paths.
Normal pip resolution installed each product wheel and its exact public Git
core, including the candidate's exact HyperGAN Git build. No `--no-deps`,
resolver override or copied installed packages were used in these environments.
Both use the existing official Torch2.6.0+cpu wheel, actual imported version
checked, and both pass `pip check`. NumPy is now an explicit base dependency
because the native digest code uses tensor-to-NumPy conversion.

## Final checks

- Clean supported installed wheel on Torch2.6.0+cpu: **30 passed**: the original
  equivalence/lifecycle/installer checks, cleanup/context/import guards and
  three built-wheel/resolver checks.
- Clean prerelease installed wheel on Torch2.6.0+cpu: **59 passed**: the18
  original HNDL checks,38 review regression checks and three wheel/resolver
  checks. Coverage includes native graph/critic derivatives, actual shared-game
  updates, exact EMA/split resume, zero strength, atomic source/model/config/
  builder/checkpoint rejection and fresh-process raw/EMA inference after training
  assets are removed.
- Historical CDF source path is tested separately from installed lifecycle
  behavior. Its public CLI hashes its source companion `scripts/train_pg09.py`;
  that historical CLI is not an installed-wheel API.

Correction to the earlier minimum-version receipt: the old environment labeled
Torch2.6 actually imported Torch2.14 through a borrowed path. Its18 passes are
not Torch2.6 evidence. The fresh resolved environments above replace that claim.
The original stable21, candidate18 and CDF29 observations remain historical
receipts, without being relabeled as clean installs or GPU qualification.

Installed tests run from `/tmp` with `python -I -m pytest --import-mode=importlib
-p no:cacheprovider -o pythonpath=`, empty PYTHONPATH, plugin autoload disabled,
offline model settings, no visible CUDA and one operator thread. Stable selectors
are `tests/core081 tests/test_install_cohorts.py tests/test_wheel_contents.py`;
supply exact fd89e1c source via `SUPRA_REFERENCE_CORE`. Candidate selectors are
`tests/hypergan_profile tests/test_wheel_contents.py`. The packaging suite receives
both exact wheel paths as `SUPRA_STABLE_WHEEL` and `SUPRA_CANDIDATE_WHEEL`.

Native CUDA/T5/VAE loading, pretrained cache parity, actual256px decoding, GPU
fit, convergence and image quality remain unrun. There is no model download,
GPU training, weight publication, deployment, package release or merge. There
are no tracked GitHub workflows or merge-triggered release/deployment hooks.
The dependency-publication blocker is resolved; native qualification and the
root-owned review/merge route remain outstanding.

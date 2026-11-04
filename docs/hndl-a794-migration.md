# Coordinated final modern core migration

The active Supra candidate is `supra-particle-sliders==0.3.0.dev2` and pins
`particle-sliders-core[hypergan-hndl]` to the exact public Git commit
`a794e3b4a8813b012f2c4faa3da82b78d4a90214`. Core still reports version `0.12.0`,
with official PG `0.9.4`, HyperGAN develop
`5f368a3770cf5bdae0448995c9f76e29af8049f7` and HNDL `0.10.0`.

This is a source identity migration, not a new loss, architecture, native model
admission or HNDL optimization. The ordinary linear LoRA graph, native targets,
configuration, flow geometry and production device gates are unchanged. The
supported root `0.2.0` and historical CDF environments keep their original pins,
locks and metadata. No umbrella trainer or shared-core file is edited.

## Closure and artifact identity

`supra/core012-a794e3-receipt.json` records all 51 tracked source/data files
recursively under the target core's two package namespaces, including LICENSE.
The active loader checks the complete non-cache installed file set and each
SHA256 at the resolved imported package paths. It also checks the closure schema,
target identity, package versions, official PG sources and strict HyperGAN/HNDL
builder provenance. These are file/origin checks, not an attestation of already
executed bytecode, runtime monkeypatching or concurrent replacement.

The network/source profile is `supra-hypergan-hndl-a794e3-v1`. The prior
`core012-receipt.json` is preserved byte-for-byte as a dev1 preview record. Genuine
dev1 training, raw and EMA packets were captured before this migration and are
refused by the new cohort. No packet or old receipt is rewritten. Consumers of
an old packet must retain its exact old source environment.

## Reproducible source installation

Use a new virtual environment for each product profile. Do not install this
candidate over a stable or historical environment. The prerelease suffix is not
an isolation guard: a candidate-only wheelhouse can select it without `--pre`.
The stable wheel remains selected when both stable and candidate are available.

Build the product source into independent artifacts:

```bash
python -m pip install build==1.6.1 setuptools==84.2.1
python scripts/build_hndl_profile.py --out artifacts/hndl-a794 --sdist
```

After installing the appropriate CPU or CUDA PyTorch build in the new
environment, install either the wheel or the sdist normally. Both declare the
exact core Git requirement and resolve the same HyperGAN/HNDL dependencies:

```bash
python -m pip install --index-url https://pypi.org/simple 'artifacts/hndl-a794/supra_particle_sliders-0.3.0.dev2-py3-none-any.whl[test]'
# Alternative, in another new environment:
python -m pip install --index-url https://pypi.org/simple 'artifacts/hndl-a794/supra_particle_sliders-0.3.0.dev2.tar.gz[test]'
python -m pip check
```

`--find-links` adds candidates but does not disable PyPI. Even `--no-index` does
not disable Git network access. Fully offline installation requires every
dependency and direct Git requirement to be supplied as a verified local
artifact; no incomplete offline recipe or dependency override is implied here.

## Product qualification

Tests use installed packages in isolated CPU environments, no checkout on the
Python import path, no visible CUDA, offline model settings, disabled pytest
plugin autoload and one operator thread. Tests and tiny configuration fixtures
are copied byte-for-byte outside the checkout. The product checkout is also
temporarily moved away during the final runs and restored afterward.

The candidate suite covers actual shared-game updates and exact EMA/split resume;
OFF, raw and EMA native-only inference; source and recursive-data tampering;
old-preview refusal; native flags/RNG/observation cleanup; admission rejection
with a successful retry; and an injected failed step that closes the session,
followed by a fresh session restoring the last valid checkpoint exactly. A
failed session is not claimed to support in-place reuse.

Final results on actual Torch `2.6.0+cpu`:

| Installed artifact | Result |
|---|---:|
| Standalone wheel | 72 passed, zero skipped |
| Standalone sdist | 72 passed, zero skipped |

Both normal installations and test runs completed while the canonical product
checkout path was absent. The checkout was restored afterward. Both isolated
environments pass `pip check`. The source/data tamper cases include a changed
LICENSE and an unexpected nested JSON file. Three genuine dev1 preview packets
(training, raw and EMA) are rejected; they remain archived as old cohorts.

The active candidate contains 27 product package files. A newly built stable
wheel still contains the same 21 package files, byte-for-byte, as before this
migration. Stable and historical dependency metadata, source locks, model code,
licenses, objectives and native gates are unchanged.

Artifact SHA256 values for these builds:

```text
wheel  872fba9b3eb657e15279072b528324af883e26f38a80fa5fdf4b95f5d30ca6ac
sdist  20358545c7c7476c54ce90638f9204bd04310c3db6f8819e6aa6054bec0c7edf
```

Archive hashes identify these build outputs. Rebuilt archives can have different
container bytes; the admitted recursive source/data closure is checked
independently. Previous review receipts remain in
[the dev1 review history](hndl-cpu-validation.md).

EMA remains an option of shared `SessionConfig`/`SliderSession`. It is not a new
`ConditionalPG09` recipe option or an added objective. CPU inference tests use an
explicit fixture factory, while separate tests exercise shipped device/rank/
precision/base validation.

The coordinated core owner reported reviewed CPU and bounded synthetic CUDA
software fixtures. This product migration runs no GPU job and does not qualify
Supra pretrained loading, T5/VAE/cache parity, native 256-pixel decoding, GPU fit,
convergence or image quality. Those remain centrally coordinated native gates.
No package release, deployment, model download, weight publication or merge is
performed. The actual product diff and receipts require root's Claude RETURN
review before landing.

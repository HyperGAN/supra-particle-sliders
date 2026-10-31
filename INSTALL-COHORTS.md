# Installation cohorts

The current packaged lifecycle uses `particle-sliders-core` 0.8.1 at
`32f6830f6402e85aa77b6631db64fcbbb2131f6b` with exact official ParticleGAN `0.9.1`.
Install from this repository root with `python -m pip install -r requirements.txt`.
The requirements entry delegates to `pyproject.toml`, so the two install paths
resolve the same immutable shared core. No unpublished PyPI core release is assumed.

The minimal package supports the shared lifecycle and synthetic CPU fixtures.
For the separately admitted native CUDA path, install a suitable PyTorch build
and `python -m pip install '.[native]'` in that same environment. The `native`
extra supplies the pinned text encoder and native loading dependencies; it does
not qualify pretrained output. `supra.lifecycle:BACKEND` and
`supra.native_inference:BACKEND` are the installed lifecycle factories.

`requirements-legacy.txt` preserves the previous installer verbatim for historical
trainer reproduction in a separate environment. Existing source locks, model
identities, checkpoints and receipts retain their original meaning. Do not combine
legacy requirements and the packaged lifecycle in one environment, or repoint a
historical lock to newer bytes. The product runtime still verifies its own
canonical source map; a successful installation alone does not qualify execution.

This repairs installation consistency. It does not adopt core0.11.0/PG0.9.3 or
PG0.9.4, migrate checkpoints, qualify pretrained/GPU output, or enable refused
continuous-response/render operations. Those require distinct reviewed cohorts.

Keep this PG0.9.1 environment separate from core0.11.0/PG0.9.3 and
HyperGAN develop, which currently requires PG0.9.4/HNDL0.10.
These Git dependency wheels are for source installation, not a PyPI release.

The common lifecycle uses the packaged install above. The historical
train_pg09.py and released-LoRA inference use requirements-legacy.txt and
their unchanged CDF source lock in a separate environment.

## Upgrading the packaged lifecycle

Keep the current exact dependency and canonical `supra/core081-receipt.json`
paired until a replacement product cohort is qualified. A newer core wheel is
not admitted by changing only a version or replacing hashes in this receipt.
The shared core 0.12 / PG 0.9.4 / HyperGAN develop work is coordinated separately;
its CPU qualification does not qualify Supra's native model or its checkpoints.

A promotion must add a distinct source receipt and product cohort, verify the
installed wheel and entry points, and rerun native adapter ownership, signed and
zero-strength responses, config/source rejection, exact split resume and atomic
corrupt-checkpoint rejection. Any required response-domain admission must be
provided and qualified upstream; changing shape or bypassing width admission is
not an upgrade. Pretrained loading, native CUDA/T5/VAE parity and image quality
remain separate checks. Preserve the old receipt, loaders and checkpoint labels
so existing artifacts continue to name their original source cohort.

An opt-in [HyperGAN/HNDL candidate profile](profiles/hndl/README.md) now has
separate 0.3.0 wheel metadata and build plumbing. It registers `supra_hndl`,
executes real HNDL native LoRA graphs and the shared HNDL spatial critic, and
uses the common core0.12 game/lifecycle. The root 0.2.0 installer still selects
the supported core0.8.1 cohort. Use one profile per environment; the two exact
core pins are intentionally incompatible. Candidate CPU tests do not promote
the default or qualify pretrained/GPU output.

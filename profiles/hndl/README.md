# Supra HyperGAN/HNDL candidate

This separately built 0.3.0 wheel registers `supra_hndl`, using the real shared
ParticleGAN game, HyperGAN develop builder, HNDL LoRA branches and shared HNDL
spatial critic. The frozen Supra DiT and native sampler are unchanged. The
ordinary linear LoRA law is preserved; these branches are not routed particles.

It requires the exact source cohort in `supra/core012-receipt.json`. Build from
the repository with `python scripts/build_hndl_profile.py --out artifacts/hndl`.
Install the resulting wheel in a separate virtual environment, using
`python -m pip install --find-links /path/to/verified-core-wheel artifacts/hndl/supra_particle_sliders-0.3.0-py3-none-any.whl`.
The stable
0.2.0 root wheel and this candidate cannot be installed together. Do not combine
their requirements or override the resolver. There is no public core0.12 release
or fetchable VCS pin for this local candidate. Supply the owner's exact wheel
with SHA256 `9afad3afc1f6c653a574074dc14f0da1f1c43975c31ca45c4eac049df0b0b4ad`.
The source receipt verifies every packaged shared Python/JSON/HNDL file, so
another wheel with the same version is refused before native preparation.
Public installation promotion awaits the shared owner's reviewed publication.

The native factory accepts the same local campaign/config/cache/context/base
checksums as the stable lifecycle, with backend `supra_hndl` and a new explicit
cohort name. It uses shared training, response, render, evaluation and full-state
resume APIs. Raw/EMA native-only v2 exports use the distinct `supra_hndl` inference
factory without training contexts/cache/game construction. The stable export
format and historical loaders retain their original roles.
No old checkpoint or receipt is converted or relabeled.

Native CUDA/T5/VAE loading, pretrained cache parity, actual 256px decoding,
GPU fit and image quality need separate qualification. CPU tests use tiny native
DiT models, synthetic caches and an explicit staged fixture runtime; their
results do not qualify those native operations. No new GAN loss is introduced.

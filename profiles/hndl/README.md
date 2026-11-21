# Optional Supra HNDL profile

The optional wheel uses the same reviewed core 0.15.2 and ParticleGAN 0.11.1 as
the canonical Supra package. It registers the separate `supra_hndl` backend and
keeps the pinned HyperGAN/HNDL native LoRA binding and shared spatial critic.

```sh
python scripts/build_hndl_profile.py --out dist --sdist
```

Install the HNDL wheel in its own Python environment. Use the matching original
checkout and dependencies for an earlier profile’s saved artifact. The
[archived HNDL runtime and evidence](https://github.com/HyperGAN/supra-particle-sliders/blob/b35e634f4611687c22b1254555c7c1eeec7692fe/profiles/hndl/README.md) retain their original pins.

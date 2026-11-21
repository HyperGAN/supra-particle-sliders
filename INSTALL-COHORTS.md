# Installation cohorts

Install this checkout in a fresh Python 3.11 or later environment. It pins core
0.15.2 at `b1f780decc516808cfd3019d5e5b674457161f86` and ParticleGAN 0.11.1.

```sh
python -m pip install '.[test]'
```

The native image factory uses the same checked core. Install `.[native]` for its local runtime dependencies or use the [separate HNDL package](profiles/hndl/README.md).

Replay older FULL/RAW artifacts from the matching product revision and environment.
The [archived installation guide](https://github.com/HyperGAN/supra-particle-sliders/blob/b35e634f4611687c22b1254555c7c1eeec7692fe/INSTALL-COHORTS.md)
retains their exact source and dependency pins.

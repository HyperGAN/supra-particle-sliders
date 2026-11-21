# Shared core

The installed `supra` image backend uses the exact core and ParticleGAN
release pinned in [the backend lock](../backend.lock.json). The product
owns its DiT, native LoRA, teacher caches and source-bound asset I/O.
The shared core owns the adversarial game, optimizer, response adapter
and checkpoint lifecycle.

See [installation and source admission](../INSTALL-COHORTS.md) and the
[fresh cache contract](../FRESH-CACHE.md).

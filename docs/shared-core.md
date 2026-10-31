# Shared core

The packaged lifecycle uses `supra.lifecycle:BACKEND` with the exact core 0.8.1
dependency in `pyproject.toml`. See [installation cohorts](../INSTALL-COHORTS.md).
The historical source training entrypoints dispatch to `scripts/train_pg09.py`.
They execute the shared native-response `particle_sliders.pg09.ConditionalPG09`
GAN. The earlier FormulationGame demonstration optimized a projector on frozen
pooled velocity edits, did not train or export a native Supra LoRA, and has been
removed from the product train entry.

See [the source, asset, inference and full checkpoint contract](pg09.md).
The packaged lifecycle owns the unchanged native DiT/LoRA and campaign wiring in
`supra/native_model.py`, `supra/cache.py` and `supra/campaign.py`; the GAN, critic
and response adapter remain shared. The historical CDF path still uses its
original shared source closure. This product keeps native loaders and
asset/checkpoint IO. Historical MSE
reproduction requires explicit `--allow-legacy-mse`.

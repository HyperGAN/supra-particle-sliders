# Shared core

All public Supra training entrypoints now dispatch to `scripts/train_pg09.py`.
They execute the shared native-response `particle_sliders.pg09.ConditionalPG09`
GAN. The earlier FormulationGame demonstration optimized a projector on frozen
pooled velocity edits, did not train or export a native Supra LoRA, and has been
removed from the product train entry.

See [the source, asset, inference and full checkpoint contract](pg09.md).
Architecture/LoRA and campaign implementations live in the shared package;
this product keeps native loaders and asset/checkpoint IO. Historical MSE
reproduction requires explicit `--allow-legacy-mse`.

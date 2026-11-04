# One training command

```sh
./train --config training.example.json
```

`./train` calls the common `particle-slider train` dispatcher with this project's
`training_profile.json`. Install the shared frontend separately, then select this
product's existing pinned environment with `--runtime-python /path/to/product-venv/bin/python`.
Keep the product's dependency and source locks; the frontend does not repin them.
`PARTICLE_SLIDER_COMMAND` can name the installed frontend executable.

The example is a **template**, not a ready model download: replace the local
model/cache paths, hashes and native options with an owned, verified run's inputs.
Do not enable dummy mode as a substitute for native training. `supra/lifecycle.py` owns
its exact input schema and gates. Use `--plan` for allocation-free request inspection
and `--check` for the selected runtime's allocation-free backend validation.
A validation rejection is not a training run.

PG 0.9.1 adapter-only GAN (particles=None); published Final Boss weights used MSE. Native execution requires reviewed local campaign config and matching frozen context/cache hashes.

The `source_only` label records that a factory exists; it does not assert native
GPU or Modal qualification. Training uses that factory's unchanged math, saves
full checkpoints periodically, and exports through its existing lifecycle.
The command performs no LR search, sample generation or publication. Resume via
`--resume /path/to/complete/checkpoint` preserves the original recipe identity.
Cloud placement requires a separately qualified private runner profile.

Existing product commands and source files are unchanged. This adoption adds
only the dispatcher profile, wrapper and usage template.

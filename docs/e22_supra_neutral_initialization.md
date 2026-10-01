# Full-Supra verification of neutral particle initialization

This applies the initialization control from ParticleGAN
[PR #227](https://github.com/255BITS/ParticleGAN/pull/227) to the actual Supra
backbone. Two fresh gated particle adapters retain all 71 routing sites,
rank-16 branches, a shared 128-by-4 bank, source/time conditioning, native
RpGAN/KA2, DV12 and zero learned-feature-harm structural guards.

Both arms use the public `particlegan.init.initialize_` API with
`sample_distributions_v1` and independent CPU streams named by role and full
parameter name. Both zero their up factors. The intervention then zeros only
the initial additive hidden bridge H and bias b before policy and EMA
construction. H, b, C, bank and router remain trainable. Legacy initialization
and saved checkpoints retain their previous behavior.

The [fixed protocol](e22_supra_neutral_initialization_protocol.json) declares
editing-only training from update one through 6,400, with full evaluations at
5,120 and 6,400 edits. The first endpoint matches the original ordinary
LoRA's editing-update count; the final endpoint gives both fresh particle
arms the same larger budget. Both are mandatory. No metric selects a seed,
head, checkpoint, horizon or stopping decision.

Primary evaluation uses clean FAST under both predetermined historical
critics, D1,856 and V2 D6,400, over all 240 held-out editing contexts. Every
model receives identical private CPU72 panels. Historical ordinary LoRA and
particle V2/V3 exports are rescored on those panels and labeled as historical
cohorts. They differ in initialization, schedule and native revision from the
fresh causal controls. Output RMSE and preservation are secondary diagnostics;
output error never enters the particle optimizer or structural decisions.

The native version is merged `develop` at
`6ec7e5788e14ea15ddc3e16ac71110458108b6a6`, with #223 included. ParticleGAN
[PR #226](https://github.com/255BITS/ParticleGAN/pull/226) contains a separate
paired-residual critic diagnostic and changes no native package source.
A CPU probe of Supra's actual saved critics confirms conditional odd forces
at zero residual. That observation does not establish a training improvement,
and no critic parity or Gaussian-estimator changes are mixed into this run.

The live graph is [http://pop-os:8784](http://pop-os:8784), bound to `0.0.0.0`.
It shows the fresh control and neutral-start curves, a separately rescored
ordinary-LoRA level to beat on the exact same small held-out probe, rolling
training game losses, and read-only plateau diagnostics. Full-set endpoint
results are reported separately from progress probes.

```sh
CUDA_VISIBLE_DEVICES=0 \
PARTICLE_SLIDERS_ROOT=/ml2/hypergan/supra-e22-verification/vendor/particle-sliders \
python -u scripts/experiment_e22_supra_neutral_initialization.py \
  --particlegan-root /ml2/hypergan/ParticleGAN-supra-neutral-develop \
  --output outputs/e22-supra-neutral-initialization-6400 --device cuda:0 \
  > outputs/e22-supra-neutral-initialization-6400.log 2>&1

python scripts/serve_e22_supra_neutral_initialization.py \
  --run outputs/e22-supra-neutral-initialization-6400 --host 0.0.0.0 --port 8784
```

Use a fresh output directory for reproduction. Raw logs, checkpoints and
source snapshots remain in ignored local artifact storage. The run binds its
actual source hashes, protocol, inputs, recipe, panels and frozen critic
identities. Exact 800-to-802 native replay, frozen-owner checks, particle
ablations and FAST export/reload parity are required software witnesses.

Before training, 39 CPU initialization/adapter/load tests passed. Independent
review of the actual two initial native checkpoints passed 3,408 checks,
including identical non-H/b owners and matched early editing batches and
Gaussian streams. A full-Supra improvement remains unestablished until the
fixed endpoint evaluations and independent review complete.

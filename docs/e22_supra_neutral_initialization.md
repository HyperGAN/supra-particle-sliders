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
The [compact saved-critic probe](e22_supra_saved_critic_parity_results.json)
binds both critic tensors and all 60 source/time pairs. Native single-Gaussian
zero-residual force has mean normalized per-context RMS `1.569e-4` and
`1.845e-4`; the odd components isolated by antithetic pairing are `5.411e-5`
and `1.046e-4`. Evenizing alone leaves finite-sample forces because Supra
uses a single G draw. Evenizing with antithetic G cancels the force in this
observational probe. This does not establish a better trained generator.

The [token-force decomposition](e22_supra_frozen_critic_token_force_results.json)
checks the same saved critics and private Gaussian law. Their token-mean
component accounts for 3.51% and 10.60% of single-draw force energy; it does
not dominate the finite-noise forces. The critic pools shared token features
without position inputs, but nonlinear features still detect zero-mean token
variance. This observation does not identify a convergence cause.

The [2,000/2,400 controller observations](e22_supra_neutral_control_observations_2000_2400.json)
show no increase in G, router or bank rates and no output-noise change during
the temporary held-out regression. The last generator-gradient alignment
changes sharply at the two saved boundaries. These are observations under
different stochastic batches, not a causal replay or a sign-error proof.

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

The [zero-update basis audit](e22_supra_initial_basis_results.json) compares
all 71 fresh down factors with the historical trained ordinary LoRA. Their
mean squared principal cosine is 0.027653; the fresh spans capture 2.7625%
of its linear weight update. PR #227's teacher starts in the exact student
span, with 100% overlap. This identifies a limitation of that toy: it did
not test acquiring a different input basis. These weight-space observations
omit activation covariance and do not measure the caption target's span or
establish a cause of the full-model convergence gap. A separately declared
rotated-teacher toy is prepared for use if the fixed Supra comparison fails;
its quality training has not started.

The CPU-only artifact reviewer can audit an active prefix without qualifying
completion. After the run finishes, omit `--partial` to require every fixed
checkpoint, both endpoints, all common-critic reductions, retained particle
contributions, and the recorded GPU replay/export witnesses:

```sh
python scripts/review_e22_supra_neutral_initialization.py \
  --run outputs/e22-supra-neutral-initialization-6400 \
  --particlegan-root /ml2/hypergan/ParticleGAN-supra-neutral-develop
```

The reviewer checks CPU artifacts; it does not claim to repeat the GPU
training, native replay, or full-backbone export forward passes on CPU.

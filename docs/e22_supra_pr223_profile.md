# PR223 shared routed profile

The supplied profile is sufficient to enable PR223's shared additions in
Supra. ParticleGAN is pinned to
`bdf05d1be0f68cfdb0c71e81e7e0d3cce477572f` from
[PR223](https://github.com/255BITS/ParticleGAN/pull/223).

Fresh `scripts/train_e22_supra_full.py` runs select
`pr223_shared_routed_v1` and construct:

```python
get_recipe(
    "e22_routed", num_particles=128, z_dim=4, batch_size=4,
    output_noise_std=.125,
    birth_death_backend="auto", reopen_guard="settled",
)
```

The settled guard gates network reopening. Shared execution and checkpoint
fixes come from the pinned implementation. The automatic selector records
`actual_backend="routed"`, `sampling_backend="routed"`, and
`selection_reason="routed_rows_owns_controls"`. Its generator/noise rate
factor is 1. The shared bank, all 71 sequential sites, whole-model candidate
reruns, native per-site DV12, learned critic-feature structural scoring, and
zero per-context feature-harm guard remain active. Output accuracy does not
enter optimization, structural decisions, stopping or selection.

Atlas feature-cell counting, placement and birth are inactive. This is a
shared-controls integration, not a port of Atlas's population law. Such a
port still requires proposals that account for coupled token routing and
conditional guards; for this project their criterion must remain in the
learned game rather than raw output accuracy. Toy results do not demonstrate
Supra quality improvements.

## Checkpoint boundaries

Legacy default `knn`/no-guard recipe dictionaries remain byte-for-byte
compatible in PR223. Programmatic research callers keep their legacy default,
and the full training CLI resumes each checkpoint's saved profile. A missing
profile tag means `pr155_routed`. An explicit conflicting `--particle-profile`
is rejected. Normal provenance validation still rejects a ParticleGAN source
change during continuation.

Turning on `auto` and `settled` changes the recipe and introduces additional
state owners. A legacy checkpoint cannot be silently resumed as the new
profile. Qualification uses the old run's frozen teacher and cached contexts
to initialize fresh particle, router, critic, optimizer and controller state.
Its separately recorded legacy restore check tests source compatibility,
without representing the old run as a shared-profile run.

## Verification

The local CPU particle suite passes **109 tests**, including strict profile
rejection, unchanged routed ownership, exact recovery of the new owners and
atomic rejection of malformed guard state. The targeted upstream suite passes
**110 CPU tests and five CUDA/BF16 checks**, covering recovery across accepted
moves, full-model replay, optimizer transport and 71-site parity. An additional
combined `auto`/`settled` fixture passes exact CPU-loaded recovery on CPU and
CUDA, including subsequent native updates and clean serving.

The real 71-site Supra GPU smoke passes **47 checks** in 42.05 seconds on an
A6000. It initializes fresh particles through the public deterministic API,
runs five native updates including preservation, loads the step-two checkpoint
from CPU, and exactly reproduces updates three through five and complete
native state. All 128 bank rows receive gradients from step two onward. Both
training passes traverse all 71 sites; frozen tensors, fresh base identity and
post-training zero-strength identity remain exact. The separate historical
restore retains the original step-6400 state exactly. The final and replay
native digest is
`f04d3a7e3ddb1adf1ce67d659cd24abe3bffe9e4b9958110125382bf71fb672b`.

Local receipts are `outputs/pr223-upstream-profile-validation.json` and
`outputs/e22-pr223-shared-profile-smoke/result.json`, with source hashes,
configuration, native rows and ownership checks. The latter log is
`outputs/e22-pr223-shared-profile-smoke.log`. An initial report-serialization
failure is retained under `outputs/e22-pr223-shared-profile-smoke-aborted-json-report`;
the identical smoke was repeated after fixing tensor diagnostics serialization.
An independent CPU artifact review passes 97 checks against actual source
snapshots, input checksums, the stored checkpoint, raw rows and replay;
its receipt is `outputs/e22-pr223-shared-profile-smoke/independent-review.json`.

Reproduce the real-model smoke without changing the shared Python environment:

```bash
CUDA_VISIBLE_DEVICES=1 /ml2/ntc-image-studio/.venv-anima/bin/python \
  scripts/verify_e22_supra_pr223_profile.py \
  --particlegan-root /ml2/hypergan/ParticleGAN-pr223-supra \
  --output outputs/e22-pr223-shared-profile-smoke
```

Choose a fresh output directory for a new invocation. The smoke is a
compatibility and recovery check; its short horizon does not establish that
the new guard improves convergence or beats the original LoRA.

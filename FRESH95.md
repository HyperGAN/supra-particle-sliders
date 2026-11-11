# Explicit cached fresh initialization (Supra HNDL)

`supra_fresh95` is an opt-in Python API for the original HNDL adapter-only lane. The registered `supra` route, command line, existing checkpoints and [measured learning-rate results](LEARNING-RATES.md) keep their existing behavior. This API does not migrate or reinitialize an old cohort, and it has no particle bank or weight EMA.

Use an already completed original Supra response cache, its whole-file checksum, and the original 19-context data plus matching JSON provenance sidecar. Keep the native configuration, model revisions, frozen-base identity, FP32 host and original field geometry. The API validates those original assets; it never collects teachers, encodes replacement contexts or runs teacher parity. An owner supplies the admitted HNDL runtime factory and remains responsible for staging, device ownership and final runtime unloading.

The package includes the original `supra/supra_lora.hndl` and `supra/core013-pg095-hndl-receipt.json` in their original locations. The HNDL context must also provide the exact core0.13 Git/source closure, official ParticleGAN0.9.5 initializer and original HNDL/HyperGAN dependencies accepted by the unchanged guards. Version strings alone are insufficient. The separate historical HNDL profile builder is unchanged; this addition does not claim that builder installs the sibling helper or that a public pretrained HNDL environment is qualified.

```python
from particle_sliders.lifecycle import SessionConfig
from supra_fresh95 import open_training, open_inference

# Owner-supplied cached_options and callbacks; no implicit model/cache loader.
config = SessionConfig("supra_hndl", "a-new-fresh95-cohort", **cached_options)
with open_training(config, "new-fresh-run", mode="fresh",
                   runtime_factory=owner_factory,
                   owner_admission=owner_admission, ema_decay=None) as trainer:
    metrics = trainer.update()                 # explicit new work
    full = trainer.save("new-full.pt")
    raw = trainer.export("new-raw.pt")

# Reuse the same config/cohort and a new destination. No initialization/teachers.
with open_training(config, "new-resume-run", mode="resume", resume=full,
                   runtime_factory=owner_factory,
                   owner_admission=owner_admission, ema_decay=None) as restored:
    pass

with open_inference(raw, runtime_factory=owner_factory,
                    owner_admission=owner_admission, ema_decay=None) as inference:
    image = inference.render(prompt="an explicit prompt", strength=1.0, seed=7)
```

`cached_options` is the original HNDL options dictionary: `native_config`, `cache`/`cache_sha256`, `contexts`/`contexts_sha256`, `frozen_base_sha256`, explicit `device`, and the original rank, horizon, batch and seed. Set `ema_decay=None`. The factory receives the requested operation and options; RAW inference receives only its native scaffold options, without training assets. Fresh and restore scaffolds must retain the original HNDL modules and zero UP weights before the saved RAW/FULL state is applied. If factory construction fails, the factory must release its own partial runtime.

Fresh mode runs the public PG0.9.5 initializer once on the existing down Parameters before optimizers. QR preserves the original Kaiming(a=√5) variance, `1/(3·fan_in)`; it does not preserve a Uniform entrywise bound. UP stays zero and the critic retains its constructor values. The receipt is bound into the game signature; FULL0 and RAW0 are saved before the entry returns. Strict FULL resume restores the whole typed game, optimizers and RNG without initializing. RAW inference restores only adapters, without a critic, cache or initializer. A failed close retains pending restoration work for retry; close does not unload the caller's runtime.

Four distinct tiny synthetic software cases have passed across retained executions, with one tiny update in total. They cover identity/zero-UP/critic policy, deterministic QR moments/RNG, strict FULL/RAW and retryable cleanup. The corrected QR case ran separately; the other three passing cases and their update were retained. The helper was copied into the original installed HNDL context for those observations. A newly distributed helper, public HNDL packaging, pretrained rendering, native parity and any new LR/effect result remain unqualified by these software observations.

## Optional installed-owner software check

The separate fixture is `tests/fresh95_owner_check.py`. Its name deliberately stays outside default pytest file discovery, so ordinary repository tests keep their existing scope. To select it explicitly, first use genuine installed wheels with the frozen compatible original dependencies/provenance required above. An editable installation or repository Source/PYTHONPATH is not an equivalent installed-owner context.

Copy this exact fixture alone into an empty directory outside the repository, change into that directory, and use the compatible installed environment's Python:

```bash
/absolute/path/to/compatible-installed-env/bin/python -I -m pytest \
  -c /dev/null --import-mode=importlib /absolute/neutral-directory/fresh95_owner_check.py
```

The neutral directory contains no repository configuration, conftest or Source packages. `-I` ignores Source PYTHONPATH; `-c /dev/null` avoids the repository's pytest configuration. This is an explicitly selected synthetic software check, not a public CI/native-availability claim or a requirement to rerun completed observations. Existing tests/workflows and original guards are unchanged.

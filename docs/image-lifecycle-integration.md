# Supra image lifecycle migration, source-inspection plan

This records the initial 0.7 draft. The later parent-approved
[thin product 0.8.1 source proposal](thin-product-core081.md) preserves this
milestone and its receipts; its prepared equivalence tests have not run.

This is a new, unpublished source cohort against particle-sliders-core 0.7.0 at
`fd89e1c780ab5210cd1be98ec3b70977e6349379`. The isolated product base is draft
[PR4](https://github.com/HyperGAN/supra-particle-sliders/pull/4), exact head
`dabeb2ad091d406e89a1281e03d02d3a88e43436`. Inventory main
`a908cc8ace35e4019a19bd25c1af7fd7e9ec72b6` was not selected as a migration base.
The PR payload, legacy requirements, CDF source lock, receipts and checkpoints
remain available. No published refs or active environments are changed.

## Actual entrypoints and ownership

| Path | Current behavior | New lifecycle scope |
|---|---|---|
| `supra.runtime.SupraRuntime` | Local-only pinned Supra DiT/T5/VAE loading, actual native LoRA, CUDA required | Reuse behind factory; no weight loading in preflight |
| `scripts/prepare_pg09_contexts.py` | Exact config/revisions, 19 frozen contexts including stored unconditional, runtime source receipt | Reuse its staged assets; do not re-encode during training |
| `scripts/train_pg09.py`, `train_lora_supra.py`, `python -m supra.train` | Native shared GAN, fixed 28800 maximum, batch4, legacy product envelope | Retain historical CLI; new installed `particle-slider` commands use registered backend |
| `supra.pg09.build_run`, `verify_teacher_cache` | Context/cache/base validation and shared `build_supra_campaign`; teacher parity before native worker updates | Factory retains equivalent validation and shared parity gate |
| `supra.pg09.save_checkpoint/load_checkpoint` | Immutable directory receipt + full game + native safetensors; CDF/product source identity | Original schema stays on original loader; new lifecycle full state/EMA has separate source/cohort |
| `scripts/infer_supra.py`, `SupraRuntime.render` | Metadata selects trained CFG policy, native seed/noise, increasing time, positive Euler, VAE | Product callback reuses prepared backend and native sampler; returns PIL only |
| `scripts/distill_final_boss.py` | No-grad rank compression, carries original CFG/source policy | Unmigrated; not GAN training or common inference export |
| `train_final_boss.py`, `converge_final_boss.py`, launcher | Explicit opt-in archival supervised MSE | Unmigrated historical reproduction |
| release build/publish scripts | Historical artifact packaging/publication | Outside migration scope |

The generator is **adapter GAN**, `particles=None`. The shared backend registers
only actual FP32 native LoRA leaves once. Supra's frozen 104.1M DiT stays native;
no backbone rewrite is authorized. Fields stay `[1,4,32,32]`, masked neutral T5
pooling plus time/path/preservation flags, balanced sampling with existing
edit/preservation repetition and exact calibrated cache scale. CFG applies the
adapter only to the conditional branch; frozen unconditional stays unadapted.
Time remains `t=0` noise to `t=1` data, Euler `z += v/steps`; VAE scaling is .18215.
Finite signed strengths are supported, references are unsupported, base weights
remain FP32 with native BF16 CUDA autocast. These are interface capabilities,
not pretrained quality qualification.

## Bounded implementation

1. Add package metadata and `particle_sliders.image_backends` registration; new
   dependency points product -> core -> ParticleGAN. Preserve `requirements.txt`
   and `shared.lock.json` as legacy CDF records. A separate packaged source lock
   validates exact 0.7 milestone bytes in a scoped product resolver.
2. Validate JSON options, fixed recipe bounds, config and local asset hashes,
   full frozen-context receipt and native cache schema before native allocation.
   Accept only local weights; caller supplies context/cache/frozen digests.
3. Factory constructs native runtime itself (or accepts already staged runtime
   for later CPU tests), validates frozen weights, freezes text training to
   exported contexts, calls shared `build_supra_campaign`, then teacher parity.
   Include product sources, context receipt and VAE identity in model signature.
4. Transfer complete game, cleanup and native sampling callback to `Binding`.
   Restore caller-owned cache/encoder/CFG, scales, modes and checkpoint hooks on
   failure/close. Keep HyperGAN run presentation on common `on_step`/manifest/
   artifacts. Native callback is image inference only.
5. Leave installed-wheel tests, fixtures, model loads and resource admission to
   parent scheduling. Initial verification is source/AST/metadata inspection only.

## Core rough spots / review boundaries

Follow-up: [clean core 0.8.1 adoption assessment](core081-adoption-assessment.md)
records the parent's reviewed fixes and the missing Supra source-closure gate.
The points below retain the original 0.7 inspection context; PG already refused
training from inference exports, even before the newer session diagnostic gate.

- At exact fd89e1c, `SliderSession.load_inference` marks `inference_only=True`,
  but `step()` and `train()` do not check it. Shared docs promise adapter-only
  artifacts cannot resume training. Fix in core; no product workaround.
- `_observation` restores adapter buffers/modes and RNG, but arbitrary runtime
  caches and native model modes are product responsibilities. Supra's callback
  will restore new-prompt text cache/token counts around rendering.
- Compact inference-only loading is parent/core-owned. This factory reconstructs
  its qualified prepared cache/game, as the exact milestone requires.
- `CampaignFieldCritic` (shared spatial convolution/FiLM critic), shared
  `supra_model` LoRA modules, and native DiT definition require separately
  parity-tested HNDL migration. Do not silently substitute critics or network laws.
- Parent early-report dispatch to supplied thread id returned **thread not found**;
  outcome is failed delivery, not an acknowledged core issue.

No suites, dependency installs, native model loads, CPU-heavy builds, GPU work,
model downloads, publication, merges or active pin changes were performed in the
source inspection phase. Later conformance remains pending admission.

## Light implementation receipt and scheduled validation queue

Implemented `supra.lifecycle:BACKEND`, new package registration/dependency pin,
separate 39-file exact source/data lock, scoped resolver, owned native campaign
factory, source-bound model identity, local-only runtime construction, frozen
training encoder, full shared teacher parity gate and reversible native callback.
The original requirements, root shared lock and backend lock are byte-identical
to PR4. Changing `shared.py` creates a new product source identity: this draft
does not claim to restore old CDF packets from the edited checkout. Use their
original source cohort/loader, as before.

Static checks only: both edited/new Python modules parsed with `ast.parse`,
`pyproject.toml` parsed with Python 3.11 `tomllib`, backend registration and
dependency direction inspected, all 39 source/data hashes matched the exact
local milestone, copied CDF lock matched original, and `git diff --check` passed.
The first metadata attempt used system Python without `tomllib`; it exited
before checks and was corrected using an existing Python 3.11 interpreter with
bytecode writing disabled. No interpreter environment was changed. No wheel was
built, so wheel hashes and installed origin proofs do not exist for this draft.

Parent should schedule one isolated CPU worker, offline and single-threaded,
to build/install source-cohort wheels outside the source checkout and qualify:

- Existing tiny real Supra DiT/LoRA fixture with a separately identified synthetic
  cache scale (never label that fixture as the reviewed native cache).
- Exact split training/resume: next metrics, complete state, EMA, RNG, optimizer
  clocks and balanced row coverage; full export/load velocity parity.
- Invalid config/receipt/cache/base/source/artifact/EMA/optimizer restores reject
  before mutation; original legacy packet diagnostics remain clear.
- Zero, negative and fractional native strengths, unsupported references/output/
  precision gates, update/setup failure cleanup and staged-runtime restoration.
- Observation restores RNG, adapter modes/buffers, native text cache/counters,
  scales and checkpoint hooks; new-prompt rendering leaves training contexts fixed.
- Installed entry point discovery, missing/shadowed dependency diagnostics,
  common CLI and dependency-direction guard. Core must fix inference-only update
  gating before claiming the exported-artifact training rejection requirement.

Actual native loading, 19-context preparation, cache/base parity, 256px decode,
GPU fit and signed image quality remain separately resource-admitted gates.
Core/HNDL changes and compact inference require a new exact pin and compatibility
pass coordinated through parent; this derivative does not edit shared sources.

# Supra thin product source proposal for clean core 0.8.1

Status: **source implementation prepared; no derivative execution or conformance
claim**. Parent approved the thin product relocation after the source-closure
assessment. The proposed package dependency now names immutable clean core
`32f6830f6402e85aa77b6631db64fcbbb2131f6b`, PG 0.9.1. It is local/unpublished.
No environment was installed or repinned. No source cohort or active receipt was
rebound. Earlier product commits `3ed555b4b53c7c0a9892a3d5b6e35a2fc3ebf3a4`
and `f1f33f645995166cd18c7d79f7ec5de33d05754d` preserve the 0.7 draft and its
initial assessment. Original PR4 and CDF requirements/root locks remain intact.

## Source closure and ownership

| Piece | New owner | Preservation boundary |
|---|---|---|
| Supra native DiT/ordinary LoRA/checkpoint definition | `supra/native_model.py` | Byte-identical to fd89e1c `particle_sliders/supra_model.py`; original source SHA and MIT/Apache notices packaged |
| Exact 19-context guards/export/load, tensor receipts, native teacher rows | `supra/cache.py` | Seven named definitions moved unchanged, with exact old config digest, native geometry, masked conditioning and repetition policy |
| Native run cases/render callback, campaign construction, teacher parity gate | `supra/campaign.py` | Three named Supra definitions moved; sole body rewrite is importing `ConditionalPG09` from its public package |
| Spatial critic and RNG initialization | `particle_sliders.field_critic` | Imported directly, including historical alias names; no local critic or initialization-law copy |
| Native response/CFG/flow schedule, GAN, optimizers, penalties, priors | `particle_sliders` | Existing common implementations; no math/optimizer/lifecycle machinery relocated |
| Native-only identity, local loading, reversible native state and sampling dispatch | `supra/native_io.py`, `supra/native_inference.py` | Product callbacks use the actual native model and shared response backend; no training rows/cache/game/critic construction |

[Packaged provenance](../supra/thin-product-provenance.json) records original
commit/path/file hashes and per-definition source-segment hashes. The original
MIT and Supra2-IMG Apache-2.0 texts and a new explicit relocation notice live
under `supra/licenses/`, and are included as package data. Existing historical
`licenses/particle-sliders/` notices are unchanged. No model weights are included.

No whole `pg09_remaining_campaigns` or `pg09_moss_campaign` module was copied.
FinalForm builders, MOSS rows/attachment, campaign scheduling and proposal helpers
were excluded. `teacher_parity` is a read-only Supra asset admission check; it
does not become an optimized loss. `build_supra_campaign` only wires the actual
native adapter/provider/critic into the common game with the original recipe.
Native DiT/LoRA definitions stay unchanged; HNDL replacement is separate work.

## Provider injection and clean API adoption

The existing common provider protocol already accepts the product-owned
`SupraCampaignRows` object through `ConditionalPG09(backend, rows, critic, ...)`.
Its `identity`, `conditions`, `__len__` and `__getitem__` supply source/data-bound
native `PreparedResponse` rows. No new shared hook or reverse import is needed.
Fields remain `[1,4,32,32]`, particles=None, batch4, balanced routing, exact
cached response scale, native increasing flow time and conditional-only CFG.

The new factory imports the product provider/constructor rather than missing
historical core modules. Native loading injects the exact relocated model module
through an optional `SupraRuntime(native_model=...)` keyword; historical callers
still default to their original shared model loader. Defaults and all native
forward/attention/LoRA/CFG/VAE calculations are unchanged. Runtime source changes
are strict new cohort identities; old context/checkpoint receipts require their
original source checkout, not the edited proposal.

The resolver consumes the **unaltered canonical 30-entry producer map** from
`supra/core081-receipt.json`, identical to the parent's canonical receipt. It now
understands the canonical package-src-relative keys, including the compatibility
namespace, while preserving legacy root-relative locks. The existing 0.7 lock
is retained and is not relabeled or extended with invented hashes.

`Backend.preflight` rejects unsupported native request/device/rank/decoder
combinations before native/RNG preparation. JSON/path/config/hash validation
does not deserialize tensors; full context/cache/provider checks remain in
preparation before native loading. `Binding.before_update` is a pure decision
over established native evidence and parameter ownership flags; it does not
execute the model, read assets or consume RNG. `Binding.observation` restores
the native cache, encoder selection, token-counter object/values, CFG/scales,
model modes/gradient flags and checkpoint settings on success or failure.
Core 0.8.1 independently owns adapter buffer/state and RNG restoration.

`particle_sliders.inference_backends` separately registers a render-only native
factory. Training creates an `InferenceSpec` from its actual native backend and
adapter; `session.export_inference(selection='raw'|'ema')` emits v2 artifacts.
The factory accepts local device/precision/base identity and the artifact's rank,
not training config, contexts or cache. The new native identity includes actual
base/source/architecture/protocol and model/T5/VAE revision identity without
training data. PIL-only rendering and finite signed strengths are declared
explicitly. Velocity response on training rows remains the training session API;
no data-independent velocity or latent-output feature is fabricated.

Native-only artifacts cannot resume training. The old v1 export API and native
safetensors remain explicit original-cohort formats; no converter is provided.
HyperGAN remains responsible for run presentation via the common observer seams.

## Prepared tests, deliberately not executed

`tests/core081/test_thin_relocation.py` prepares exact native bytes and selected
AST equivalence, native parameter names/shapes/dtypes/ownership, matched native
forward/gradient probes at zero/negative/fractional/unit scales, native provider
identity/targets/condition/routing equivalence and shared critic/RNG provenance.
The preserved source path is reference data only, never added to `sys.path`.

`tests/core081/test_lifecycle_closure.py` prepares real tiny-native GAN split
resume with complete metrics/state/EMA/RNG/coverage, legacy frozen digest
equivalence, corrupt restore atomicity, update-admission rejection, native
observation isolation on success/failure, preparation cleanup, request preflight,
and raw/EMA native-only load after deleting training assets and forbidding all
training constructors. Its explicit test-only CPU factory does not relax the
production CUDA loader. Tiny VAE output is a software fixture, not 256px decode
or pretrained image qualification.

No test, product import, native model allocation or GPU operation was executed.
Only static Python parsing, metadata/source/license inspection, byte checks and
whitespace inspection are permitted in this phase. The next explicit CPU slot
must build/install from the exact proposed product/core commits, verify package
contents/origins and run these tests from `/tmp` with source injection disabled,
one thread, no visible CUDA and offline environment variables. Set
`SUPRA_REFERENCE_CORE` to task-5/core-milestone's preserved fd89e1c package source
directory; omitted-reference skips do not qualify adoption.

Pending beyond those tests: fresh-process native inference with training inputs
absent, original legacy source-cohort compatibility checks, full negative packet
and public CLI gates, and any issues surfaced by the first executed suite.
Actual pinned native loading, T5/VAE/base/cache parity, image quality, GPU fit and
HNDL equivalence remain separately admitted. The new dependency proposal is not
a tested derivative promotion and introduces no active pin or public write.

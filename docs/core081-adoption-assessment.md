# Supra assessment of clean core 0.8.1

Historical assessment: parent subsequently selected the thin product path.
[The follow-up source proposal](thin-product-core081.md) resolves the missing
closure in product code and stages the exact candidate dependency. Execution,
equivalence qualification and derivative promotion remain pending a CPU slot.

Decision: **pin adoption blocked by the clean package source closure**. This is
not a ParticleGAN downgrade or a proposed change to the mathematical formulation.
The operative local proposal remains core 0.7.0 at
`fd89e1c780ab5210cd1be98ec3b70977e6349379`, product implementation commit
`3ed555b4b53c7c0a9892a3d5b6e35a2fc3ebf3a4`, based on preserved public PR4
`dabeb2ad091d406e89a1281e03d02d3a88e43436`. No new lock was invented and no
active pin, shared checkout, public branch or old receipt was changed.

## Exact candidate and canonical evidence

Candidate: clean core **32f6830f6402e85aa77b6631db64fcbbb2131f6b**, version 0.8.1,
PG 0.9.1. Canonical handoff is task-2/CORE081-HANDOFF.md; canonical build map is
task-2/evidence/core081-build/receipt.json. The parent reports independent review
passed, including 220 tests, all 30 files matched and no actual CUDA execution.
The copied [producer receipt](../evidence/image-lifecycle/core081-build-receipt.json)
is byte-identical to that canonical receipt, SHA256
`81c9407544a97568ff8f8eb5d78166595cb7971837f14ef0e08dc78a3bb38046`.
Its older `review_status` remains producer data; the parent's newer review result
is the authorization to assess adoption. The producer's wheel SHA256 is
`279aaab83da05b6ebeca030dfe115e1ae2232ca4924fe4278ea48103918fbd8d`;
no derivative wheel was built or installed in this task.

Read-only local fetch placed the candidate object in task-5/core-milestone.
Git warned that shallow roots could not be updated; subsequent `cat-file -t`
confirmed the exact commit is available. That checkout remains at fd89e1c and
clean. All 30 canonical source-map entries were independently compared with
the exact candidate's Git blobs using SHA256, without importing the core or
executing model/test code. This was a static source check, not a test suite.

## Concrete source-closure blocker

The clean main package deliberately excludes these modules:

| Missing module | Actual Supra dependency | Consequence of a bare pin update |
|---|---|---|
| `particle_sliders.supra_model` | `supra.runtime.model_module/model_source` loads the exact native Supra DiT and ordinary LoRA definition | Native runtime construction and existing real CPU fixtures fail |
| `particle_sliders.pg09_remaining_campaigns` | `supra.lifecycle._assets/prepare`, legacy `supra.pg09`, context preparation and tests | Frozen-context guards, native rows, campaign constructor and teacher parity disappear |
| `particle_sliders.pg09_moss_campaign` | Imported by the previous module for critic/utilities and unrelated FinalForm implementation | Restoring the whole historical module would reintroduce excluded MOSS orchestration and reverse-import concerns |

The canonical clean source map has no hashes for these absent modules. Relabeling
the 39-file 0.7 lock as 0.8.1 would be false; adding local extra files to a 0.8.1
install would create an unreviewed package closure. Neither is implemented.

The clean package does retain `SupraRuntimeResponseAdapter`, `FieldCritic`,
`seed_initialization_rng`, `ConditionalPG09` and the new lifecycle/inference APIs.
PG remains 0.9.1 and the actual Supra generator remains adapter GAN,
`particles=None`, balanced batch4, unchanged field geometry/CFG/time/schedule.
The missing modules are source/ownership compatibility, not a need to change GAN
math, copy losses/optimizers or move this backend to PG 0.9.3.

Parent decision needed before implementation: approve a bounded relocation of
unchanged Supra-specific native model, context/cache policy and orchestration
into the product, reusing the retained shared FieldCritic/game, or supply an
explicitly reviewed common closure with a new canonical receipt. Avoid importing
the whole historical MOSS/FinalForm stack. Native model relocation is separate
from the later matched-weight HNDL migration; no network definition substitution
or checkpoint rebinding is justified here.

## Concrete hook adoption after source closure is resolved

1. Consume the single canonical `source_sha256` map. The resolver currently
   expects paths relative to `particle_sliders`; the canonical map is relative to
   package `src` and includes `concept_slider_core`. Adapt resolver path handling,
   preserving the canonical map bytes/keys rather than inventing a second map.
2. Register `particle_sliders.inference_backends` with a native-only Supra
   `InferenceBackend`. Its factory constructs only native base/LoRA/T5/VAE and
   `SupraRuntimeResponseAdapter`, with no cache, rows, critic, game or optimizer.
   Bind verified base/model/T5/VAE/source identity and architecture options
   (rank, targets, alpha) in `native_identity`/`InferenceSpec`; keep cache/context
   receipt identity in the training signature alone. Raw/EMA export comes from
   `session.export_inference`; old v1 packets keep their original explicit loader.
3. `Backend.preflight(options, request, staged)` validates native prompt/steps/
   CFG/output/geometry and staged CPU versus native CUDA support before allocation.
   Move cache tensor reads out of allocation-free `validate`; retain JSON/receipt/
   path/hash guards there and perform full native cache validation in preparation.
4. `Binding.before_update` uses pure existing evidence about frozen native assets
   and teacher parity, without model execution, mutable callbacks or new I/O in
   the admission hook. A rejected update must leave training and EMA untouched.
5. `Binding.observation` and `InferenceBinding.observation` own native text cache,
   token counters, model/T5/VAE modes and any mutable state outside the adapter.
   Replace the render-only context with this common hook so response evaluation
   receives the same isolation. Compose `native_cleanup` with the existing LoRA
   scales/cache/encoder/CFG restoration and product-owned cleanup.

The old missing session-level inference check is resolved in 0.8.1. Correction
to the initial report: PG already blocked updates from inference exports in the
earlier cohort; the newer session gate provides earlier diagnostics. The 0.8.1
buffer/device restoration changes are accepted core fixes, not newly established
Supra conformance. No product workaround for those fixes was introduced.

## Pending execution and outcome

The pin update and native-only inference/hook implementation are **pending the
source-closure decision**, not denied by an approval reviewer. No uncertain
mutation remains; the fetch object's availability and unchanged checkout were
checked explicitly. CPU installed-wheel conformance remains pending a separate
explicit slot. No suites, dependency installation, build, native model load,
GPU execution, downloads, publication, merge or system changes occurred.

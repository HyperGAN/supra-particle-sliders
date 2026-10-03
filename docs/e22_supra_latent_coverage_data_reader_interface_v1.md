# Expanded Supra data: saved CPU reader interface

This is source preparation, with no execution or actual result. Root freezes
the reader and JSON preregistration before the single GPU600 pure-teacher data
preparation. Existing helper, generation script, qualification files and cards
remain unchanged. The approved next training design uses the full14-block
trained12800 parent; this reader does not authorize or run that trial.

The command is:

```text
CUDA_VISIBLE_DEVICES='' <qualified-python> scripts/review_e22_supra_latent_coverage_data.py --descriptor <root-bound-descriptor.json> --out <exclusive-review-folder>
```

The descriptor has `id=supra_latent_coverage_saved_data_review_v1`, `seconds=120`,
`final_reserve_seconds=10`, and an exact `output` path. `sources` maps logical
names to `{path,sha256}`, including the frozen reader and preregistration.
`inputs` has these exact roles, each `{path,sha256}`:

- `artifact`: actual `latent-coverage-data.pt` produced after generation.
- `data`, `neutral_checkpoint`: exact original aliases from generation card.
- `generation_card`, `generation_report`, `generation_completion`.
- `generation_external`: canonical null-science qualification watchdog receipt.
- `generation_manifest`: the actual watchdog manifest, including all sources,
  inputs, card and launcher byte pins.

Root fills future hashes only from real completed outputs. This later metadata
finalization changes no predicate, cap, digest, shape, ordering or count law.
The generator ID remains `supra_latent_coverage_data_preparation_v1`, GPU600,
BF16 student training precision, with the original frozen BF16 teacher.

The reader lazily imports Torch with CUDA hidden and uses CPU mmap/weights-only
loads for the original38MB data, expected74–90MB expanded artifact, and original
1.76GB tensor-only native checkpoint. It hashes the additional two ~1.76GB judge
inputs required by the frozen loader, without loading or evaluating them.
Nested tensor/scalar digest computation follows the independently transcribed
qualified law; native diagnostic NaN sentinels are digested without treating them
as learned/data tensors. Data itself must be finite. All non-FIT structure and
bytes, original240 subset, canonical960 row tails/prompts, time0 reuse, scale,
mixed guard64, text/masks, and original saved config/encoder/state are checked.

The result is `report.json` plus `completion.json` in an exclusive new folder.
Exit0 means numerical integrity qualification PASS; exit1 means a completed
integrity FAIL; exit2 means error/INCOMPLETE. `scientific_status` stays null.
The whole CPU120 clock includes imports, all hashes, cleanup and final writes;
late errors downgrade both receipts. A root external CPU120 wrapper must bind
the descriptor/source/inputs and preserve the first attempt.

No exact FIT/TEST packed-row or latent overlap, and disjoint declared seeds,
establish these specific exclusion checks. They do not prove statistical
independence. Seed/path trajectory realization, actual1080 teacher callbacks,
pure-teacher arithmetic and complete native/class/caller rollback remain
authenticated producer/source proofs. The reader does not reconstruct a model,
replay teacher outputs, call ParticleGAN, calculate quality scores, change an
optimizer/guard, or claim convergence or a Supra win.

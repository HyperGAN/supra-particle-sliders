# Fresh native teacher caches

The historical no-override route retains its exact `SUPRA_SCALE` calibration and representative teacher checks. A newly captured cache from current native response code uses an explicit fresh source cohort and these three session options together:

```json
{
  "response_scale": 0.123,
  "cache_receipt": "/absolute/path/current-cache.json",
  "cache_receipt_sha256": "<sha256 of that JSON file>"
}
```

The number above is illustrative. Compute it from the actual captured training cache using `SupraCampaignRows.response_scale()`; it is never an accepted replacement for that measurement. Preserve the original latent/time/prompt/role coordinates in a new cache file, capture frozen neutral/positive responses through the current `SupraRuntimeResponseAdapter`, preserve all edit/preservation coverage and the exact nineteen admitted frozen contexts, and keep the historical cache untouched.

After creating `SupraCampaignRows` with that new file's real SHA, use `supra.fresh_cache.receipt_fields(options, rows)` to create the receipt object, write it to a new JSON file, and supply that file's real SHA in session options. The exact receipt schema is `supra-native-pg094-cache-v1`. It binds actual cache/context/context-receipt SHA values, native config digest, expected frozen-base SHA, original runtime/context source identities, complete executed product-module hashes, exact shared core receipt and response-adapter module source/origin, actual measured scale, edit/preservation counts, and the unchanged parity tolerances.

A receipt describes inputs; its text cannot attest native parity. Session admission re-reads the data, recomputes all receipt fields and the scale, verifies actual runtime/base/context identities, then compares every distinct cached native teacher row before the first update. Sampling repeats are deduplicated, so a 240-edit/30-preservation cache checks all 270 underlying rows. A self-consistent receipt for a corrupted later teacher row still fails. Absolute/relative teacher tolerances are unchanged.

Cache receipt SHA and resolved scale bind the new game's identity and full-state resume. Changing executed source, cache bytes, scale, contexts, or frozen base requires a fresh receipt/cohort and any required native qualification. This path does not requalify or relabel historical MSE weights, old caches, or the separately archived HNDL profile.

# Finished-output rate guidance: Supra Final Boss

Use the existing **G0.036 RAW128** Final Boss checkpoint for this measured
native256 FP32/rank16/batch4/horizon128 setup, at fixed **D0.018** with no particle
bank. The [portable record](finished-output-rate.json) preserves the exact values.
The rate won the two-nominee matched RAW64 comparison: **0.01690487563610077**
versus G0.072 **0.014670006930828094**. Do not pool this budget with earlier RAW32
scores. The same selected direction's retained RAW128 has a separate fresh
four-case mean **0.00974137553324302**. Both negative fresh gains remain included;
this completed fresh result is not a requirement for another confirmation.

## Current selection method

Use one scalar:

```text
case_gain = intended_effect(ON) - intended_effect(exact matched OFF)
rate_score = arithmetic_mean(case_gain over the declared cases)
```

Use the highest existing score within the same concept, native profile, training
budget and horizon. Retain negative gains, native zeros and exact ties; technical
failures stay missing. Teacher/student SSE is diagnostic only. Preservation,
beauty, confidence and completion never select or break a tie. Preserve each
completed primary comparison and its separate fresh result; do not pool them.
No additional confirmation, repeated search or worse-arm completion is required.
Reuse the original RAW checkpoints, proofs, teachers, outputs and first scores.
A full training state resumes only its exact unfinished prefix; a RAW inference
export is not a training checkpoint. Do not change initialization or relabel an
old packet. Official scalar optimizer statistics stay unchanged.

## Apply the actual rate options

The existing [Supra session validator](../supra/lifecycle.py) accepts these keys
inside the complete matching session's `options`:

```json
{"lr": 0.036, "d_lr_mult": 0.5, "ema_decay": null}
```

`lr` is G; actual D is `lr * d_lr_mult`. Supra is adapter-only: omit
`prior_lr_mult`. Use RAW current weights throughout; `ema_decay=null` is explicit
no model-weight EMA. [The product entry](../supra/train.py) forwards the complete
configuration to the common training dispatcher; it has no separate LR parser.
This fragment supplies no model/cache/contexts/source bindings and does not
retarget a checkpoint. The source-bound native256 research backend and the
public `supra` backend retain their own geometry and contracts. Keep the measured
sampler, conditioning, precision and original identities; copying the rates into
another backend is not reproduction of this effect result.

Changing G while keeping its multiplier fixed also changes D. A G-only
comparison fixes absolute D and checks actual optimizer groups. This finished
G selection does not assert that D was jointly optimized. [Historical PG0.9.5
response-SSE measurements](learning-rates-pg095.md) and [PG0.9.4 measurements](learning-rates.md)
keep every original value; they are diagnostic history, not the current selector.

## Availability and reproduction

The JSON record publishes rate options, scope, all selected per-case gains,
candidate means and hashes of the retained score records. It is a guidance
record, **not a complete training request or downloadable checkpoint**. Private
campaign paths are not portable links. The selected RAW, original conditioning,
full native configuration, frozen instrument and output bank are not bundled
here; the existing released adapter weights keep their separate provenance.
Thus the public files let a reader reproduce the arithmetic from the recorded
gains, but do not promise image regeneration from this repository alone.
For example, from the repository root:

```sh
python - <<'PYCODE'
import json
from math import fsum
from pathlib import Path
record = json.loads(Path("docs/finished-output-rate.json").read_text())
for name in ("primary", "separate_fresh_result"):
    gains = list(record[name]["per_case_gains"].values())
    print(name, "mean from gains:", fsum(gains) / len(gains),
          "original recorded mean:", record[name]["mean_gain"])
PYCODE
```

This checks the published arithmetic, with ordinary floating-point rounding;
it is not a new rating or a reproduction of the private instrument/images.

With access to the exact saved cohort, use its existing RAW and matched outputs;
do not repeat training or rating to use this result. For genuinely missing
measurements, freeze the intended-effect definition, instrument, cases and native
settings before the first observation, reuse matched OFFs, then retain every case
at equal weight. The score is model-instrument evidence, not human/statistical
validation, a universal optimum or a new default.

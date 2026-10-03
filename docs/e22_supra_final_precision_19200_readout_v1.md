# Final precision continuation: qualified scientific failure

The fixed paired continuation completed with scientific FAIL, and independent
saved-state verification qualified PASS. Neither arm beat the original on any
of the three aggregate metrics; FP32 training also trailed the BF16 control
under the same FP32 evaluation.

Each row below covers the same physical **TEST240** editing task: six sources,
40 contexts each, the same live teacher and frozen D1856/D6400 panels. Lower is
better. Both continuation arms use **FP32-final evaluation** here, which separates
training arithmetic from the serving arithmetic change.12800 is their shared
authenticated starting point, not another newly trained arm.

| Point | Training arithmetic | TEST RMSE | D1856 | D6400 |
|---|---|---:|---:|---:|
| Original selected28000 reference | Historical ordinary LoRA; FP32-final evaluation |0.063360985|0.864760392|0.874255461|
| Shared12800 | Inherited native BF16 point; FP32-final evaluation |0.069659898|0.891451932|0.902655429|
|16000 diagnostic | BF16 final |0.068725071|0.886910063|0.897472707|
|16000 diagnostic | FP32 final |0.068720825|0.886813845|0.897645186|
|19200 fixed terminal | BF16 final |0.067914335|0.883909163|0.894555875|
|19200 fixed terminal | FP32 final |0.068108527|0.884305793|0.894960727|

The permanent original-arithmetic target remains RMSE **0.063777300**,
D1856 **0.867055504**, D6400 **0.876714643**. Both terminal arms also miss that
target on all three scores. The small mixed16000 arithmetic differences were
diagnostic only; they did not select an arm, alter a rate or stop the run.
Terminal FP32-trained results are worse than BF16-trained results on all three
common-FP32 aggregates. Both nevertheless improved from the shared12800 point.

Particles remain useful in the independently verified offline zero-code captures. This
table compares each live arm with code removal under **its own training
arithmetic**, matching the actual ablation source law; BF16 zero-code is not
silently treated as a common-FP32 counterfactual.

| Point / own arithmetic | Live RMSE | Zero-code RMSE | Zero-code D1856 | Zero-code D6400 |
|---|---:|---:|---:|---:|
| BF16-trained16000 / BF16 |0.069094682|0.117517032|1.157035701|1.159206705|
| FP32-trained16000 / FP32 |0.068720825|0.117283061|1.154950273|1.157084086|
| BF16-trained19200 / BF16 |0.068298346|0.117383136|1.156533898|1.159067519|
| FP32-trained19200 / FP32 |0.068108527|0.117299505|1.155441046|1.157585799|

The producer records changed bank/router tensors in both arms,71 finite positive
code norms per arm, trainable H/b, and6400 live/dense bank-gradient updates in
each arm. `particles_retained=true`. All6400 paired DV12 stream states matched;
there were **zero accepted structural moves**. The bank and router changed continuously. Zero accepted moves alone does
not establish a cause for the remaining gap.

The separately completed saved guard inspection found the same ordered64
active contexts in data and original12800/both16000 checkpoints: **57 editing
and7 holds**, exact context/paired-target correspondence, strict zero permitted
per-context feature harm, and output-error guard disabled. Thus editing-only
describes the gradient update pool; retained structural guard constraints still
include holds. This does not identify which context rejected any proposal or
prove that those constraints caused slow convergence.

The unchanged protocol resumed both arms exactly from12800 to19200, with6400
additional public native editing updates per arm,12800 updates total,25600
optimizer steps and zero preservation updates. Teacher arithmetic, native
game/noise/controller laws,71 routed sites and within-site shared Up architecture remained
fixed. Offline TEST scores never supplied an objective, guard, rate, stopping
rule or selection. Whole external cost was **10442.849117s /14400s**; final
producer charges were BF16 **5199.953778s** and FP32 **5235.367577s** /7200 each.
All77 external input/source pins remained unchanged. Independent saved verification completed in **322.589837s /1200s**,
with 192,984 checks, all149 bindings stable, all12 raw evaluations checked,
all32 checkpoints authenticated and two terminal checkpoints loaded. Summary
differences were exactly zero; maximum F32 reduction difference was
7.450580596923828e-09, within the frozen tolerance. No models,
forwards, API calls or updates were performed by the verifier.

Original selected28000 used ordinary LoRA with a supervised MSE/AdamW objective
and a mixed editing/preservation schedule. Its fixed score is a quality target,
not a matched update-budget or optimizer-superiority comparison. This native
continuation does not establish a precision remedy, convergence win or promotion.

Local producer, independent verifier and reference evidence:

- [Frozen protocol](/ml2/hypergan/supra-final-precision-continuation/docs/e22_supra_final_precision_continuation_protocol.json), SHA `94503f27f178797d0676e202ba41d73e0be65b329c0a1d815d707c1a9a24835f`.
- [Completed producer report](/ml2/hypergan/supra-final-precision-continuation/outputs/e22-supra-final-precision-19200-v1/report.json), SHA `fcd933a638aff1e06d99ab94e126828cf34c0a8d0f0c310bcd37e9462c4093df`.
- [Producer completion](/ml2/hypergan/supra-final-precision-continuation/outputs/e22-supra-final-precision-19200-v1/completion.json), SHA `dfe424ad12ba75c5778d3a81c445f6821e9db168c681b72e4d9d054423f37a6d`.
- [External watchdog receipt](/ml2/hypergan/supra-final-precision-continuation/outputs/e22-supra-final-precision-19200-v1.external.json), SHA `39181bbd0ae6352d55a2d05cb581499becd1d218a646e80439efc09a64a48a5c`.
- [Qualified12800/original28000 scorer](/ml2/hypergan/ParticleGAN-supra-native-adam-direction-develop/runs/full12800-selected28000-student-final-precision-v2/report.json), SHA `86f919102dca4330c2569886a00c93da0eb008a14c3225317fb2b4c31c8d15f7`; its [independent qualifier](/ml2/hypergan/ParticleGAN-supra-native-adam-direction-develop/docs/e22_routed_full_student_final_precision_v2_independent_review.json) is the existing baseline authority.
- [Saved guard inspection](/ml2/hypergan/supra-final-precision-continuation/outputs/e22-supra-guard-pool-inspection-v1/report.json), SHA `68cb913ccd775be290b9c43a1b594498c2ae10c09a7db36f160d5408e5a799b0`.
- [Independent saved verification](/ml2/hypergan/supra-final-precision-continuation/docs/e22_supra_final_precision_continuation_independent_review_v1.json), SHA `74ebbeed9f1f15d40d7f776f07ec8998b56062abbbe3e90303303f462f54f247`.
- [CPU1200 completion](/ml2/hypergan/supra-final-precision-continuation/outputs/e22-supra-final-precision-19200-v1-saved-review-v1/completion.json), under [the frozen149-binding plan](/tmp/supra-final-precision-19200-v1-saved-review-root-plan.json), SHA `c7e6ba3c8e6d1e836c1e1f2f289c1516b305fa43743121ac18ae93c618d64f1e`.

The next diagnostic is the separately frozen actual FIT clean/noisy game-gradient
pair at the original12800 point. It measures existing game directions without
training updates or output-accuracy feedback. Removing preservation guards
remains an unverified application hypothesis requiring an explicit checkpoint
and reservoir-law transition.

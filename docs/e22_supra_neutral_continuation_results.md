The fixed Supra continuation is independently qualified. Neutral particles improve on the matched particle control and the historical ordinary LoRA 6,400 target at both 9,600 and 12,800 editing updates. They still trail ordinary LoRA 12,800 and the selected original 28,000 on all six subjects. The convergence gap remains; this extra training result does not establish faster convergence or a top-model win.

All scores below use the same full 240 held-out editing contexts, private Gaussian panels, and two frozen critics. Lower is better. RMSE is an offline diagnostic and never enters the optimizer, structural decisions, selection, or stopping.

| Model and horizon | D1856 | D6400 | Diagnostic RMSE |
|---|---:|---:|---:|
| Ordinary LoRA 6,400 | 0.912806 | 0.924385 | 0.073333 |
| Ordinary LoRA 12,800 | 0.884893 | 0.895550 | 0.067283 |
| Original selected 28,000 | 0.867056 | 0.876715 | 0.063777 |
| Particle control 6,400 | 0.928838 | 0.940390 | 0.077302 |
| Neutral particles 6,400 | 0.919552 | 0.930878 | 0.075577 |
| Particle control 9,600 | 0.924515 | 0.936297 | 0.075607 |
| Neutral particles 9,600 | 0.905596 | 0.917945 | 0.072220 |
| Particle control 12,800 | 0.908859 | 0.920710 | 0.072857 |
| Neutral particles 12,800 | 0.893615 | 0.904961 | 0.070021 |

Each subject has 40 held-out contexts. Cells contain the D1856 / D6400 pair; the accompanying [compact JSON](e22_supra_neutral_continuation_results.json) includes every subject's diagnostic RMSE and paired comparisons across all 240 contexts.

| Subject ID | Ordinary 6,400 | Ordinary 12,800 | Original selected 28,000 | Control 9,600 | Neutral 9,600 | Control 12,800 | Neutral 12,800 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4 | 0.979023 / 0.995304 | 0.945749 / 0.961395 | 0.918935 / 0.933282 | 0.995568 / 1.011157 | 0.975509 / 0.992957 | 0.971511 / 0.988672 | 0.956456 / 0.973480 |
| 5 | 0.908852 / 0.919364 | 0.880750 / 0.890231 | 0.860277 / 0.869167 | 0.918627 / 0.930092 | 0.898742 / 0.910669 | 0.910194 / 0.922675 | 0.891848 / 0.903383 |
| 15 | 0.848605 / 0.857549 | 0.832180 / 0.840234 | 0.822050 / 0.829158 | 0.858098 / 0.869474 | 0.839641 / 0.849397 | 0.844270 / 0.852948 | 0.833758 / 0.841908 |
| 16 | 0.899987 / 0.906822 | 0.871895 / 0.877974 | 0.857526 / 0.863555 | 0.906942 / 0.913515 | 0.889290 / 0.897210 | 0.893412 / 0.900422 | 0.876907 / 0.883865 |
| 17 | 0.952929 / 0.968169 | 0.920457 / 0.934744 | 0.897192 / 0.909572 | 0.969760 / 0.984696 | 0.948534 / 0.963894 | 0.945036 / 0.959070 | 0.929006 / 0.941917 |
| 18 | 0.887443 / 0.899103 | 0.858324 / 0.868721 | 0.846352 / 0.855554 | 0.898094 / 0.908845 | 0.881858 / 0.893540 | 0.888733 / 0.900473 | 0.873717 / 0.885215 |

At 9,600, neutral improves 160 of 240 contexts versus ordinary 6,400 under each critic. At 12,800 it improves 205 / 202 contexts versus ordinary 6,400, but only 53 / 57 versus ordinary 12,800. It improves 209 / 209 contexts versus the matched 12,800 particle control. All six subject averages favor neutral over that control at both endpoints.

Both arms retain a shared 128×4 particle bank and 71 routing sites. All 6,400 continuation updates per arm produce gradients on all 128 bank rows; the bank and router change, H/b remain trainable, and all site code norms remain nonzero. Accepted structural moves: 0. At 12,800, removing neutral's learned codes worsens the full games to 1.155845 / 1.158576; retaining only mass worsens them to 1.109145 / 1.117161. The particle contribution is material.

Qualification passed 85,687 checks over 36 full states (34 new plus two parent 6,400 inputs), both traces of 6,400 updates, four FAST exports, and all endpoint/reference/ablation records. The eight actual GPU recovery updates proved exact 6,400→6,402 replay, followed by an exact authoritative 6,402 match. The CPU reviewer made zero native updates and zero model-forward calls; GPU execution remains an explicitly bound runtime witness. Native source stays `6ec7e5788e14ea15ddc3e16ac71110458108b6a6`. The public initializer runs only before strict native restoration, never on trained parameters.

The new GPU runner spent 7,925.833 s, charged 3,961.265 s to control and 3,964.568 s to neutral (limits 7,200 s each / 14,400 s total). Actual outer GPU launch through exit was 7,931.954 s. External CPU qualification spent 411.479 s through exit (inner review 405.945 s; limit 1,200 s). Conservative combined cost was 8,343.432 s, below 15,600 s. The earlier 6,400 campaign, source preparation, scalar observations and separate 65.425 s historical rescore are separate costs. A caption toy shared GPU 0 during this continuation, precluding speed comparisons.

Native particles train on editing only: 9,600 / 12,800 editing updates and zero preservation. Historical ordinary 6,400 used 5,120 editing + 1,280 preservation; ordinary 12,800 used 10,240 + 2,560; selected 28,000 used 22,400 + 5,600. Those MSE/AdamW runs differ in objective, initialization and architecture. The 6,400 reference remains the frozen primary target, 12,800 remains the secondary budget context, and selected 28,000 is a separate historical best-model comparison. Neither endpoint was selected or omitted. Both learned critics belong to one architectural family, so these scores do not establish SOTA.

The JSON binds the actual source/card/native/input-qualified review and completion SHAs, execution/recovery/export witnesses, final checkpoint/native digests, and separate selected 28,000 report. Full raw data stay in ignored outputs. Held training sources, native code and the authorized protocol were unchanged.

# Full Supra: paired native final-precision continuations

The original Supra LoRA remains ahead. At the authenticated particle step12800,
changing only the student final projection to FP32 improves held-out RMSE from
0.0700214950 to0.0696598979; the original selected LoRA scores0.0637773004 in its
historical arithmetic and0.0633609846 with its final projection in FP32. Those
zero-update scores establish a small serving effect, not a training benefit.

This campaign continues the exact same trained editing-only particle checkpoint
in two arms, BF16 and FP32_final, for6400 additional native updates each. The
only changed computation is the student final.linear arithmetic. Stored frozen
head weights are FP32 in both arms. Teacher arithmetic, shared-Up particles,
71 sequential routing sites,128x4 bank, routers, native optimizers, learned-noise
law, schedules, KA2, and structural feature controls stay as restored. All fresh
owners use public particlegan.init.initialize_ before optimizer/EMA creation;
trained restore never reinitializes a learned owner.

Training uses the original FIT CPU7 B4 draw law, private paired CUDA43 noise,
and native game-only updates. There are no preservation updates, output-MSE
objectives, output-error structural guards, score-driven stopping, or score-driven
checkpoint choices. Per-arm global CPU/CUDA RNG is owned and recovered exactly.
Different adaptive native controllers may consume different DV12 draws; that
is reported without forcing their stream positions to agree.

Fixed endpoints16000 and19200 evaluate all240 retained TEST contexts under BOTH
student final arithmetic modes and with an offline zero-code ablation in each
arm's training arithmetic. Original CPU72 four-panel paired scores from frozen
D1856/D6400 accompany diagnostic RMSE. Common-FP32 evaluation of both trained
arms distinguishes training effects from serving arithmetic. Terminal19200 is
the sole decision point; intermediate scores are observation only.

A top-score PASS requires the FP32-trained terminal arm to improve all THREE
RMSE,D1856,D6400 scores strictly over BOTH original reference triplets and retain
useful particles: changed bank/router, all71 finite positive C norms, trainable
H/b, and zero-code RMSE at least1.001 times live RMSE. Common-FP32 improvement
against the BF16-trained control is a separate descriptive training-benefit
result, not an additional top-score gate. Global code benefit does not imply
per-source no harm or accepted structural moves.

The original28000 reference was selected by validation after22400 editing and
5600 preservation updates with MSE/AdamW. This is a score target comparison,
not a matched experiment proving optimizer superiority over ordinary LoRA.

The new tiny public-API software qualification passed10 cases with3 native
updates. Actual full-model qualification then passed6 native updates/12 optimizer
steps in141.49234 seconds of an external300-second cap: both initial score
anchors exact, own-arm12801->12802 recovery exact, FAST/EMA precision-tagged
exports exact at strengths0/1, wrong-precision recovery rejected, and rollback
exact. These are engineering prerequisites, not convergence evidence.

One root GPU0 process gets14400 seconds total and7200 charged seconds per arm,
including preparation, input hashes, training, fixed400-step checkpoint saves,
endpoint scoring, FAST exports, cleanup and final qualification. Shared costs
are split equally. A separately frozen independent CPU saved-result checker
gets1200 seconds and invokes no models, trainers, optimizers or ParticleGAN API.
Any execution/schema/nonfinite/source/matching/budget error is INCOMPLETE;
complete numeric gate misses are FAIL. Failed evidence is retained unchanged.

Raw weights, observations and traces remain ignored local artifacts under
outputs/e22-supra-final-precision-19200-v1. The read-only dashboard listens on
0.0.0.0:8786 at http://pop-os:8786; it plots native losses live, includes historical
reference scores and leaves missing endpoints pending instead of zero.

The exact sources, inputs, precision law, horizons, thresholds, external clock,
receipts and independent-checker identity are bound in the protocol JSON and
preparation record before the first long training update. The run has not yet
been launched when this document is prepared.

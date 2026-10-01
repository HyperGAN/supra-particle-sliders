# Final V2 critic signal and bounded-capacity test

The qualified particle V2 model at step 6,400 still has a restorative native
game gradient on the selected contexts. There is no sign reversal or detached
critic signal in this probe. Adding a bounded learning route around the second
critic tanh did not repair its weak response to large edit residual patches.
This is a negative capacity diagnostic, not a full-model convergence experiment.

The source is ParticleGAN `cabe2084284db923d525918cbf3e18de6f20faac` and the
checkpoint is `outputs/e22-particle-v2-6400/final.pt`. The read-only GPU capture
uses the next saved edit and preservation minibatches, four contexts each, plus
the first four test contexts. It reconstructs the native BF16 host, FP32 particle
branch and CFG3, then records clean and four private per-site DV12 residuals.
These twelve contexts are a diagnostic sample, not the complete evaluation set.

The capture verifies frozen model, controller and global RNG immutability and
the checkpoint, export, cached data and archived ParticleGAN source hashes.
Results are in `outputs/e22-particle-final-critic-causal/`.

## Saved critic at step 6,400

Four fixed private CPU Gaussian panels measure the saved critic's native RpGAN
output gradient on the native captured residuals. Positive residual cosine
means output-space game descent reduces the paired residual. It does not
establish that the actual Adam parameter step improves another minibatch.

| Selected pool | Raw residual / G-gradient cosine | Largest 10% residual power | Their G-gradient power | Their gain / lower-half gain |
| --- | ---: | ---: | ---: | ---: |
| Edit fit | 0.7830 | 42.79% | 19.77% | 0.483 |
| Preservation | 0.7954 | 26.83% | 40.02% | 1.633 |
| Edit test | 0.7845 | 52.42% | 14.31% | 0.312 |

This confirms the earlier edit-tail asymmetry persists at the final checkpoint.
Preservation has a different allocation. Residual bins and cosines are read-only
diagnostics; they are never loss weights or optimizer/structural criteria.

## Initially equivalent bounded extension

The experimental critic changes the second feature activation to
`tanh(a) + tanh(gain) * tanh(a/4)`, with sixteen gains initialized to zero.
All features stay within absolute bound 2. Initial scores, structural features,
input derivatives, old parameter derivatives and EMA features exactly equal
the native critic. Old Adam moments, learning rate, KA2 record and EMA values
are retained. Only the newly appended gain lacks prior Adam moments.

The isolated state has an explicit `bounded_second_feature_bypass_v1` tag and
dedicated restore; an ordinary native checkpoint cannot silently reinterpret it.
The public ParticleGAN source and production Supra architecture are unchanged.

Two predeclared private critic copies train for exactly 128 D-only updates on
captured DV12 residuals. They share all paired Gaussian panels and the native
four-edit/one-preservation phase, retaining preservation payoff weight 0.1.
Only native RpGAN plus KA2 trains the critic. KA2 still sees the full changed
input derivative, including the bounded bypass, with unchanged patch units,
real R1, caps, anchor and spike guard. The application controller, output sigma
and G/router/bank are frozen, and test contexts never enter an update.

This is a critic response stress test on repeated captured minibatches. It is
not a continuation of the application policy: captured DV12 residuals are
cycled, private CPU panels differ from native CUDA training draws, and the
application learning-rate/data/game controller does not advance. Once active,
the extra tensor also participates in KA2's tensorwise surprise statistic.
Therefore this tests the bounded extension with its native critic controls,
not the feature law independently of optimizer/controller topology.

| Final critic after 128 D-only updates | Edit-test raw cosine | Edit-test tail gradient power | Tail / lower-half gain |
| --- | ---: | ---: | ---: |
| Native capacity | 0.7654 | 13.97% | 0.302 |
| Bounded additional capacity | 0.7642 | 13.91% | 0.301 |

The extension does not restore the missing tail response. All sixteen gains
end negative, between -0.0154 and -0.0994, reducing the added broad feature
term. At the initial fit residuals, D-game and KA2 gain-gradient norms are
0.013873 and 0.013996 with cosine -0.98245. Their sum initially pushes eight
gains negative. This identifies opposition from the native regularizer along
this proposed capacity direction; it is not proof that KA2 is incorrect or
that weakening its safeguards would improve full-model learning.

Own-critic payoff values are recorded as geometry but are not a shared judge
for comparing different critics. No full-model improvement or original-LoRA
win is claimed. The current recipe is not changed by this negative trial.

## Qualification and reproduction

The independent reviewer passed 28 checks, including exact 128-update full
critic/Adam/KA2/EMA state replay for both arms, complete trace values and clocks,
source/input digests, the bounded feature law and a separate native KA2 value
and full parameter-gradient decomposition including the new gain. Three
focused CPU tests pass. Exact replay preserves native graph construction order;
algebraically equivalent reversed payoff construction changed shared-parameter
float32 gradient accumulation, so the reviewer uses native real-first order
without relaxing tolerances.

```bash
CUDA_VISIBLE_DEVICES=1 /ml2/ntc-image-studio/.venv-anima/bin/python \
  scripts/capture_e22_supra_final_critic_payload.py
/ml2/ntc-image-studio/.venv-anima/bin/python \
  scripts/diagnose_e22_supra_final_critic_capacity.py
/ml2/ntc-image-studio/.venv-anima/bin/python \
  scripts/review_e22_supra_final_critic_capacity.py
```

Use fresh `--output` locations when repeating the first two commands. Evidence:
`capacity/receipt.json`, `capacity/independent-review.json`, both per-update
JSONL files and explicitly tagged isolated critic checkpoints under
`outputs/e22-particle-final-critic-causal/`.

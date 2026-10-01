# Native E22 game-control audit

The 6,400-update full Supra run follows ParticleGAN `f459cb6d`'s control
formulation. Its very small critic stationarity scale does not explain the
frozen KA2 critic average through a learning-rate input to KA2. The two controls
have separate inputs.

The evidence is in
`outputs/e22-convergence-gap/control-audit.json`, produced by
`scripts/audit_e22_game_controls.py` using the archived historical source and
CPU-mapped checkpoints. The audit reads small critic tensors and controller
state, performs no training or GPU work, and uses no output-accuracy metric.

The native learning-rate assignment in `particlegan/policy.py:604` is:

```python
critic_scale = max(critic_tester.s, 0.75 * table_tester.s)
critic_lr = initial_critic_lr * critic_scale * controller.critic_scale()
```

The table scale stays at one. Once the critic tester falls below 0.75, later
stationary decisions cannot lower its applied LR. The tester still observes
the actual applied/base LR through `_settle_observe`, so its intrinsic clock
continues running. Its repeated decisions halve its internal proposal without
changing the dominant floor. This is explicit native precedence, not a caller
passing the proposed LR into the actual optimizer.

| Update | Critic tester scale | Applied critic LR | KA2 alpha | EMA updates | EMA skips |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1,600 | 0.25 | 0.002048 | 1.48e-8 | 467 | 334 |
| 2,400 | 1.91e-6 | 0.002457 | 3.28e-18 | 673 | 928 |
| 3,200 | 1.82e-12 | 0.002600 | 3.68e-115 | 704 | 1,697 |
| 4,000 | 3.47e-18 | 0.002715 | 0 | 704 | 2,497 |
| 4,800 | 8.27e-25 | 0.002808 | 0 | 704 | 3,297 |
| 5,600 | 1.97e-31 | 0.002879 | 0 | 704 | 4,097 |
| 6,400 | 9.40e-38 | 0.002937 | 0 | 704 | 4,897 |

KA2's `advance_blend` in `particlegan/ka2.py:105` compares the median of the
last 24 critic Adam gradient-surprise observations to its early reference.
The target alpha is zero when that ratio is at most one. Alpha approaches its
target and is multiplied by DV12's game trust. `record_step` uses
`decay = 1 - alpha * (1 - anchor_min_decay)` to update the critic EMA. Neither
calculation consumes `tester.s`, `lr_last`, or `lr_max`; the last two fields
record the actual LR after the EMA operation.

The saved EMA tensors are identical at every checkpoint from 3,200 through
6,400, while the current critic tensors change at each checkpoint. Thus the
reference really is fixed; the skip counters are not merely a logging effect.
At 6,400, the saved KA2 short-window ratio is 0.5555, alpha is zero, and the
anchor weight is one. The blind DV12 controller has no data-drift evidence, so
its rule forces the anchor weight on. No R1 event releases it.

The audit includes two native control witnesses:

* A native `begin_step`, with the same payoff damping and table scale one,
  applies exactly the same critic LR with critic proposals 0.25 and 9.40e-38.
  Reducing only the table scale to 0.25 reduces that applied LR fourfold.
  No optimizer step is performed.
* Changing only KA2's recorded LR from the tiny proposal to the saved applied
  LR leaves its next alpha, decay, surprise ratio and blend weight identical.
  A declared hypothetical starting alpha of 0.4 gives alpha 0.2 and decay 0.98
  under both LRs, ruling out an equality caused solely by zero alpha.

The witness's next LR is 0.002945, slightly different from the saved last LR
0.002937: it uses the final payoff damping, whereas the last optimizer update
used the previous update's damping. This is the normal controller timing.

R1 also behaves as declared. Its final ratio is 2.1949 but its last calm
update is 444 detector updates earlier and its streak is zero. A slow rise
does not meet the abrupt-rise rule. A ratio above two alone does not fire R1.

These facts expose a control limitation worth measuring: the critic can keep
taking substantial steps against a fixed historical anchor while its own
stationarity proposal is ignored by the table floor. They do not prove that
the anchor causes the observed convergence gap. Freezing an anchor is an
intentional KA2 action, and a substantial applied critic LR is an intentional
table-linked floor.

The isolating game diagnostic is a matched replay from the same checkpoint,
context, paired noise, DV12 stream and optimizer state. Compare native KA2
with a copy whose only change is `anchor_weight=0`, retaining the A term,
both B gradient caps, all LRs and all optimizer memory. Report the
adversarial, A, B-cap and anchor-proximal critic gradient norms; the cosine of
anchor-proximal and adversarial gradients; the native critic update; and the
generator payoff and game gradient after it. This can establish whether the
frozen anchor opposes the current game. Persistent repair would require a
predeclared critic-adaptation horizon and paired game evidence. Output errors
must remain reporting only.

## Matched native critic-step witness

`scripts/diagnose_e22_supra_anchor_step.py` reconstructs the saved final critic,
EMA and Adam state, then uses the native penalty and native guarded Adam step.
The recorded host residuals come from native CUDA BF16 forwards and FP32
routing. The critic diagnostic runs in CPU FP32. Both arms receive the same
source/time contexts, saved DV12 residual draws and exact next-6,401 CUDA
paired Gaussian bases. The preservation batch uses those same bases as a
counterfactual; it does not claim to replay update 6,405's noise. The sole
override is the native penalty's `anchor_weight=0`.

The receipt is `outputs/e22-convergence-gap/native-anchor-step.json`. The
proximal gradient is the difference between the two native penalty gradients,
so the diagnostic does not substitute another implementation of KA2. The
remaining penalty gradient contains native A and B caps together.

| Fixed batch | Weighted game gradient norm | Cap gradient norm | Proximal gradient norm | Proximal/game cosine | Native game loss decrease | Without-anchor game loss decrease |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Edit | 0.15315 | 0.14585 | 0.02996 | -0.4766 | 0.0003515 | 0.0019187 |
| Preservation | 0.01369 | 0.00857 | 0.00350 | -0.8028 | 0.0000418 | 0.0000852 |

The frozen anchor locally opposes the adversarial critic gradient on both
batches. Removing it changes the native critic step and gives 5.46 times the
edit-batch adversarial decrease and 2.04 times the preservation-batch decrease.
No spike guard clips either step. Thus this is a causal witness of local
proximal restriction, rather than an inference from a frozen-average counter.
Such restriction is part of the intended regularizer; this result alone does
not show that removing it improves the coupled game or explains the training
gap.

The remaining caps already strongly oppose the adversarial gradient: their
cosines are -0.9787 for edits and -0.9439 for preservation. Their norms are
95.2% and 62.6% of the corresponding weighted game-gradient norms. The full
penalty norms are 103.7% and 84.2% of those norms. The local game is balancing
an adversarial gradient against substantial native regularization.

Preservation multiplies the adversarial losses by 0.1 while retaining the
penalty coefficient. On that batch its unscaled adversarial gradient norm is
0.13688, and its weighted norm is 0.01369. Consequently the unchanged full
penalty is 8.42% of the unscaled game-gradient norm and 84.2% of the weighted
one. This tenfold relative strengthening follows from the application's
declared preservation weighting; it is not an LR or EMA-alpha calculation.

After the matched critic step, the two arms' generator game gradients with
respect to student velocity differ by 6.79% on edits and 1.72% on preservation.
Their directions remain very close, with cosines 0.99774 and 0.99994. These
are native adversarial gradients, not an output-error objective. The diagnostic
does not reconstruct the generator's parameter Jacobian or apply a generator
update. It establishes a local critic restriction and a small change to the
downstream game signal; persistent repair remains unestablished.

## Generator stationarity and Adam memory

The generator's first and only stationary decision occurs at update 5,880.
Its scale remains one and LR remains 5e-5 through that update; its LR is
2.5e-5 for updates 5,881 through 6,400. Early generator LR settling therefore
does not account for this run's earlier learning gap.

The saved generator owner uses AMSGrad, beta1 zero and beta2 0.999. The native
generator optimizer delegates its denominator calculation to PyTorch Adam;
its latent/direct helpers do not replace that calculation. A setting of
`amsgrad=False` on a restored generator-owner group makes Adam use the decayed
second moment while retaining the first moment, second moment, step, LR and
stored maximum. The stored maximum is then ignored. The critic optimizer and
its guard/KA2 state are separate and need not change.

`outputs/e22-convergence-gap/generator-memory-audit.json` measures denominator
memory from saved checkpoints. Across generator coordinates, the ratio of the
AMSGrad denominator to the decayed-moment denominator grows from median
1.326 at 1,600 to 1.795 at 3,200, 2.296 at 5,600 and 2.589 at 6,400. At 6,400
its 90th percentile is 4.678 and 99th percentile 8.756; 68.7% of coordinates
exceed two. Every one of the 71 sites has an inflated median, ranging from
1.558 to 3.554.

Those coordinate ratios alone do not measure actual gradient-weighted
attenuation. Beta1 zero allows a stronger saved-state witness: `exp_avg` is
the applied last gradient. With that same gradient, saved moments and LR,
compare the native last-step formula to the formula with only the maximum
removed from its denominator. No new host forward, gradient or update is
needed. The CPU float64 reconstruction is saved in
`outputs/e22-convergence-gap/generator-step-memory-witness.json`.

| Owner role | Native last-step norm | Same-gradient decayed-denominator norm | Norm ratio | Direction cosine |
| --- | ---: | ---: | ---: | ---: |
| Generator branches | 0.0007208 | 0.0016899 | 2.344 | 0.8860 |
| Router | 0.0008375 | 0.0010416 | 1.244 | 0.9518 |
| Bank | 0.0010642 | 0.0063339 | 5.952 | 0.8821 |
| Learned noise | 0 | 0 | — | — |

The generator branches have no latent/direct numerator override. The last
bank update has all 128 gradient rows active, so its sparse A2 numerator
override is also inactive. All 71 generator sites have a larger counterfactual
step norm, by factors 1.482 through 5.217. This establishes denominator
attenuation of the applied gradient, not better learning from a different
denominator. A generator-owner AMSGrad ablation changes router, bank and noise
groups as well as the branches; any resulting improvement cannot be assigned
to branch memory alone. In particular its bank-step change is larger than
its branch-step change.

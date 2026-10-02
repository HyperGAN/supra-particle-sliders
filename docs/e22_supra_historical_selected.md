# Original best-model reference on the shared Supra task

The original ordinary LoRA run selected update 28,000 and stopped at 28,800.
The 6,400 and 12,800 snapshots used by the fixed particle studies are separate
horizon references; they are not the best original model. Ordinary LoRA adds
standard rank 16 down/up adapters without a particle bank or routing.

The separate zero-update observation scores both original 6,400 and selected 28,000
using the qualified particle study's identical 570 contexts, live BF16/CFG3
source-to-target pairing, B4 batches, four CPU72 Gaussian panels, and frozen
D1856/D6400. All original 6,400 records replay exactly. The selected file and
checkpoint 28,000 have identical learned tensors despite different export metadata.

| Full 240-context held-out editing set | D1856 game ↓ | D6400 game ↓ | RMSE diagnostic ↓ |
| --- | ---: | ---: | ---: |
| Original ordinary 6,400 | 0.91280650 | 0.92438515 | 0.07333330 |
| Original selected 28,000 | 0.86705550 | 0.87671464 | 0.06377730 |
| Qualified particle H/b-neutral 6,400 | 0.91955221 | 0.93087777 | 0.07557706 |

The best original model leads on both learned games for all six subjects.
It received 22,400 editing and 5,600 preservation updates under historical MSE/AdamW
with validation-based plateau control and selection. The particle model received
6,400 editing updates under the native learned game. These are best-model results
with different budgets, objectives and initialization; they do not isolate
ParticleGAN optimizer causality. Output RMSE remains reporting only.

The observation completed with exit 0 in 65.425 seconds within its separate 300-second
startup-through-final-write allowance. It passed 377 runtime checks and 2,316
record/reduction checks. Native optimization never runs. Sources, model/teacher/
critic ownership, modes, gradients, ordinary multipliers, data and global RNG remain
unchanged. Detailed captures and receipts remain in ignored
`outputs/e22-supra-historical-selected-28000`; compact results are in
[e22_supra_historical_selected_results.json](e22_supra_historical_selected_results.json).

The running 6,400→12,800 study keeps its predeclared 6,400 primary and 12,800 secondary
references. The dashboard separately labels the selected 28,000 best-model target;
its horizontal level is the observational 12-context probe, while all 240-context
scores appear below. No result changes optimization, structural guards, horizon,
or stopping. No timing superiority is inferred on the shared GPU.

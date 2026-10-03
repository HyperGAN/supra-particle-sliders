# Saved-point game directions: selected FIT helps, full TEST harms

The fixed original12800 clean/noisy gradient measurement completed and its
independent saved-array verification qualified PASS (262,211 checks). This is
an offline diagnostic with no convergence verdict or training intervention.

Positive accuracy-gradient dot game-gradient means descending the game gradient
locally improves physical residual MSE. Negative means locally worsening it.

| Accuracy context panel | Clean game direction | DV12 game direction |
|---|---:|---:|
| Selected FIT12 | +0.0009549267 | +0.0007073328 |
| Full TEST240 | -0.0003772554 | -0.0003757698 |

Both directions help the selected training contexts and harm the aggregate
held-out tangent. The paired noisy-minus-clean TEST projection is +0.0000014856,
with draw standard error 0.0000575250; its descriptive two-SE band crosses zero.
Thus disabling learned noise is not established as a remedy. Noise weakens FIT
alignment, while its shared-Up and router effects on TEST oppose each other.

The diagnostic selects two contexts per source at times 0.3 and 0.6. Actual
editing updates sample uniformly from all FIT240, so this does not establish a
training sampler bug. Training and held-out data use the same ten times;
the fixed split changes latent seeds. The selected FIT12 additionally couples
time with seed index and neutral/positive trajectory. The next observation must
separate those factors before attributing the result to time or latent transfer.

The measured field uses saved D and completed-point DV12 bandwidth. It differs
from the native post-D/refreshed-bandwidth generator graph and does not identify
an optimizer defect or a whole-run cause. The earlier actual displacement and
these raw gradients are distinct observations. Accuracy never enters training,
structural guards, rate selection, checkpoint selection or stopping.

Evidence:

- [Producer report](/ml2/hypergan/supra-final-precision-continuation/outputs/e22-supra-fit-game-gradient-pair-v1/report.json).
- [Independent numerical review](/ml2/hypergan/supra-final-precision-continuation/outputs/e22-supra-fit-game-gradient-pair-v1/independent-review.json), SHA256 `a55f7b882ed974684caf0046ecdc4395f9bc3f2000609846974a2bdb6309696d`.
- [CPU300 completion](/ml2/hypergan/supra-final-precision-continuation/outputs/e22-supra-fit-game-gradient-pair-v1-saved-review-v1/completion.json).
- [Long-run convergence comparison](/ml2/hypergan/supra-final-precision-continuation/docs/e22_supra_final_precision_19200_readout_v1.md): the original still leads; precision alone did not close the gap.

All executed sources, protocols and failed attempts remain unchanged. This
readout adds an interpretation of qualified observations; it does not regrade
the completed campaign or select a new training configuration.

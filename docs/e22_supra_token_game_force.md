# Current trained-critic payoff granularity diagnostic

The [fixed card](e22_supra_token_game_force_v1.json) and
[qualified results](e22_supra_token_game_force_results.json) test a proposed
per-token payoff on twelve predetermined, balanced fit contexts from each
6,400-update particle run. It reuses actual full-Supra residuals and Gaussian
panels; it performs no new Supra forwards or optimizer updates.

Each run's own trained critic has a restoring radial derivative on all twelve
contexts. The proposed per-token payoff gives weaker equal-norm descent under
both common native games, in both arms. Mean token-minus-native directional
slopes are positive: control `0.0003293 / 0.0003734`, neutral
`0.0001838 / 0.0001906`. Only one of 48 context/judge comparisons improves.
The declared decision therefore stops this proposed correction before quality
training. All 59 checks passed, and the completion receipt records 28.73 seconds
within the separate 120-second CPU budget.

Both payoffs use identical critic weights, conditioning, noise and physical
gradient units. The native negative gradient passes its own steepest-descent
control. Residual-power allocation is descriptive and never affects the
decision. These local residual-space observations do not establish behavior
through the adapter Jacobian, under Adam updates, or during joint D/G training.
They do not exclude every possible tokenwise formulation.

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 timeout --signal=TERM 120 \
  /ml2/ntc-image-studio/.venv-anima/bin/python -u \
  scripts/diagnose_e22_supra_token_game_force.py \
  --capture-report-sha256 617e0fa0fca3fbd1eb44e7f19b4efed0b533b15401b7f832024463f0f9a304f0 \
  --capture-sha256 5a9ea3d65f1ce14783d41482323c7d891312eb8cc0aa6888361ef536a64ffa91 \
  --qualified-review-sha256 a142f2fa8e074062fd9618b16180d974c23bbebdc2d56e0e6accff79895b62b4 \
  --output outputs/e22-supra-token-game-force-v1
```

The completed output directory is preserved and cannot be overwritten. Raw
artifacts and logs stay in ignored local storage. The observation does not
introduce output-MSE/RMSE optimization, structural guards or selection.

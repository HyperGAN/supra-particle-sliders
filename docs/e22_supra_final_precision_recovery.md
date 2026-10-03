This is a local full-Supra recovery qualification before a precision continuation. It is not a convergence result or a portable ParticleGAN toy. The original qualified neutral checkpoint at update 12,800 supplies all trained models, optimizer/controller state and streams; no initializer runs after restoring learned state.

The BF16 and FP32-final arms differ only in the FAST and EMA student's frozen final `Linear` arithmetic. Parameters retain FP32 storage. The teacher keeps its original BF16 autocast arithmetic. FP32 also changes the student's strength-zero prediction, so strength-zero exports are compared with their own declared arithmetic.

The software prerequisite passed 10 new cases in 3.938928 seconds of an external CPU60 cap. Its one tiny split/recovery case spent three native updates and six optimizer steps; it used no actual Supra weights or CUDA. Those checks are software qualification only. The full GPU qualification has not run.

The fixed GPU300 check will:

- Publicly restore the exact original 12,800 state in each freshly constructed precision owner, authenticate the complete legacy roundtrip before adding the explicit precision tag, and reproduce both qualified TEST240 scorer captures. It checks physical residuals, per-context MSE, four-panel games, source records, summary values and CPU72 panel identity.
- Interleave two reference native updates per arm with owned CPU/CUDA global streams. Sample the original CPU7 FIT B4 sequence, use zero residual targets and game weight 1, and call the unchanged native game helper. Save each arm at 12,801, then independently load that midpoint into matching fresh owners and reproduce the next row and complete 12,802 state. This costs three native updates per arm, six total.
- Compare each arm's full native/caller tensor bytes and devices, scalar values, gradients, modes, flags, forward classes and streams. Across arms it compares sampled indices and actual private draw traces; adaptive controller values, losses and learned weights may differ.
- Export independent explicit FAST and EMA snapshots, reload their precision-aware format, and compare clean raw velocities exactly at strengths 0 and 1. It also requires wrong-precision recovery/export rejection without modifying live owners, and rolls all retained owners back to their initial 12,800 state.

Required source-counted callbacks are 120 initial TEST student/live-teacher pairs, 12 native student/live-teacher pairs, and 16 clean export student calls. Native KA2 and structural callbacks remain unchanged. The 300-second cap includes startup, file checks, constructors, scorer anchors, six updates, disk recovery, exports, rollback and final writes.

The prepared local command is:

```bash
CUDA_VISIBLE_DEVICES=0 /ml2/ntc-image-studio/.venv-anima/bin/python \
  scripts/qualify_e22_supra_final_precision_recovery.py \
  --card docs/e22_supra_final_precision_recovery_protocol.json \
  --out outputs/e22-supra-final-precision-recovery-v1
```

Use `--preflight-only` for stdlib source/input hashes, schema and authenticated qualification receipts; it imports no Torch or ParticleGAN. The run refuses an existing output. Exit 0 means complete engineering qualification, and exit 2 means incomplete/error; partial evidence is retained. Its `scientific_status=PASS` denotes these plumbing checks only. Output accuracy never becomes a training objective, guard, checkpoint selection or stopping rule. A separate root-authorized continuation would still be needed to assess convergence; original Supra remains unbeaten.

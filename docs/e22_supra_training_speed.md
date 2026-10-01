# Full Supra ParticleGAN training speed

The fixed 6,400-update run is advancing normally. The current implementation
does substantially more work than the original ordinary LoRA recipe, and it
also performs avoidable GPU-to-CPU scalar transfers at every routing site.
This investigation was read-only: the training game, configuration, sampling
streams, live process and GPU jobs were unchanged.

## Measured timing

The log window from update 1,625 through 4,925 contains 3,300 updates and
2,362.32 seconds of wall time. Timing comes from synchronized log boundaries
every 25 updates, not individual profiled operations.

| Measurement | Time |
| --- | ---: |
| Ordinary 25-update interval, median per update | 0.533 s |
| All intervals, average per update | 0.716 s |
| Additional time per structural check, local interval estimate | 11.46 s |
| Additional time per checkpoint, local interval estimate | 26.08 s |
| Original 1,600-update LoRA trace, median per update | 0.0786 s |
| Original long-run updates 401–6,400, median per update | 0.0471 s |

Structural checks occur every 100 updates. Comparing their 25-update intervals
with neighboring ordinary intervals attributes about 378 seconds, or 16% of
this window, to those checks. Four native checkpoints, saved every 800 updates,
similarly account for about 104 seconds, or 4.4%. These are estimates from
interval differences; GPU contention and ordinary-step variation affect them.
Approximately 80% remains routine training and shared-machine effects.

CPU mmap metadata inspection of `checkpoint-04800.pt` found a 1,775,657,213-byte
file. Four frozen base-model tensor sets—fast generator, fast encoder teacher,
averaged generator and averaged encoder teacher—occupy 1,665,515,776 bytes,
**93.8% of the checkpoint**. Models and averages together occupy 96.3%; native
stationarity state uses 30.5 MB and optimizer state 23.3 MB. Tensor names,
shapes, storage sizes and `requires_grad` flags establish this storage breakdown;
the four sets were not hashed to establish content identity. A compact resume
format still needs verified base pins/hashes and complete native resume parity
before any frozen storage can safely be omitted.

Ordinary new updates are roughly 6.8 times the historical original 1,600-run
median and 11.3 times the original longer-run median. Those are historical
timing comparisons, not an isolated, matched GPU benchmark. The completed
first 1,600 new updates took 1,095.5 seconds including checkpoints and preflight;
the original 1,600 updates took 134.4 seconds including checkpoints and preview.

## Work required by the current formulation

The old recipe performs one student Supra prediction and backward pass against
cached teacher targets. Each current game update performs two routed student
predictions, one for the critic and one for the generator. The residual callback
also reruns the frozen target-caption Supra model for each prediction. That is
four complete Supra forward passes, followed by the generator backward pass,
critic learning and the native KA2 penalty.

The two student predictions each execute all 71 routing sites. Every site
computes an FP32 query, a 128-row softmax, a mixed bank code, DV12 nearest-support
perturbation and a nonlinear adapter. CFG doubles each batch's host activations.
The added math and kernel launches are real even though only about 1.9 million
generator/router/bank parameters are trainable.

A fully evaluated structural check uses a 64-context reservoir and up to 21
complete candidate reruns: one baseline, eight deletions, eight proposals and
four fast/averaged guard evaluations. The frozen teacher is currently rerun for
each candidate too. In this timing window 22 of 33 checks reached guards; the
remaining 11 stopped after the fit check. Whole-model reruns are required for
the sequential routing formulation because changing an early site changes
later routing queries.

## Avoidable synchronization found in source

Native `routed_generate` requests DV12 application diagnostics at every site.
`DataDriftController.perturb_latent` immediately converts five CUDA reductions
to Python floats: radius minimum, mean and maximum, perturbation RMS and clipped
fraction. Across 71 sites and two student predictions, that is **710 synchronous
scalar transfers per ordinary update**. Only the last two application records
are retained.

`RoutedExecution.mix` separately reads back logits and softmax finite checks at
every site, adding **284 synchronous scalar transfers per update**. The combined
994 transfers exclude context validation, critic checks, game logging and other
native optimizer bookkeeping. This count is established from the executed
source path; its exact percentage of step time is unmeasured.

Sources:

- [DV12 diagnostic reductions](/ml2/hypergan/ParticleGAN-pr155-merge/particlegan/continuous.py:148)
- [Per-site finite checks](/ml2/hypergan/ParticleGAN-pr155-merge/particlegan/routing.py:61)
- [Native per-site diagnostic request](/ml2/hypergan/ParticleGAN-pr155-merge/particlegan/policy.py:932)
- [Frozen teacher residual callback](/ml2/hypergan/supra-e22-verification/supra/particle_training.py:18)
- [Native game lifecycle](/ml2/hypergan/supra-e22-verification/supra/particle_game.py:130)
- [Structural candidate reruns](/ml2/hypergan/ParticleGAN-pr155-merge/particlegan/routing.py:690)

Both new optimizers explicitly use `foreach=False`. The generator optimizer
updates 427 trainable tensors, compared with 142 ordinary LoRA tensors. Eager
per-tensor launches and native stationarity histories add further overhead;
their stage timings were not measured.

## Shared hardware

GPU 1 is shared with another active Python training process. Process utilization
samples showed our run using roughly 39–80% SM and the other training job using
4–28%. A separate 17 GB llama-server allocation was idle in these samples.
Contention is demonstrated, but its isolated wall-time contribution is unknown.

GPU temperature samples were 86–89 C and SM clocks varied from 1,515 to 1,800
MHz. The queried thermal-slowdown reason was inactive, while software power
capping was intermittently active. These observations do not establish thermal
throttling as the cause. The main training thread used about one CPU core;
scheduler delay was under 1% of its scheduled CPU time in sampled counters.

Nonblocking `py-spy` attachment was attempted but denied by the operating
system's ptrace permissions. No elevated attachment, pause, competing GPU
benchmark or live profiler instrumentation was performed.

The subsequent 6,400-step evaluator also spent several minutes preparing its
checkpoint, model and export before scoring. Read-only process observations
showed waiting on file pages, approximately 1 GB swapped out, and swap-device
utilization of 87–100% with roughly 200–300 ms I/O latency. In those later
samples, `/ml2` disk read latency was under 1 ms; shared paging/swap pressure
was the immediate bottleneck. The evaluator recovered without a restart and
completed all metrics and 32 renders. This is a separate preparation delay,
not part of the training timing decomposition above. No isolated stack or
specific file cause for each earlier fault was established.

## Changes to validate after this benchmark

1. Defer and batch DV12 diagnostic scalar transfers; retain the same final
   application records and values without synchronizing every site.
2. Combine or asynchronously check routing finiteness while preserving failure
   detection. This needs ParticleGAN API support and failure-path tests.
3. Reuse the identical live teacher batch between critic and generator passes.
   For structural checks, reuse the matching teacher batch across candidates
   and guards with identical frozen teachers. Preserve full batch shape/order:
   cached individual BF16 examples can differ from the live batch's computation.
4. Deduplicate frozen model tensors in resumable checkpoints or keep a compact
   state plus verified pinned base. Preserve exact optimizer, controller, RNG,
   routing and averaged states and test resume equivalence.
5. Benchmark native-compatible foreach/fused optimizer execution separately.
   Treat numerical equivalence as unproven until verified.

None of these requires an output-MSE optimization criterion. Reducing routing
sites, DV12 behavior, candidate budgets or structural frequency would change the
experiment, so those are not proposed as implementation-only speedups for this
matched run. No speed change was applied to the live 6,400-update benchmark.

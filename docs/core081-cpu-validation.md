# Supra bounded CPU validation receipt

**18 passed, zero failures/skips**, exact tested product head
`73537b7214878cf22c1fd44b36ee8f78091ae9d7`, against clean core
`32f6830f6402e85aa77b6631db64fcbbb2131f6b` (0.8.1), released PG 0.9.1.
The product implementation is unchanged from
`0259a67c8078fddbd22aa586c84fff8d67720f6a`; the sole subsequent test change
compares the critic AST rather than requiring its historical EOF whitespace.

## Exact artifacts and environment

Product wheel SHA256:
`d26cef5d9ff138af44a4fe0853aac9b30293079abb50a047b97fed98c9b330ce`.
It is saved at task-5/validation/73537b721487/dist/
supra_particle_sliders-0.2.0-py3-none-any.whl.
Core wheel SHA256:
`279aaab83da05b6ebeca030dfe115e1ae2232ca4924fe4278ea48103918fbd8d`.
PG wheel SHA256:
`b01b7957169cb3de45887e22b4d525fd54071577e1de58089707cf05fe6992ec`.

Only the thin product wheel was built, from an exact Git archive in a separate
directory. Core and PG came from the parent's existing offline wheels. Pure
Python wheels were extracted into a new task-local venv; existing CPU dependencies
were borrowed read-only via a local .pth file. No pip/network resolution, global
installation, active-environment modification or core build occurred.
The tests ran under Python 3.12.15, Torch 2.14.1+cpu from `/tmp` with `-I`,
`--import-mode=importlib`, empty PYTHONPATH, source injection disabled, pytest
plugin autoload disabled and no cache provider. Offline/CUDA/thread variables
are captured in the driver/resource receipts. No production device gate was
relaxed: the native inference cases use an explicitly separate CPU test profile.

All **30 canonical core source/data hashes** matched. All **21 packaged product
source/data/license files** matched the exact tested Git blobs. Both installed
entry points resolved, and `particle_sliders.supra_model`,
`pg09_remaining_campaigns` and `pg09_moss_campaign` were confirmed absent.
The tests therefore do not rely on an untracked historical package closure.

## Scope established

- The relocated model is byte-identical to fd89e1c; selected Supra definitions
  are AST-equivalent except for the explicitly recorded public core import.
- Tiny real native DiT/LoRA parameter schemas, forward outputs and gradients
  match the old definition at zero, negative, fractional and unit scales.
- Native rows retain exact route/condition/target/provider identity and cache
  response scale. The critic/RNG helper remain actual common imports.
- The native factory performs real GAN updates, strict split resume with exact
  next metrics/full game state/EMA/RNG/coverage, and unchanged frozen-base digest.
- Corrupt restore and native update-admission rejection preserve live state.
  Native observation and failed preparation restore owned cache/counters,
  modes/flags/scales/encoder/CFG state. Invalid native requests fail in preflight.
- Raw and EMA v2 inference load/render after training files are deleted, with
  training/cache/game constructors forbidden. These cases execute the native-only
  factory through the explicit test profile and cannot train.

The final pytest worker took **5.01 seconds** including import/preflight, with
**355024 KiB peak RSS** (about 347 MiB), an enforced **2 GiB RLIMIT_AS**, one
operator thread and one inter-op thread. Minimum sampled host MemAvailable was
**44969279488 bytes** (about 41.9 GiB), above the 16 GiB reserve. The monitor
sampled every .2 seconds and would terminate a worker below that reserve.
CUDA was uninitialized before and after the suite. The first actual suite's
maximum RSS was 355728 KiB; all slot workers stayed within the same limits.

## Failure history and proof limits

The original exact 0259a67 suite passed 17/18. Its only failure compared the
shared critic bytes to fd89e1c; the reviewed clean core had removed one final
blank line. The correction compares the complete AST while keeping the separate
canonical-byte check. No model, provider, GAN, native inference or hook code was
changed in response to that failure.

Two driver-only startup issues were corrected: the system Python lacked the
newer tar extraction filter, so the existing Python 3.12 interpreter was used;
a generated JSON newline escape caused a SyntaxError before imports/tests on
one attempt. That attempt's log/resource receipt is preserved separately.
No approval denial or uncertain mutation occurred. The complete initial/final
test logs and XML remain available; failed receipts were not relabeled as passes.

This slot does **not** establish fresh-process native inference isolation,
full public CLI behavior, all legacy loader combinations, native CUDA/T5/VAE
loading, actual 256px decode, pretrained weight/cache parity, GPU fit, image
quality/convergence or HNDL equivalence. Production native rendering remains
separately admitted. No public write, publication, merge, model download, CUDA
execution or active training pin change occurred.

## Evidence

- [Final summary and product file hashes](../evidence/core081-cpu-slot/73537b721487/summary.json)
- [Final pytest XML](../evidence/core081-cpu-slot/73537b721487/tests.xml)
- [Resources and exact command](../evidence/core081-cpu-slot/73537b721487/tests-resources.json)
- [Installed origins](../evidence/core081-cpu-slot/73537b721487/installed-origins.json)
- [Artifact hashes](../evidence/core081-cpu-slot/73537b721487/artifacts.json)
- [Initial failure XML](../evidence/core081-cpu-slot/0259a67c8078/tests.xml)
- [Exact driver snapshot](../evidence/core081-cpu-slot/driver-used.py) (executed
  from task-5/validate_cpu.py; the archived copy preserves that original layout)

# Memory requirements

| Operation / profile | GPU count | Calculated VRAM per GPU (GiB) | Calculated host RAM (GiB) | Input limits / precision |
|---|---:|---:|---:|---|
| Train native default256px | 1 | 8 | 8 | 256×256; ≤128 tokens; B=4; FP32 storage/activation envelope; BF16 CUDA autocast; no checkpointing |
| Sample native256px | 1 | 8 | 8 | 256×256; ≤128 tokens; B=1; FP32 storage/activation envelope; BF16 CUDA autocast; no checkpointing |

These are **calculated planning budgets, not measured peaks or hard runtime ceilings**. GiB means 2³⁰ bytes. They describe the finite profiles above, one worker and one learned slider, with RAW weights and **no model-weight EMA**. A larger image, token sequence, image/reference count, logical batch, cache or number of simultaneous workers needs a new calculation; there is no finite universal maximum for unbounded inputs.

## Scope and residency

Base: **`SupraLabs/Supra2-IMG@10dec6e4b4b5d1c44fd1d7d3fe5e50137333da5b`**. The native Supra runtime keeps its FP32 DiT, FP32 T5 encoder and FP32 VAE resident together, using CUDA BF16 autocast for operations. Native latent4×32×32 fixes the output at256px. Published checkpoint filename model_final_ema.pt is an immutable upstream base filename; no new model-weight EMA is allocated.

The source constructor fixes256 tokens/256px and ctx_len128. It is not a scalable1536px DiT profile. Quantization, CPU execution and larger native images are not enabled by this loader.

The table's efficient-SDPA rows require a flash/efficient implementation. An automatic backend can fall back to dense math; use the dense row where provided, or add the dense term below before planning capacity. A planning row does not change code, precision, optimizer, rates, initialization, checkpoints or qualification.

## Frozen and trainable ledger

- DiT: **416,378,944 bytes** (0.388 GiB). 104,094,736 parameters.
- Flan-T5 encoder: **438,514,176 bytes** (0.408 GiB). 109,628,544 parameters.
- VAE serialized FP32 upper bound: **334,643,276 bytes** (0.312 GiB).

DiT constructor declares 104,094,736 parameters. Flan encoder only: V×d +12×(4d²+3d×d_ff+2d)+d+32×heads = 109,628,544; the full T5 decoder file is a loading allowance, not extra GPU decoder residency.

Trainable allowance: **1,758,913 parameters**. 16×(768+576)+14×16×(13×576)+60,097 critic = 1,758,913; no particle bank. FP32 trainable weights and gradients require `4A+4A` bytes. Frozen base parameters have no gradient or optimizer tensor allocation. Routed particle-bank parameters, where present, are counted once; a frozen response cache is not a trainable bank.

The current `bispf_optimized` PG0.11.1 recipe uses **DualNorm, momentum 0, direction_blend**; it is not assumed to be Adam. Persistent per-parameter moment tensors are0 in this profile; scalar counters are small. The peak ledger additionally reserves **`48A+32MiB` bytes** for protected normals, displacement/direction copies, FP64 promotion and thin-matrix SVD/Gram workspace. This is a conservative temporary-allocation allowance, not a claim that the optimizer uses no memory. Selecting historical `bispf`/Adam instead adds its actual state (typically another`8A` bytes for two FP32 moments); nonzero momentum or a different solver requires recalculation. Weight EMA adds0 bytes and must stay disabled.

## Numeric peak ledger (GiB before headroom)

| Profile | Frozen GPU weights | Activation / decode | Trainable + optimizer | Other GPU allowance | Host loading/cache subtotal |
|---|---:|---:|---:|---:|---:|
| Train native default256px | 1.108 | 2.953 | 0.123 | 0.000 | 2.821 |
| Sample native256px | 1.108 | 1.000 | 0.007 | 0.000 | 2.821 |

CPU rows place their active tensors in host RAM; their frozen/activation entries are not GPU residency. Sequential phase sub-ledgers are in the source-facts report.

Cache bound for the training profile: **128 rows**, each with three FP32 native fields, plus **19 cached prompts**. Tensor-only cache allowance: `6,291,456+7,471,104` bytes; masks/indices and serialization overhead remain covered by the explicit host subtotal and reserve. These bounds are inputs to this calculation, not new runtime limits.

## Activation, cache and host calculation

The [archived calculation report][source-facts] retains the original inputs and per-profile subtotals. For a transformer envelope, let `N=image tokens+bounded text tokens`, `D=hidden width`, `I=FFN width`, `H=attention heads`, `L=covered blocks`, and `B=simultaneously retained native graphs`:

- One-block inference allowance: `4B(16ND+6NI)` bytes. The4-byte coefficient deliberately covers FP32 intermediates even for BF16 weights;16 and6 allow Q/K/V/output, normalization/residual, gated FFN and adapter intermediates.
- Uncheckpointed training allowance: `4LB(16ND+6NI)`; checkpointed allowance: `4(4LBND+2B(16ND+6NI))`. Checkpointing saves layer inputs and recomputes block internals; it does not remove the frozen model, adapter or critic.
- Dense math fallback adds `8BH N²` bytes for inference and `16LBH N²` for uncheckpointed training, or `32BH N²` for the checkpointed current-block envelope. These are conservative materialized-score/probability allowances; flash/efficient SDPA avoids that quadratic storage. GQA expansion uses the query-head count for this bound.
- Decode allowance is separately based on bounded output/tile area and channel width; profile notes record the bound. Peak is the **maximum of sequential phases**, with the actual overlapping model weights added inside each phase. Sampling steps are sequential and do not multiply resident peak.
- Frozen CPU rows are bounded by the profile (three FP32 native fields per cached row, plus explicitly bounded prompt embeddings). Loader staging/serialization copies and the largest published shard are included in host subtotals. Host cache/page residency and kernel workspaces vary, so these are planning allowances, not an allocator theorem.

Final GPU budget is `ceil_to_4GiB(1.25×GPU subtotal+2GiB)`; host is `ceil_to_4GiB(1.25×host subtotal+4GiB)`. The explicit25% plus runtime reserve covers allocator fragmentation, CUDA context/libraries, kernel workspaces, small FP32 islands and Python/serialization overhead. These chosen reserve multipliers are explicit planning assumptions; architecture metadata alone cannot prove every allocator or fused-kernel peak. It is not borrowed from scheduler admission caps, card size or global free-memory floors. Do not add every row's peak together for sequential phases; parallel workers do add their simultaneous requirements.

This profile's text-state allowance is **0 bytes** (0.000 GiB): bidirectional T5 encoder; no autoregressive KV. It is already included in the phase or host allowance, not added twice. Text-only encoding has no generated-token loop. If the actual encoder returns KV state, it is explicitly included as `2×layers×B×KV_heads×tokens×head_dim×dtype_bytes`; cache-disabled encoders allocate no persistent autoregressive KV. Recurrent/linear-attention state is accounted separately where applicable. Arbitrary output-token generation and image-conditioned encoders are outside these profiles.

## Sources and reproducibility

- [supra/native_model.py](supra/native_model.py)
- [supra/runtime.py](supra/runtime.py)
- [supra/card.py](supra/card.py)
- [supra/lifecycle.py](supra/lifecycle.py)
- [supra/cache.py](supra/cache.py)
- The [archived calculation report][source-facts] retains exact model revisions, metadata SHA256s, direct Hugging Face config URLs and the optimizer source metadata used for the calculation. No weights were downloaded or opened for this calculation.
- The [current backend lock](backend.lock.json) selects ParticleGAN 0.11.1.

Documentation was calculated from source and published metadata on 2026-11-14. No model construction, import of Torch/Diffusers, forward pass, benchmark, sampling, training or new GPU work was performed.

[source-facts]: https://github.com/HyperGAN/supra-particle-sliders/blob/b9b6a2ae1ab37e5d6683e6369115d6fb03013ceb/docs/memory/source-facts.json

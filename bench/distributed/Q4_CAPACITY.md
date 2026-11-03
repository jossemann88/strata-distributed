# Qwen3.8-Flash-Next Q4 capacity on the M4 Pro and RTX 4070

**3 November 2026.** The eight-layer IQ3_S candidate (layers 2–9) needs 13.04 GiB of Q4 weights. That exceeds the 12 GB RTX 4070 before the first activation or cache allocation. A shorter block, layers 5–9, needs 7.72 GiB of Q4 weights. Its runtime footprint and speed on the RTX still need measurement.

## Weight budget

The counts below include every stored tensor named `blk.L.*` for each indicated layer, including all 512 routed experts, shared experts and attention weights. They exclude global tensors, format conversion, padding, driver allocations, CUDA workspace, recurrent state and KV for the requested context. GiB means 2³⁰ bytes.

| Inclusive block | IQ3_S weights, GiB | UD-Q4_K_XL weights, GiB |
| --- | ---: | ---: |
| 5–8 | 3.90 | 6.18 |
| 5–9 | 4.91 | 7.72 |
| 4–9 | 5.93 | 9.51 |
| 2–9 | 7.94 | 13.04 |
| 2–11 | 9.73 | 16.13 |

Across all contiguous blocks within layers 2–46, weight budgets of **4, 6, 8 and 10 GiB** fit at most **4, 6, 8 and 10 IQ3_S layers** respectively, or **2, 3, 5 and 6 Q4 layers**. Across all 48 layers the routed expert tensors alone occupy 46.84 GiB for IQ3_S and 71.73 GiB for Q4. The 32 GB of Windows RAM can stage weights, but any host-resident weights needed during decode add transfers or CPU work. Use measured free VRAM after context and scratch allocation to set the block budget.

**How to reproduce.** IQ3_S uses the installed ISTA-DASLab shard 1. For Q4 we read bytes 0–131071 of each of Unsloth's three tensor-bearing files (shards 2–4, 384 KiB in total). Shard 1 contains metadata only. Parse each directory with the repository's GGUF `Reader`, group its `tensor["bytes"]` by the integer in `blk.L.`, and sum the requested inclusive intervals. This counts gate, up and down once each, including layer 11 whose tensors cross shards. Count names ending in `_exps.weight` separately for the routed totals. The results describe the exact stored tensor shapes and quantizations. [Pinned Q4 files](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/38bb39ee97821de2c9009abb7e93950eec396e66/UD-Q4_K_XL) · [GGUF reader](../../metal/tools/metal_repack.py) · [IQ3_S block screen](results/2026-11-02-m4-pro/layer-wait/interior-blocks.json)

## Storage on the Mac

The model volume showed approximately 51 GB available at inspection. A separate Q4 setup requires 111.3 GB for the original shards and about 82.6 GB for the Metal repack, roughly **194 GB before other files**. If generated, the contiguous expert sidecar adds about 77 GB. Allow additional space for the pack, draft and temporary files. The 24 GB Mac would stream Q4 experts from SSD. Only the file headers were sampled for this report. [Mac pack requirements](../../docs/MAC.md) · [Repack and sidecar logic](../../metal/setup_mac.py) · [Shard sizes](../../docs/UNSLOTH_Q4.md)

## Quality and speed gates

Upstream compared each pack with Q8_0 over 4,995 teacher-forced positions on one EPYC/R9700 system. IQ3_S matched the Q8_0 top token 87.69% of the time, with mean top-64 KL of 0.144. Q4 matched 93.09%, with mean KL of 0.056. These results support testing Q4 for higher fidelity. A shorter remote Q4 block cannot reproduce those results by substituting Q4 weights into an IQ3_S run. Use the same complete pack on both hosts and verify its identity before a worker accepts a request. [Upstream quality method and results](../../docs/QUANTS.md#quality-the-yardstick)

1. Compare IQ3_S local versus distributed, then Q4 local versus distributed, with independent same-pack parity checks. Replay fixed activations and compare residuals at both block boundaries. Check logits and top-1 agreement on held-out teacher-forced tokens, including repeated and long-context requests. Keep the full routed expert set.
2. To compare the packs, use the same tokenizer, fixed upstream text set and representative coding prompts with executable pass/fail checks. Record task pass counts and quality-reference metrics separately from performance. Use the same context, sampling, speculation, power state and warmup. Interleave at least five paired runs. Report accepted output tokens/s, time to first token, inter-token p95 and peak dedicated VRAM for each configuration.
3. Measure Q4 local per-layer waits, full-block time, shaped TCP RTT and worker block time before estimating a gain. The recorded IQ3_S waits and the Q4 byte counts answer different questions. The decision is whether the complete Q4 system can deliver the required quality at the user's chosen throughput. [IQ3_S measurements and benchmark protocol](../../docs/DISTRIBUTED.md)

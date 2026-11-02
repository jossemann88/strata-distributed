# Strata Distributed

The experiment pairs a 24 GB M4 Pro running Qwen3.8-Flash-Next IQ3_S with a Windows 11 RTX 4070 (12 GB VRAM, 32 GB RAM) over home Wi-Fi. The question is whether keeping some experts resident on the RTX can save time on the Mac's decode critical path. The exploratory target is 10–12 generated tokens/s. Phase 0 adds standalone measurement tools. The first implementation of distributed inference depends on the measurements below.

## Provenance and baseline, 2026-11-02

[Upstream](https://github.com/Niko1221/Strata) and the [existing fork](https://github.com/jossemann88/strata-distributed) both pointed to `c62d5e3a15b40ea51c96aae9e8d36928965f53ab` on main. The project [license](../LICENSE) is MIT. Preserve attribution and permission notices. Check bundled dependency notices and the model provider's terms separately if packaging a release.

The Mac app is 0.2.9. Its bundled Metal engine 0.1.55 identifies source commit `4cf0d8cd3809`. The Metal source tree matches the experiment base. The saved app benchmark and new probes measure different things. The saved tune specifies expert streaming and `--spec 2`. Record the effective settings again for each paired comparison.

| Local evidence | Result | What was measured |
| --- | --- | --- |
| [Saved app benchmark](../bench/distributed/results/2026-11-02-m4-pro/saved-baseline.json) | 5.5 output tokens/s | Historical engine result, 24 input and 200 output tokens, greedy, thinking off. The user's observed range is 4.7–5.9. |
| [GGUF read samples](../bench/distributed/results/2026-11-02-m4-pro/gguf-reads.json) | Median 1.467 ms, p95 2.812 ms | 24 uniformly sampled expert reads simulating misses, 47.4 MB total. Three simultaneous reads from shard 1 per expert, one expert at a time, seed 4070. Includes Python scheduling. No sidecar or global cache flush. |
| [Resident Metal microbenchmark](../bench/distributed/results/2026-11-02-m4-pro/metal.txt) | 0.253 ms | One synthetic resident layer's route plan and gate/up/down kernels, top 10, T=1, IQ3_S / IQ4_NL. Median across seven batches of 32. Excludes SSD, attention, combine and network. |
| [Metal parity on model weights](../bench/distributed/results/2026-11-02-m4-pro/metal.txt) | PASS, output relative L2 1.03e−7 | Ten routed experts from layer 7, T=1, random activation against the upstream CPU reference. This layer uses IQ3_XXS / IQ4_NL despite the pack name. |

The first GGUF shard is 54.82 GB. The 28.80 GB second shard contains the n-gram table. The local log reported a 3.78 GB expert arena. A real miss can overlap prefetch, other misses or compute. Multiplying the single-expert latency by the expert count would overstate the time a remote worker can save.

## Next measurement on Wi-Fi

Use Python 3.8+ on each machine. Copy just [lan_probe.py](../lan_probe.py) to Windows. In PowerShell, run `ipconfig` and read the **IPv4 Address** under the active Wi-Fi adapter. Substitute that address for `WINDOWS_IP` in both commands. Start in the folder containing the file on Windows and in this checkout on the Mac.

**Windows PowerShell**

```powershell
py -3 .\lan_probe.py serve --bind WINDOWS_IP
```

**Mac**

```sh
python3 lan_probe.py measure --host WINDOWS_IP > wifi-idle-1.json
```

The worker listens for one connection on the specified address and TCP port 47321, then exits. Run it on the trusted home LAN. For a Windows Firewall prompt use the private network profile. Start it again for each repetition. Run three times idle and three times during representative Wi-Fi traffic, giving each file a different name and recording the traffic and link type. If available, repeat with wired Ethernet. Keep raw JSON with the test notes.

The tool measures sequential round trips on one warmed TCP connection. It sends 0/0, 10,240/10,240 and 10,240/102,400 payload bytes (request/reply), plus 8/4 bytes of framing. These cover control latency, one FP32 hidden row, and up to ten unweighted return rows. JSON contains all 100 timings per shape and nearest-rank p50/p95 after 10 warmups. It includes socket, scheduling and Python costs on both ends. The probe uses synthetic data. Loopback tests cover framing, response checks and timing statistics. The first [Mac to Windows Wi-Fi run](../bench/distributed/results/2026-11-02-m4-pro/wifi-run-1.json) measured p50/p95 of 3.42/4.35 ms for control, 5.97/10.95 ms for equal rows and 6.35/8.15 ms for the long return. Background traffic was uncontrolled. Compare the complete distributions before deciding on a partition.

## Proposed split and decision gates

1. **Establish costs.** Reproduce a warm, local-only decode in an isolated engine process when memory permits. Use the same GGUF, context, prompt, seed, speculation and output cap as the later A/B. Collect [ExpertStream::report](../metal/src/engine/stream.cpp) on normal exit, including miss and read-ahead hits, MB/window, reads, host and blocked wait. Keep prompt and generated-token counts separate. Report counters cover a process, and read bytes are logical. Use phase deltas for decode-only attribution. Relate windows to accepted tokens, especially with speculative decoding. Record real routed IDs and miss/wait per candidate layer with opt-in tracing before selecting a resident set.
2. **Choose the first layer.** Rank candidates by *avoidable non-overlapped time* and frequency on a held-out trace. Compare the measured local critical path against the proposed split: with concurrency, the slower of remaining local work and (marshal + payload RTT + CUDA + copies), plus merge time. Measure overlap explicitly. For T=1 with one synchronous request per layer, 48 layers at an illustrative 2 ms/round trip consume 96 ms per token before compute. At 5.5 tokens/s, reaching 10 or 12 requires saving about 82 or 98 ms per accepted token. Proceed to an isolated worker fixture only if an identifiable subset has headroom against the measured payload RTT. Retain a local-only result if no plausible subset qualifies.
3. **Prove one layer offline.** Match model digest, tensor names/types/shapes, format version and ordered expert slots. Measure free VRAM after CUDA context creation, budget workspace and outputs, then pin only a manifest of selected (layer, expert) slabs. The pack mixes quant types. Reuse the grouped kernels invoked by [RemoteExperts](../src/core/remote_experts.cpp). That class itself expects a second device in one process. Compare one real routed row on CPU, Metal and CUDA before adding sockets. Freeze tolerance from the local reference and quantized CUDA input path, record max error and relative L2, reject NaNs and verify routing order. Test each supported gate/up/down type pair before selecting it.
4. **Integrate one opt-in decode layer.** Mac owns attention, KV, routing, shared expert and final accumulation. At the streamed decode cut in [Window::at_cut and Window::layer](../metal/src/engine/window.cpp), split routed slots into local and resident-remote subsets. Exclude remote slots from both local SSD jobs and Metal dispatch. Send a normalized activation and ordered slot IDs over one persistent authenticated session. Return unweighted FP32 rows and combine exactly once in the original order. Keep prefill and unsupported window shapes on the existing path. Start at T=1. Verify supported speculative windows before testing the user's tuned configuration.
5. **Test loss and compare.** Check protocol version, model identity and resident set before inference. Start locally on a missing or incompatible worker. A request carries a session, sequence, window and layer ID, bounded lengths and a deadline. On timeout, recompute missing rows locally from retained inputs and commit once; discard late/duplicate replies. Test unplugging the worker before and during decode. Confirm the disabled path matches the upstream token sequence against an identical build. Run paired local/remote trials with warmup and alternating order (at least five pairs). Record accepted output tokens/s, prompt rate, TTFT, inter-token p50/p95, output parity, SSD wait and RPCs per accepted token. Record hardware, OS, driver, power, model identity, warm/cold state and revisions. Claim an improvement only if repeated gains exceed run-to-run variation and parity passes.

## Reproduce the Mac checks

Set `GGUF` to the existing first shard and `BUILD_DIR` to a separate scratch build directory. The read probe uses the upstream GGUF layout reader and opens the file read-only. Its time covers a cache-policy request on its own descriptor. Actual physical disk traffic requires separate tracing.

```sh
python3 -m unittest discover -s bench/distributed -p 'test_*.py' -v
python3 -m unittest discover -s metal/tools -p 'test_expert_read_probe.py' -v
python3 metal/tools/expert_read_probe.py --gguf "$GGUF" --samples 24 --seed 4070
cmake -S metal -B "$BUILD_DIR" -DSTRATA_METAL_ENGINE=OFF
cmake --build "$BUILD_DIR" --target expert_parity
"$BUILD_DIR/expert_parity" --bench --gu IQ3_S --dn IQ4_NL --ntok 1 --layers 1 --reps 32
"$BUILD_DIR/expert_parity" --gguf "$GGUF" --layer 7 --ntok 1 --seed 4070
```

The measured Mac build used `-DCMAKE_OSX_SYSROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk` because its default SDK and linker did not match.

## Measure blocked reads during generation

Build the full Metal target `strata` in a separate build directory using the upstream pinned ggml and metal-cpp dependencies. The engine reports layer statistics when launched with `STRATA_METAL_LAYER_WAIT=1` and closed normally. The option applies to streamed, single-sequence verifier windows that request a sampled head. Prompt-only reads and batch slots are excluded. Each layer reports total windows (including cache hits), windows that entered the wait path, and elapsed milliseconds in that wait path. The clock includes reacquiring the I/O mutex and waking the host thread.

For an isolated diagnostic on the IQ3_S pack, set `BUILD_DIR`, `PACK`, `GGUF`, `PLE_GGUF` and `EXPERT_PROFILE` to the build directory, pack, first shard, second shard and installed ranking file. Run it after the regular engine releases memory. Use a fresh scratch directory for output. Compare the printed arena size with the production arena because available memory affects it.

```sh
STRATA_METAL_LAYER_WAIT=1 STRATA_METAL_STREAM=1 "$BUILD_DIR/strata" generate \
  --pack "$PACK" --native "$GGUF" --ple-gguf "$PLE_GGUF" \
  --expert-profile "$EXPERT_PROFILE" --expert-cache 64 --mmap-experts \
  --prefill auto --spec 2 --suffix-draft 0 --max-context 65536 --vram-reserve-mib 2048 \
  --greedy --tokens 9707 --max-new 64 > tokens.txt 2> engine.log
```

This short fixed-input run is for attributing wait, not for reproducing the historical app throughput. For parity, repeat with the flag unset, compare the generated token IDs, and verify the layer report appears only with the flag enabled. From the checkout, rank T=1 rows with:

```sh
python3 bench/distributed/layer_wait.py /path/to/engine.log bench/distributed/results/2026-11-02-m4-pro/wifi-run-1.json
```

The report places average wait per layer beside the measured average RTT (7.71 ms for the one-row response in run 1). Both include occasional stalls. This is a screen for a serial one-request-per-layer design. Evaluate overlap, RTX compute, staging and reduction separately. A whole layer of routed experts in this pack takes 0.72–1.27 GiB depending on its tensor types, before CUDA work buffers. A partial resident set can save only the wait attributable to its selected experts.

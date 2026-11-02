#!/usr/bin/env python3
"""Measure one randomly selected GGUF expert miss at a time on macOS.

Targets Strata's 48-layer, 512-expert Metal decode path with default read
split=1 and no contiguous .experts sidecar. Gate, up and down are read in
parallel (three preads). Includes Python scheduling and buffer allocation.
A descriptor-local cache policy does not establish physical SSD latency.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time

import stream_experts

MAX_PAYLOAD = 120 * 1024 * 1024


def expert_slices(layer, expert, file_size):
    tensors, sizes = layer[:3], layer[3:]
    spans = []
    for tensor, size in zip(tensors, sizes):
        if (tensor["dims"][-1] != stream_experts.NE or size <= 0
                or size * stream_experts.NE != tensor["bytes"]):
            raise ValueError("unexpected routed expert dimensions")
        offset = tensor["abs"] + expert * size
        if offset < 0 or offset + size > file_size:
            raise ValueError("expert slice outside GGUF")
        spans.append((offset, size))
    return spans


def read_span(fd, offset, size):
    while size:
        data = os.pread(fd, size, offset)
        if not data:
            raise OSError("short GGUF read")
        offset += len(data)
        size -= len(data)


def probe(gguf, samples, seed):
    if sys.platform != "darwin":
        raise ValueError("the Metal file-cache policy requires macOS")
    import fcntl

    if not 1 <= samples <= 48:
        raise ValueError("choose between 1 and 48 expert samples")
    reader, layout, *_ = stream_experts.plan(gguf)
    with reader.f as model:
        fd = model.fileno()
        file_size = os.fstat(fd).st_size
        fcntl.fcntl(fd, getattr(fcntl, "F_NOCACHE", 48), 1)
        fcntl.fcntl(fd, getattr(fcntl, "F_RDAHEAD", 45), 0)
        picks = random.Random(seed).sample(range(len(layout) * stream_experts.NE), samples)
        jobs = []
        total = 0
        for pick in picks:
            layer, expert = divmod(pick, stream_experts.NE)
            spans = expert_slices(layout[layer], expert, file_size)
            total += sum(size for _, size in spans)
            jobs.append((layer, expert, spans))
        if total > MAX_PAYLOAD:
            raise ValueError("selected expert payload exceeds 120 MiB")
        rows = []
        with ThreadPoolExecutor(max_workers=3) as pool:
            for layer, expert, spans in jobs:
                start = time.perf_counter_ns()
                reads = [pool.submit(read_span, fd, offset, size) for offset, size in spans]
                for read in reads:
                    read.result()
                elapsed_ms = (time.perf_counter_ns() - start) / 1e6
                rows.append(dict(layer=layer, expert=expert, bytes=sum(n for _, n in spans),
                                 elapsed_ms=round(elapsed_ms, 6)))
    times = sorted(row["elapsed_ms"] for row in rows)
    return dict(schema="strata.gguf-expert-read.v1", timestamp=datetime.now(timezone.utc).isoformat(),
                gguf=gguf.name, gguf_bytes=file_size, samples=samples, seed=seed,
                layout="three independent GGUF slices per expert", concurrency=3,
                cache_policy="F_NOCACHE=1,F_RDAHEAD=0 on probe handle; no cache purge",
                metric="wall time from submit to completion per expert; includes Python overhead",
                total_expert_bytes=total, median_ms=statistics.median(times),
                p95_nearest_rank_ms=times[math.ceil(.95 * len(times)) - 1], rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", required=True, type=Path, help="first GGUF shard used by Metal")
    parser.add_argument("--samples", default=24, type=int)
    parser.add_argument("--seed", default=4070, type=int)
    args = parser.parse_args()
    try:
        result = probe(args.gguf, args.samples, args.seed)
    except (KeyError, ValueError, OSError) as error:
        parser.exit(2, f"expert read probe failed: {error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

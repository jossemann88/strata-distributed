"""Run the isolated IQ3_S Metal control, capture, timing and TCP replay for layers 4–9."""
import argparse
import json
import math
import os
import re
import statistics
import struct
import subprocess
import sys
from pathlib import Path

from fixture_replay import load_fixture, replay

REQUEST = b"GEN 64 9707\nQUIT\n"
TIMING = re.compile(r"strata metal block_time: begin=4 end=10 T=1 pos=(\d+) elapsed_ms=([\d.]+)")


def engine_output(path):
    lines = path.read_text().splitlines()
    ids = [int(line[2:]) for line in lines if line.startswith("T ")]
    done = [line.split() for line in lines if line.startswith("DONE ")]
    if len(ids) != 64 or len(done) != 1 or done[0][1:3] != ["64", "1"] or done[0][5] != "length":
        raise ValueError(f"incomplete one-token prompt / 64-token response in {path.name}")
    return ids


def block_times(path):
    samples = [(int(m[1]), float(m[2])) for m in TIMING.finditer(path.read_text())]
    if [p for p, _ in samples] != list(range(64)):
        raise ValueError("expected one T=1 timing at each position 0 through 63")
    if any(not math.isfinite(ms) or ms <= 0 for _, ms in samples):
        raise ValueError("invalid block time")
    steady = [ms for pos, ms in samples if pos > 0]
    return {"first_position_ms": samples[0][1], "later_positions": len(steady),
            "later_p50_ms": statistics.median(steady),
            "later_p95_ms": sorted(steady)[math.ceil(.95 * len(steady)) - 1]}


def run(args):
    if sys.platform != "darwin":
        raise ValueError("capture requires macOS and a Metal build")
    active = subprocess.run(["pgrep", "-x", "strata"], capture_output=True)
    if active.returncode != 1 or active.stderr.strip():
        raise ValueError("cannot confirm the normal Strata engine is unloaded; leaving it alone")
    binary, pack, shard, ple, profile = (p.resolve(strict=True) for p in
            (args.engine, args.pack, args.shard, args.ple_shard, args.expert_profile))
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    capture = out / "boundary"
    capture.mkdir()
    command = [str(binary), "--serve", "--pack", str(pack), "--native", str(shard),
               "--ple-gguf", str(ple), "--expert-profile", str(profile), "--expert-cache", "64",
               "--mmap-experts", "--prefill", "auto", "--spec", "2", "--suffix-draft", "0",
               "--max-context", "65536", "--vram-reserve-mib", "2048", "--greedy"]
    env = {k: v for k, v in os.environ.items() if not k.startswith("STRATA_")}
    env.update(STRATA_METAL_STREAM="1", STRATA_METAL_ARENA_GB="3.78")
    env["STRATA_METAL_EXPERT_PROFILE"] = str(profile)
    for name in ("control", "probe"):
        if name == "probe":
            env.update(STRATA_METAL_BOUNDARY_DIR=str(capture), STRATA_METAL_BLOCK_TIMING="1")
        with (out / f"{name}.out").open("wb") as stdout, (out / f"{name}.log").open("wb") as stderr:
            subprocess.run(command, input=REQUEST, cwd=out, env=env, stdout=stdout, stderr=stderr,
                           check=True, timeout=600)
    ids = engine_output(out / "probe.out")
    if ids != engine_output(out / "control.out"):
        raise ValueError("probe changed the generated tokens")
    reference = Path(__file__).parent / "results/2026-11-02-m4-pro/layer-wait/serve-parity.txt"
    if ids != list(map(int, reference.read_text().splitlines()[1].split())):
        raise ValueError("output differs from the recorded IQ3_S reference")
    before, after, _ = load_fixture(capture)
    if before == after:
        raise ValueError("block input and output are identical")
    for payload in (before, after):
        values = [x[0] for x in struct.iter_unpack("<f", payload)]
        if not all(map(math.isfinite, values)) or not any(values):
            raise ValueError("expected finite, nonzero model activations")
    result = {"reference_tokens_matched": len(ids), "timing": block_times(out / "probe.log"),
              "replay": replay(capture)}
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("engine", "pack", "shard", "ple-shard", "expert-profile", "out"):
        parser.add_argument(f"--{flag}", required=True, type=Path)
    try:
        print(json.dumps(run(parser.parse_args()), indent=2))
    except (ValueError, EOFError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        parser.exit(1, f"capture failed: {error}\n")

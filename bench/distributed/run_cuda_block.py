"""Run the offline CUDA block on the captured IQ3_S input and compare its result."""
import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path

from compare_cuda_block import BLOCK, WEIGHT_SHA256, compare, sha256
from fixture_replay import STATE_BYTES, load_fixture

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "iq3_s-t1-p0"
SHARD_SIZE = 54817524224
# Byte identities of the three pack files used by the captured Mac engine.
PACK_DIGESTS = {
    "index.txt": "b2a014878c3b61d6a6f915e194055a5e68ede34f9465dd1cb13249d902f008ac",
    "native_experts.txt": "d9ac2dfa3ee63c55c9c6a6db26f72da0cec0ee41733f007e5cbbbba71617aa5d",
    "dense.bin": "d2af3d8331e5c7a1985d1d589747230f9d56719a6d33c7cddc9ee6b81cde6b13",
}


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def measurement(transcript):
    fields = {}
    for line in transcript.splitlines():
        key, sep, value = line.partition("=")
        if sep and key.startswith(("p0_", "vram_free_bytes_", "cuda_device")):
            if key in fields:
                raise ValueError(f"duplicate CUDA measurement: {key}")
            fields[key] = value.strip()

    def number(key, convert=float):
        if key not in fields:
            raise ValueError(f"missing CUDA measurement: {key}")
        value = convert(fields[key])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid CUDA measurement: {key}")
        return value

    if not fields.get("cuda_device"):
        raise ValueError("missing CUDA device name")
    warm = [number(f"p0_warm_{i}_ms") for i in range(1, 11)]
    first = number("p0_first_graph_ms")
    if min(warm + [first]) <= 0:
        raise ValueError("expected positive block times")
    return {"device": fields["cuda_device"], "first_graph_ms": first,
            "warm_p0_samples_ms": warm, "warm_p0_median_ms": statistics.median(warm),
            "max_repeat_delta": number("p0_max_repeat_delta"),
            "vram_free_bytes": {phase: number("vram_free_bytes_" + phase, int)
                                for phase in ("before_setup", "after_init", "after_first_graph")}}


def run(engine, pack, shard, out):
    engine, pack, shard = (p.resolve(strict=True) for p in (engine, pack, shard))
    out = out.resolve()
    if out.exists():
        raise ValueError(f"choose a new result directory: {out}")
    if shard.stat().st_size != SHARD_SIZE:
        raise ValueError("expected the original IQ3_S GGUF shard 1 (54,817,524,224 bytes)")
    print("Checking the model shard and pack against the Mac capture...", flush=True)
    for name, expected in PACK_DIGESTS.items():
        if file_digest(pack / name) != expected:
            raise ValueError(f"{name} differs from the pack used for the Mac capture")
    if file_digest(shard) != WEIGHT_SHA256:
        raise ValueError("GGUF shard digest differs from the Mac capture")
    before, _, identity = load_fixture(FIXTURE / "boundary")
    out.mkdir(parents=True, exist_ok=False)
    input_path = out / "input.f32"
    output_path = out / "output.f32"
    input_path.write_bytes(before)
    build_id = file_digest(engine)
    command = [str(engine), "--pack", str(pack), "--shard", str(shard),
               "--input", str(input_path), "--output", str(output_path)]
    log = out / "runner.log"
    print(f"Running layers 4-9 on CUDA. Log: {log}", flush=True)
    env = {key: value for key, value in os.environ.items() if not key.upper().startswith("STRATA_")}
    with log.open("wb") as transcript:
        completed = subprocess.run(command, stdout=transcript, stderr=subprocess.STDOUT,
                                   cwd=out, env=env, timeout=900)
    transcript = log.read_text(encoding="utf-8", errors="replace")
    print(transcript, flush=True)
    if completed.returncode:
        raise ValueError(f"CUDA runner exited with {completed.returncode}, see {log}")
    timing = measurement(transcript)
    if output_path.stat().st_size != STATE_BYTES:
        raise ValueError(f"runner must return exactly {STATE_BYTES} bytes")
    write_json(out / "manifest.json", {
        **BLOCK, "schema": "strata.cuda-block-output.v1", "backend": "cuda",
        "operation": "compute_full_block", "dtype": "f32le", "device": timing["device"],
        "build_id": "exe-sha256:" + build_id, "fixture_sha256": identity.hex(),
        "weight_sha256": WEIGHT_SHA256, "input_sha256": sha256(before),
        "output_sha256": file_digest(output_path),
    })
    result = compare(FIXTURE / "boundary", FIXTURE / "summary.json", out)
    result["position_zero_measurement"] = timing
    write_json(out / "comparison.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return result["within_tolerance"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("engine", "pack", "shard", "out"):
        parser.add_argument("--" + flag, required=True, type=Path)
    args = parser.parse_args()
    try:
        sys.exit(0 if run(**vars(args)) else 1)
    except (ValueError, OSError, subprocess.TimeoutExpired) as error:
        parser.exit(2, f"CUDA block test failed: {error}\n")

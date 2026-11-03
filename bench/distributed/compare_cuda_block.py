"""Compare a CUDA worker's computed layers [4,10) output with a Metal capture.

Run with --capture-dir RUN/boundary --capture-summary RUN/summary.json
--cuda-dir RESULT. RESULT contains output.f32 (little-endian residual, bo, inj)
and manifest.json. The worker receives the input and opaque fixture ID only.
A trusted producer emits the manifest after the CUDA block finishes, recording
schema=strata.cuda-block-output.v1, backend=cuda, operation=compute_full_block,
dtype=f32le, start_layer=4, end_layer_exclusive=10, rows=1, position=0,
device, build_id, fixture_sha256, weight_sha256, input_sha256 and output_sha256.
weight_sha256 identifies native GGUF shard 1. output_sha256 hashes output.f32.
The checker establishes numerical agreement and reports producer-declared provenance.
Exit 0 means all values met the gates, 1 means a numerical mismatch, and 2
means the artifacts or arguments were invalid.
"""

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

from fixture_replay import PARTS, STATE_BYTES, load_fixture


# IQ3_S native weight shard, from setup.py's ISTA_FILES.
WEIGHT_SHA256 = "4c1eb2ceb4915e1192f4f386021897bde56a97f40a0bb78bb86465e0f7d2aca3"
BLOCK = {"start_layer": 4, "end_layer_exclusive": 10, "rows": 1, "position": 0}
# Fixed exploratory gates (absolute and relative). Independently check every float.
LIMITS = {"residual": (1e-5, 1e-3), "bo": (1e-5, 1e-3), "inj": (1e-6, 1e-4)}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def require_fields(record, expected, name):
    if not isinstance(record, dict):
        raise ValueError(f"{name} must be a JSON object")
    for key, value in expected.items():
        actual = record.get(key)
        if type(actual) is not type(value) or actual != value:
            raise ValueError(f"{name} does not match expected {key}")


def components(payload):
    offset = 0
    for name, size in PARTS:
        values = struct.unpack_from(f"<{size // 4}f", payload, offset)
        if not all(map(math.isfinite, values)):
            raise ValueError(f"{name} contains NaN or infinity")
        yield name, values
        offset += size


def compare(capture_dir, capture_summary, cuda_dir):
    before, expected, identity = load_fixture(capture_dir)
    summary = json.loads(capture_summary.read_text(encoding="utf-8"))
    require_fields(summary, {"reference_tokens_matched": 64}, "control summary")
    fingerprint = identity.hex()
    require_fields(summary.get("replay"), {
        **BLOCK, "method": "localhost_fixture_replay", "exact_match": True,
        "request_bytes": STATE_BYTES, "response_bytes": STATE_BYTES,
        "fixture_sha256": fingerprint, "input_sha256": sha256(before), "output_sha256": sha256(expected),
    }, "Metal capture")
    inputs = dict(components(before))
    if before == expected or not any(any(values) for values in inputs.values()):
        raise ValueError("degenerate Metal capture")
    reference = dict(components(expected))
    if not any(any(values) for values in reference.values()):
        raise ValueError("Metal reference contains only zeros")

    manifest = json.loads((cuda_dir / "manifest.json").read_text(encoding="utf-8"))
    with (cuda_dir / "output.f32").open("rb") as stream:
        actual = stream.read(STATE_BYTES + 1)
    if len(actual) != STATE_BYTES:
        raise ValueError(f"CUDA output must be exactly {STATE_BYTES} bytes")
    require_fields(manifest, {
        **BLOCK, "schema": "strata.cuda-block-output.v1", "operation": "compute_full_block",
        "backend": "cuda", "dtype": "f32le", "weight_sha256": WEIGHT_SHA256,
        "fixture_sha256": fingerprint, "input_sha256": sha256(before), "output_sha256": sha256(actual),
    }, "CUDA manifest")
    for key in ("device", "build_id"):
        if not isinstance(manifest.get(key), str) or not manifest[key].strip():
            raise ValueError(f"CUDA manifest requires {key}")
    if actual == before:
        raise ValueError("CUDA output echoes the captured input")

    metrics = {}
    for name, values in components(actual):
        atol, rtol = LIMITS[name]
        errors = [abs(x - ref) for x, ref in zip(values, reference[name])]
        mismatches = sum(error > atol + rtol * abs(ref) for error, ref in zip(errors, reference[name]))
        metrics[name] = {"floats": len(values), "atol": atol, "rtol": rtol,
                         "outside_tolerance": mismatches, "max_abs_error": max(errors),
                         "rms_error": math.sqrt(math.fsum(x * x for x in errors) / len(errors))}
    return {"schema": "strata.block-comparison.v1", **BLOCK, "fixture_sha256": fingerprint,
            "cuda_output_sha256": sha256(actual), "cuda_device": manifest["device"],
            "cuda_build_id": manifest["build_id"], "components": metrics,
            "within_tolerance": all(m["outside_tolerance"] == 0 for m in metrics.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("capture-dir", "capture-summary", "cuda-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = compare(args.capture_dir, args.capture_summary, args.cuda_dir)
    except (OSError, ValueError) as error:
        parser.exit(2, f"comparison failed: {error}\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["within_tolerance"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

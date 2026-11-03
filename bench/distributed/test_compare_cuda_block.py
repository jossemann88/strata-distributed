import json
import struct
import tempfile
import unittest
from pathlib import Path

import compare_cuda_block as checker
import fixture_replay


class TestBlockComparison(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.summary = self.root / "summary.json"
        for prefix, scale in (("layer04", 1), ("layer10", 2)):
            for i, (name, size) in enumerate(fixture_replay.PARTS):
                (self.root / f"{prefix}-{name}.f32").write_bytes(struct.pack("<f", scale + i) * (size // 4))
        self.before, self.after, identity = fixture_replay.load_fixture(self.root)
        self.capture_info = {**checker.BLOCK, "method": "localhost_fixture_replay", "exact_match": True,
                             "request_bytes": 51216, "response_bytes": 51216, "fixture_sha256": identity.hex(),
                             "input_sha256": checker.sha256(self.before), "output_sha256": checker.sha256(self.after)}
        self.summary.write_text(json.dumps({"reference_tokens_matched": 64, "replay": self.capture_info}))
        self.manifest = {**checker.BLOCK, "schema": "strata.cuda-block-output.v1", "backend": "cuda",
                         "operation": "compute_full_block", "dtype": "f32le", "device": "test-only",
                         "build_id": "test-only", "fixture_sha256": identity.hex(),
                         "weight_sha256": checker.WEIGHT_SHA256, "input_sha256": checker.sha256(self.before)}

    def run_case(self, output, **fields):
        (self.root / "output.f32").write_bytes(output)
        (self.root / "manifest.json").write_text(json.dumps(
            {**self.manifest, "output_sha256": checker.sha256(output), **fields}))
        return checker.compare(self.root, self.summary, self.root)

    def test_rounded_outputs(self):
        data = bytearray(self.after)
        struct.pack_into("<f", data, 0, 2.0001)
        self.assertTrue(self.run_case(data)["within_tolerance"])

    def test_each_component_gates_independently(self):
        offset = 0
        for name, size in fixture_replay.PARTS:
            with self.subTest(name=name):
                data = bytearray(self.after)
                struct.pack_into("<f", data, offset + size - 4, 25)
                result = self.run_case(data)
                self.assertFalse(result["within_tolerance"])
                self.assertEqual(result["components"][name]["outside_tolerance"], 1)
            offset += size

    def test_nonfinite_and_partial_output(self):
        for x in (float("nan"), float("inf"), -float("inf")):
            data = bytearray(self.after)
            struct.pack_into("<f", data, 51212, x)
            with self.assertRaisesRegex(ValueError, "NaN or infinity"):
                self.run_case(data)
        for data in (self.after[:40960], self.after + b"0"):
            with self.assertRaisesRegex(ValueError, "51216"):
                self.run_case(data)

    def test_manifest_and_fixture_identity(self):
        for key, bad in (("backend", "localhost_fixture_replay"), ("weight_sha256", "00"),
                         ("input_sha256", "00"), ("output_sha256", "00"), ("fixture_sha256", "00"),
                         ("operation", "native_gr_post"), ("rows", True), ("position", 1)):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                self.run_case(self.after, **{key: bad})
        with self.assertRaisesRegex(ValueError, "echoes"):
            self.run_case(self.before)
        self.summary.write_text(json.dumps({"reference_tokens_matched": 64,
                                          "replay": {**self.capture_info, "fixture_sha256": "incorrect"}}))
        with self.assertRaisesRegex(ValueError, "fixture_sha256"):
            self.run_case(self.after)


if __name__ == "__main__":
    unittest.main()

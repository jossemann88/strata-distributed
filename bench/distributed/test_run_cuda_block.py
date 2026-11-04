import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

import run_cuda_block as probe


def log_lines():
    return ["cuda_device=test", "p0_first_graph_ms=22", "p0_max_repeat_delta=0",
            "vram_free_bytes_before_setup=180", "vram_free_bytes_after_init=100",
            "vram_free_bytes_after_first_graph=90"] + [f"p0_warm_{n}_ms={n}" for n in range(1, 11)]


class BlockProbeTests(unittest.TestCase):
    def test_measurement_requires_complete_finite_series(self):
        lines = log_lines()
        self.assertEqual(probe.measurement("\n".join(lines))["warm_p0_median_ms"], 5.5)
        for bad in (lines[:-1], lines + lines[-1:], lines[:-1] + ["p0_warm_10_ms=nan"]):
            with self.assertRaises(ValueError):
                probe.measurement("\n".join(bad))

    def test_runner_receives_only_input_and_output_is_compared(self):
        before, after, _ = probe.load_fixture(probe.FIXTURE / "boundary")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stub = root / "stub"
            stub.write_bytes(b"stub")
            out = root / "result"

            def fake_run(args, **kwargs):
                src = Path(args[args.index("--input") + 1])
                self.assertEqual(src.read_bytes(), before)
                self.assertNotIn(str(probe.FIXTURE), " ".join(args))
                Path(args[args.index("--output") + 1]).write_bytes(after)
                kwargs["stdout"].write("\n".join(log_lines()).encode())
                return CompletedProcess(args, 0)

            digest = probe.file_digest(stub)
            with patch.object(probe, "WEIGHT_SHA256", digest), patch("compare_cuda_block.WEIGHT_SHA256", digest), \
                 patch.object(probe, "SHARD_SIZE", 4), patch.object(probe, "PACK_DIGESTS", {"stub": digest}), \
                 patch.object(probe.subprocess, "run", side_effect=fake_run), redirect_stdout(StringIO()):
                self.assertTrue(probe.run(stub, root, stub, out))
                result = json.loads((out / "comparison.json").read_text())
                self.assertTrue(result["within_tolerance"])
                with self.assertRaisesRegex(ValueError, "new result directory"):
                    probe.run(stub, root, stub, out)
                stub.write_bytes(b"oops")
                with self.assertRaisesRegex(ValueError, "differs"):
                    probe.run(stub, root, stub, root / "bad")

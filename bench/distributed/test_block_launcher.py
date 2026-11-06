"""Regression checks for failures before the CUDA process starts."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

PWSH = shutil.which("pwsh")
ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(PWSH, "PowerShell is unavailable")
class LauncherLoggingTests(unittest.TestCase):
    def launch(self, build_failure):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cuda = root / "bench/distributed/cuda"
            cuda.mkdir(parents=True)
            script = cuda / "run-block.ps1"
            shutil.copyfile(ROOT / "bench/distributed/cuda/run-block.ps1", script)
            pin = root / "third_party/ggml"
            pin.mkdir(parents=True)
            (pin / "VERSION.txt").write_text("a" * 40)
            harness = root / "test.ps1"
            harness.write_text("""
function git { $global:LASTEXITCODE = 0; return ('a' * 40) }
function nvcc { $global:LASTEXITCODE = 0 }
function cmake {
    Write-Output 'SIMULATED COMPILER FAILURE C1234'
    $global:LASTEXITCODE = 1
}
& $env:PROBE_SCRIPT @args
""")
            result = subprocess.run(
                [PWSH, "-NoProfile", "-File", str(harness)] + (["-BuildOnly"] if build_failure else []),
                env={**os.environ, "LOCALAPPDATA": str(root / "local"), "PROBE_SCRIPT": str(script)},
                capture_output=True, text=True, timeout=30,
            )
            logs = list((cuda / "results").glob("run-*.txt"))
            self.assertEqual(len(logs), 1, result.stdout + result.stderr)
            log = logs[0].read_text(encoding="utf-8-sig")
            self.assertIn("FAILED. Upload this log:", log)
            self.assertNotIn("Built ", log)
            return log

    def test_missing_arguments_are_logged(self):
        self.assertIn("Set -Pack, -Shard and -Out", self.launch(False))

    def test_configuration_failure_includes_compiler_output(self):
        log = self.launch(True)
        self.assertIn("SIMULATED COMPILER FAILURE C1234", log)
        self.assertIn("Configuration failed", log)

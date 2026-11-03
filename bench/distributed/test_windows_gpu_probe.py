import ctypes as C
import unittest
from unittest.mock import Mock, patch

import windows_gpu_probe as probe


class WindowsGpuProbeTests(unittest.TestCase):
    def setUp(self):
        self.driver = Mock()
        self.buf = C.create_string_buffer(256)
        self.ram = lambda: {"total_bytes": 8192, "available_bytes": 4096}
        self.driver.cuMemGetInfo_v2.side_effect = self.set_vram
        self.driver.cuMemHostAlloc.side_effect = self.set_host
        self.addCleanup(patch.stopall)
        patch.object(probe, "TRANSFER_BYTES", (128, 256)).start()

    def set_host(self, ptr, capacity, flags):
        self.assertEqual((capacity, flags), (256, 0))
        C.cast(ptr, C.POINTER(C.c_void_p))[0] = C.addressof(self.buf)

    def set_vram(self, free, total):
        C.cast(free, C.POINTER(C.c_size_t))[0] = 4096
        C.cast(total, C.POINTER(C.c_size_t))[0] = 8192

    def test_report_and_cleanup(self):
        report = probe.measure(self.driver, 0, self.ram)
        self.assertEqual(report["ram"]["available_bytes"], 4096)
        self.assertEqual(report["vram_after_context"]["free_bytes"], 4096)
        count = len(probe.TRANSFER_BYTES) * (probe.WARMUP + probe.SAMPLES)
        self.assertEqual(self.driver.cuCtxSynchronize.call_count, count)
        self.assertEqual(self.driver.cuMemcpyHtoD_v2.call_count, count)
        self.assertEqual(self.buf.raw, b"Z" * 256)
        self.assertEqual([r["bytes"] for r in report["results"]], [128, 256])
        for row in report["results"]:
            self.assertEqual(len(row["elapsed_ns"]), probe.SAMPLES)
            self.assertGreaterEqual(row["p95_ms_nearest_rank"], row["median_ms"])
        self.assertEqual([c[0] for c in self.driver.mock_calls[-3:]],
                         ["cuMemFree_v2", "cuMemFreeHost", "cuCtxDestroy_v2"])

    def test_device_allocation_failure_releases_only_owned_resources(self):
        self.driver.cuMemAlloc_v2.side_effect = RuntimeError("device allocation failed")
        with self.assertRaisesRegex(RuntimeError, "allocation"):
            probe.measure(self.driver, 0, self.ram)
        self.driver.cuMemFree_v2.assert_not_called()
        self.driver.cuMemFreeHost.assert_called_once()
        self.driver.cuCtxDestroy_v2.assert_called_once()

    def test_low_ram_still_reports_capacity(self):
        report = probe.measure(self.driver, 0, lambda: {"available_bytes": 255, "total_bytes": 8192})
        self.assertIn("skipped", report)
        self.assertEqual(report["results"], [])
        self.driver.cuMemHostAlloc.assert_not_called()
        self.driver.cuCtxDestroy_v2.assert_called_once()


if __name__ == "__main__":
    unittest.main()

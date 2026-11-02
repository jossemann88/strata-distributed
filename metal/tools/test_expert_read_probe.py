import os
import tempfile
import unittest

from expert_read_probe import expert_slices, read_span


class ExpertReadTest(unittest.TestCase):
    def setUp(self):
        self.layer = [dict(abs=off, bytes=size * 512, dims=(2560, 640, 512))
                      for off, size in [(256, 32), (16640, 32), (33024, 64)]] + [32, 32, 64]

    def test_last_expert_slices(self):
        self.assertEqual(expert_slices(self.layer, 511, 65792),
                         [(16608, 32), (32992, 32), (65728, 64)])
        with self.assertRaises(ValueError):
            expert_slices(self.layer, 511, 65791)

    @unittest.skipUnless(hasattr(os, "pread"), "requires POSIX positioned reads")
    def test_truncated_read(self):
        with tempfile.TemporaryFile() as data:
            os.write(data.fileno(), b"abcdefgh")
            read_span(data.fileno(), 3, 5)
            with self.assertRaises(OSError):
                read_span(data.fileno(), 3, 6)


if __name__ == "__main__":
    unittest.main()

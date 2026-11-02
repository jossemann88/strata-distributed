import unittest

from layer_wait import summarize


class LayerWaitTest(unittest.TestCase):
    def setUp(self):
        self.wifi = dict(loopback=False, results=[dict(request_payload_bytes=10240,
                         reply_payload_bytes=10240, rtt_ns=[5000000, 9000000])])
        self.rows = [f'strata metal layer_wait: T=1 layer={l} windows=10 waited=1 wait_ms={l}'
                     for l in range(48)]

    def test_includes_hits_in_denominator_and_keeps_window_shapes_separate(self):
        report = summarize(self.rows + [self.rows[0].replace('T=1', 'T=2')], self.wifi)
        self.assertEqual(report['layers'][0], dict(layer=47, mean_wait_ms=4.7))
        self.assertEqual(report['mean_rtt_ms'], 7)
        self.assertAlmostEqual(report['total_wait_ms_per_window'], sum(range(48)) / 10)

    def test_rejects_incomplete_or_concatenated_logs(self):
        for lines in [self.rows[1:], self.rows * 2]:
            with self.assertRaises(ValueError):
                summarize(lines, self.wifi)

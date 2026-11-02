import concurrent.futures
import socket
import time
import unittest
from unittest.mock import Mock

import lan_probe as probe


class ProbeTests(unittest.TestCase):
    def test_scope_and_limits(self):
        self.assertEqual(probe.lan_ip("192.168.1.12"), "192.168.1.12")
        for address in ("0.0.0.0", "8.8.8.8", "255.255.255.255"):
            with self.assertRaises(ValueError):
                probe.lan_ip(address)
        for sizes, samples, warmup in (("-1", 2, 0), ("0", 0, 0), ("0", 1, -1),
                                       ("0", 4000, 1), ("0:1048576", 128, 0),
                                       ("1,1:1", 1, 0), ("0:1048577", 1, 0)):
            with self.assertRaises(ValueError):
                probe.validate_run(probe.parse_sizes(sizes), samples, warmup)

    def test_framing_and_aborted_frames(self):
        conn = Mock()
        conn.recv.side_effect = [b"a", b"bc"]
        self.assertEqual(probe.read_exact(conn, 3, time.monotonic() + 2), b"abc")
        conn.recv.side_effect = [b"a", b""]
        with self.assertRaises(EOFError):
            probe.read_exact(conn, 3, time.monotonic() + 2, allow_eof=True)
        for pair in ((probe.MAX_PAYLOAD + 1, 0), (0, probe.MAX_PAYLOAD + 1)):
            conn.recv.side_effect = [probe.REQUEST_HEADER.pack(*pair)]
            with self.assertRaises(ValueError):
                probe.echo_session(conn)
        conn.sendall.assert_not_called()

    def round_trip(self, handler, pairs, samples=1, warmup=0):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            listener.settimeout(5)

            def run_server():
                conn, _ = listener.accept()
                with conn:
                    handler(conn)

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                server = pool.submit(run_server)
                try:
                    return probe.measure("127.0.0.1", listener.getsockname()[1], pairs, samples, warmup)
                finally:
                    server.result(timeout=5)

    def test_persistent_tcp_echo(self):
        pairs = probe.parse_sizes("0,10240,10240:102400,7:3")
        report = self.round_trip(probe.echo_session, pairs, 8, 2)
        self.assertTrue(report["loopback"])
        self.assertEqual((report["request_header_bytes"], report["reply_header_bytes"]), (8, 4))
        self.assertEqual([(row["request_payload_bytes"], row["reply_payload_bytes"])
                          for row in report["results"]], pairs)
        for result in report["results"]:
            values = sorted(result["rtt_ns"])
            self.assertEqual(len(values), 8)
            self.assertEqual(result["p50_ms"], values[3] / 1e6)
            self.assertEqual(result["p95_ms"], values[7] / 1e6)

    def test_fragmented_tcp_reads(self):
        def fragment(conn):
            wrapped = Mock(wraps=conn)
            wrapped.recv.side_effect = lambda length: conn.recv(min(length, 2))
            probe.echo_session(wrapped)

        self.round_trip(fragment, [(7, 15)])

    def test_corrupted_reply(self):
        def corrupt(conn):
            probe.read_exact(conn, probe.REQUEST_HEADER.size + 1, time.monotonic() + 2)
            conn.sendall(probe.REPLY_HEADER.pack(1) + b"z")

        with self.assertRaisesRegex(ValueError, "unexpected echo response"):
            self.round_trip(corrupt, [(1, 1)])


if __name__ == "__main__":
    unittest.main()

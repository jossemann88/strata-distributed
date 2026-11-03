import concurrent.futures
import socket
import tempfile
import unittest
from pathlib import Path

import fixture_replay as fixture


class FixtureReplayTests(unittest.TestCase):
    identity = bytes(range(32))
    body = b"0" * fixture.STATE_BYTES

    def test_exact_match_across_processes(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            for layer, offset in (("layer04", 10), ("layer10", 20)):
                for index, (name, size) in enumerate(fixture.PARTS):
                    (directory / f"{layer}-{name}.f32").write_bytes(bytes([offset + index]) * size)
            report = fixture.replay(directory)
            self.assertTrue(report["exact_match"])
            self.assertEqual(report["request_bytes"], 51216)
            self.assertEqual(report["response_bytes"], 51216)
            self.assertNotEqual(report["input_sha256"], report["output_sha256"])
            (directory / "layer04-inj.f32").write_bytes(b"short")
            with self.assertRaisesRegex(ValueError, "exactly 16"):
                fixture.replay(directory)

    def check_rejected(self, wire, error, message):
        sender, receiver = socket.socketpair()
        with sender, receiver:
            def send_frame():
                sender.settimeout(1)
                sender.sendall(wire)
                sender.shutdown(socket.SHUT_WR)
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                sent = pool.submit(send_frame)
                with self.assertRaisesRegex(error, message):
                    fixture.receive(receiver, b"Q", self.identity, self.body, timeout=1)
                sent.result(timeout=1)

    def test_corrupt_request(self):
        frame = fixture.header(b"Q", self.identity) + self.body[:-1] + b"!"
        self.check_rejected(frame, ValueError, "payload")

    def test_header_and_identity(self):
        valid = fixture.header(b"Q", self.identity)
        for frame in (fixture.header(b"R", self.identity), fixture.header(b"Q", b"x" * 32),
                      valid[:-4] + bytes.fromhex("ffffffff")):
            with self.subTest(frame=frame):
                self.check_rejected(frame, ValueError, "frame header")
        self.check_rejected(valid + self.body + b"x", ValueError, "trailing")

    def test_timeout(self):
        sender, receiver = socket.socketpair()
        with sender, receiver:
            sender.sendall(b"S")
            with self.assertRaises(TimeoutError):
                fixture.receive(receiver, b"Q", self.identity, self.body, timeout=0.05)

    def test_disconnect(self):
        self.check_rejected(fixture.header(b"Q", self.identity) + b"0", EOFError, "disconnected")


if __name__ == "__main__":
    unittest.main()

"""Replay a captured Qwen Metal block in a separate localhost process. No inference."""

import argparse
import hashlib
import json
import multiprocessing
import socket
import struct
import sys
import time
from pathlib import Path


PARTS = (("residual", 4 * 2560 * 4), ("bo", 2560 * 4), ("inj", 4 * 4))
STATE_BYTES = sum(size for _, size in PARTS)
HEADER = struct.Struct("!4scBBBI32sI")
TIMEOUT = 5.0


def read_boundary(directory, prefix):
    parts = []
    for name, size in PARTS:
        path = directory / f"{prefix}-{name}.f32"
        with path.open("rb") as source:
            data = source.read(size + 1)
        if len(data) != size:
            raise ValueError(f"{path.name} must contain exactly {size} bytes")
        parts.append(data)
    return b"".join(parts)


def load_fixture(directory):
    before = read_boundary(directory, "layer04")
    after = read_boundary(directory, "layer10")
    # Identify the capture and its layout. The peer independently reads the same fixture.
    identity = hashlib.sha256(b"qwen-metal-4:10/t1/p0/f32le/residual+bo+inj/v1\0"
                              + before + after).digest()
    return before, after, identity


def header(kind, identity):
    return HEADER.pack(b"SBR1", kind, 4, 10, 1, 0, identity, STATE_BYTES)


def read_exact(conn, count, deadline):
    buf = bytearray()
    while len(buf) < count:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("fixture frame deadline exceeded")
        conn.settimeout(remaining)
        chunk = conn.recv(count - len(buf))
        if not chunk:
            raise EOFError("peer disconnected before complete fixture frame")
        buf.extend(chunk)
    return buf


def receive(conn, kind, identity, expected, timeout=TIMEOUT):
    deadline = time.monotonic() + timeout
    if read_exact(conn, HEADER.size, deadline) != header(kind, identity):
        raise ValueError("wrong frame header or fixture identity")
    if read_exact(conn, STATE_BYTES, deadline) != expected:
        raise ValueError("frame payload differs from the captured bytes")
    # One frame per direction. Require orderly EOF, rejecting trailing data.
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("fixture frame deadline exceeded")
    conn.settimeout(remaining)
    if conn.recv(1):
        raise ValueError("unexpected trailing data")


def send(conn, kind, identity, payload):
    conn.settimeout(TIMEOUT)
    conn.sendall(header(kind, identity) + payload)
    conn.shutdown(socket.SHUT_WR)


def serve_fixture(listener, directory):
    with listener:
        before, after, identity = load_fixture(directory)
        listener.settimeout(TIMEOUT)
        conn, _ = listener.accept()
        with conn:
            receive(conn, b"Q", identity, before)
            send(conn, b"R", identity, after)


def replay(directory):
    before, after, identity = load_fixture(directory)
    # Bind before spawning so there is no race when selecting an ephemeral port.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        address = listener.getsockname()
        peer = multiprocessing.get_context("spawn").Process(target=serve_fixture, args=(listener, directory))
        peer.start()
    try:
        with socket.create_connection(address, timeout=TIMEOUT) as conn:
            send(conn, b"Q", identity, before)
            receive(conn, b"R", identity, after)
        peer.join(TIMEOUT)
        if peer.exitcode != 0:
            raise RuntimeError("fixture peer did not finish successfully")
    finally:
        if peer.is_alive():
            peer.terminate()
        peer.join(TIMEOUT)
        peer.close()
    return {
        "method": "localhost_fixture_replay", "exact_match": True,
        "start_layer": 4, "end_layer_exclusive": 10, "rows": 1, "position": 0,
        "request_bytes": len(before), "response_bytes": len(after),
        "header_bytes_each_way": HEADER.size, "fixture_sha256": identity.hex(),
        "input_sha256": hashlib.sha256(before).hexdigest(),
        "output_sha256": hashlib.sha256(after).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True, help="contains layer04 and layer10 .f32 files")
    args = parser.parse_args()
    try:
        print(json.dumps(replay(args.capture_dir), indent=2))
    except (EOFError, ValueError, RuntimeError, OSError) as err:
        sys.exit(f"fixture replay failed: {err}")


if __name__ == "__main__":
    main()

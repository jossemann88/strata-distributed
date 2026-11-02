"""Measure sequential TCP echo round trips on a trusted IPv4 LAN (Python 3.8+)."""

import argparse
import ipaddress
import json
import math
import platform
import socket
import struct
import sys
import time


REQUEST_HEADER = struct.Struct("!II")  # request and reply payload lengths
REPLY_HEADER = struct.Struct("!I")
MAX_PAYLOAD = 1024 * 1024
MAX_FRAMES = 4000
MAX_BYTES = 128 * 1024 * 1024  # total application traffic in both directions
IO_TIMEOUT = 5
SESSION_SECONDS = 300
LAN_RANGES = tuple(map(ipaddress.ip_network, (
    "127.0.0.0/8", "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "169.254.0.0/16",
)))


def lan_ip(value):
    address = ipaddress.IPv4Address(value)
    if not any(address in network for network in LAN_RANGES):
        raise ValueError("use a specific private LAN, link-local, or loopback IPv4 address")
    return str(address)


def parse_sizes(value):
    pairs = []
    for item in value.split(","):
        request, separator, reply = item.partition(":")
        pairs.append((int(request), int(reply if separator else request)))
    return pairs


def frame_bytes(request, reply):
    return REQUEST_HEADER.size + request + REPLY_HEADER.size + reply


def validate_run(sizes, samples, warmup):
    if not sizes or any(min(pair) < 0 or max(pair) > MAX_PAYLOAD for pair in sizes):
        raise ValueError(f"each size must be between 0 and {MAX_PAYLOAD} bytes")
    if len(set(sizes)) != len(sizes):
        raise ValueError("payload sizes must be unique")
    if samples < 1 or warmup < 0:
        raise ValueError("samples must be positive and warmup must be nonnegative")
    count = samples + warmup
    if count * len(sizes) > MAX_FRAMES or count * sum(frame_bytes(*s) for s in sizes) > MAX_BYTES:
        raise ValueError("requested run exceeds the frame or byte limit")


def read_exact(conn, length, deadline, allow_eof=False):
    data = bytearray()
    while len(data) < length:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("receive deadline expired")
        conn.settimeout(min(IO_TIMEOUT, remaining))
        chunk = conn.recv(length - len(data))
        if not chunk:
            if allow_eof and not data:
                return None
            raise EOFError("peer closed a partial frame")
        data.extend(chunk)
    return data


def echo_session(conn):
    deadline = time.monotonic() + SESSION_SECONDS
    total = 0
    for _ in range(MAX_FRAMES):
        header = read_exact(conn, REQUEST_HEADER.size, deadline, allow_eof=True)
        if header is None:
            return
        request_size, reply_size = REQUEST_HEADER.unpack(header)
        total += frame_bytes(request_size, reply_size)
        if max(request_size, reply_size) > MAX_PAYLOAD or total > MAX_BYTES:
            raise ValueError("peer exceeded the payload or byte limit")
        payload = read_exact(conn, request_size, deadline)
        # Truncate or pad the echo to emulate the requested return shape.
        reply = payload[:reply_size].ljust(reply_size, b"x")
        conn.sendall(REPLY_HEADER.pack(reply_size) + reply)


def serve(bind, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind((bind, port))
        listener.listen(1)
        listener.settimeout(120)
        print(f"Waiting for one client on {bind}:{port} (120 seconds)", file=sys.stderr)
        conn, _ = listener.accept()
        with conn:
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            echo_session(conn)


def percentile(sorted_values, fraction):
    return sorted_values[math.ceil(len(sorted_values) * fraction) - 1]


def measure(host, port, sizes, samples, warmup):
    validate_run(sizes, samples, warmup)
    results = []
    with socket.create_connection((host, port), timeout=IO_TIMEOUT) as conn:
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        deadline = time.monotonic() + SESSION_SECONDS
        for request_size, reply_size in sizes:
            packet = REQUEST_HEADER.pack(request_size, reply_size) + b"x" * request_size
            expected = REPLY_HEADER.pack(reply_size) + b"x" * reply_size
            durations = []
            for index in range(samples + warmup):
                start = time.perf_counter_ns()
                conn.sendall(packet)
                reply = read_exact(conn, len(expected), deadline)
                elapsed = time.perf_counter_ns() - start
                if reply != expected:
                    raise ValueError("unexpected echo response")
                if index >= warmup:
                    durations.append(elapsed)
            ordered = sorted(durations)
            results.append({
                "request_payload_bytes": request_size,
                "reply_payload_bytes": reply_size,
                "rtt_ns": durations,
                "p50_ms": percentile(ordered, 0.50) / 1e6,
                "p95_ms": percentile(ordered, 0.95) / 1e6,
            })
    return {
        "schema_version": 1,
        "method": "persistent_tcp_echo_sequential",
        "percentile_method": "nearest_rank",
        "request_header_bytes": REQUEST_HEADER.size,
        "reply_header_bytes": REPLY_HEADER.size,
        "target_ipv4": host,
        "target_port": port,
        "loopback": ipaddress.IPv4Address(host).is_loopback,
        "client_os": platform.system(),
        "client_python": platform.python_version(),
        "samples_per_size": samples,
        "warmup_per_size": warmup,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    server = commands.add_parser("serve", help="serve exactly one measurement connection")
    server.add_argument("--bind", default="127.0.0.1", help="local private IPv4 address")
    client = commands.add_parser("measure", help="emit a JSON report to stdout")
    client.add_argument("--host", required=True, help="server's private IPv4 address")
    client.add_argument("--sizes", default="0,10240,10240:102400",
                        help="comma-separated payload sizes in bytes, N or REQUEST:REPLY")
    client.add_argument("--samples", type=int, default=100)
    client.add_argument("--warmup", type=int, default=10)
    for command in (server, client):
        command.add_argument("--port", type=int, default=47321)
    args = parser.parse_args()
    try:
        if not 1 <= args.port <= 65535:
            raise ValueError("port must be 1 through 65535")
        if args.command == "serve":
            serve(lan_ip(args.bind), args.port)
        else:
            sizes = parse_sizes(args.sizes)
            report = measure(lan_ip(args.host), args.port, sizes, args.samples, args.warmup)
            print(json.dumps(report, indent=2))
    except (OSError, ValueError, EOFError) as error:
        parser.exit(1, f"probe failed: {error}\n")


if __name__ == "__main__":
    main()

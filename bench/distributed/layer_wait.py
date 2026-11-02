"""Rank streamed Metal layers by observed host wait per single-row verifier window."""
import argparse
import json
import math
import re
from pathlib import Path
from statistics import mean

PREFIX = "strata metal layer_wait:"
ROW = re.compile(r"T=(\d+) layer=(\d+) windows=(\d+) waited=(\d+) wait_ms=([\d.]+)")


def summarize(lines, wifi):
    layers = {}
    for line in lines:
        if not line.startswith(PREFIX):
            continue
        match = ROW.fullmatch(line[len(PREFIX):].strip())
        if not match:
            raise ValueError("malformed layer_wait line")
        t, layer, windows, waited = map(int, match.groups()[:4])
        if t != 1:
            continue
        wait_ms = float(match[5])
        if layer in layers or not (0 <= layer < 48 and 0 <= waited <= windows and windows):
            raise ValueError("expected one valid T=1 row per layer from a single process")
        if not math.isfinite(wait_ms) or (wait_ms and not waited):
            raise ValueError("invalid wait total")
        layers[layer] = (windows, wait_ms / windows)
    if len(layers) != 48 or len({windows for windows, _ in layers.values()}) != 1:
        raise ValueError("need a complete 48-layer T=1 summary with equal window counts")
    if wifi['loopback']:
        raise ValueError("provide a measurement between the two machines")
    rtts = next((row['rtt_ns'] for row in wifi['results']
                 if (row['request_payload_bytes'], row['reply_payload_bytes']) == (10240, 10240)), None)
    if not rtts or any(not math.isfinite(ns) or ns <= 0 for ns in rtts):
        raise ValueError("need positive RTT samples for a 10240-byte request and reply")
    return dict(windows=layers[0][0], mean_rtt_ms=mean(rtts) / 1e6,
                total_wait_ms_per_window=sum(ms for _, ms in layers.values()),
                layers=[dict(layer=l, mean_wait_ms=ms) for l, (_, ms) in
                        sorted(layers.items(), key=lambda item: -item[1][1])])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('engine_log', type=Path)
    parser.add_argument('wifi_json', type=Path)
    args = parser.parse_args()
    try:
        with args.engine_log.open() as log:
            print(json.dumps(summarize(log, json.loads(args.wifi_json.read_text())), indent=2))
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f'{error}\n')

"""Screen interior blocks after the layer-0 PLE using GGUF weight bytes and measured waits."""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "metal/tools"))
from metal_repack import Reader


def candidates(shard, report, budgets):
    reader = Reader(shard)
    sizes = {}
    with reader.f:
        for tensor in reader.tensors:
            match = re.match(r"blk\.(\d+)\.", tensor["name"])
            if match:
                layer = int(match[1])
                sizes[layer] = sizes.get(layer, 0) + tensor["bytes"]
    waits = {row["layer"]: row["mean_wait_ms"] for row in report["layers"]}
    if report["window_rows"] != 1 or set(waits) != set(range(48)) or not waits.keys() <= sizes.keys():
        raise ValueError("expected a 48-layer single-row report and matching first GGUF shard")
    for budget in budgets:
        choices = []
        for start in range(2, 47):
            size = 0
            wait = 0.0
            for end in range(start, 47):
                size += sizes[end]
                if size > budget * 2**30:
                    break
                wait += waits[end]
                choices.append((wait, start, end, size))
        if choices:
            wait, start, end, size = max(choices)
            yield dict(weight_budget_GiB=budget, first=start, last=end,
                       gguf_weight_bytes=size, observed_wait_ms=wait)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shard", type=Path)
    parser.add_argument("wait_report", type=Path)
    parser.add_argument("--budgets", nargs="+", type=float, default=[4, 6, 8, 10])
    args = parser.parse_args()
    print(json.dumps(list(candidates(args.shard, json.loads(args.wait_report.read_text()), args.budgets)), indent=2))

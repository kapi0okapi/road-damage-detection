"""Latency of the ONNX Runtime detector (CPU by default), as deployed in the API.

rdd-benchmark --model models/road_damage.onnx --runs 50
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from road_damage.detector import OnnxDetector


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--width", type=int, default=3650, help="Norwegian RDD photos are ~3650 x 2044")
    parser.add_argument("--height", type=int, default=2044)
    parser.add_argument("--providers", nargs="+", default=["CPUExecutionProvider"])
    args = parser.parse_args(argv)

    detector = OnnxDetector(args.model, providers=args.providers)
    image = np.random.default_rng(0).integers(0, 255, (args.height, args.width, 3), dtype=np.uint8)
    report = {}
    for tiles in (1, 2):
        detector(image, tiles=(tiles, tiles))  # warm-up
        times = []
        for _ in range(args.runs):
            started = time.perf_counter()
            detector(image, tiles=(tiles, tiles))
            times.append(time.perf_counter() - started)
        report[f"tiles_{tiles}x{tiles}"] = {
            "mean_ms": 1000 * float(np.mean(times)),
            "p95_ms": 1000 * float(np.percentile(times, 95)),
        }
    print(json.dumps({"model": str(args.model), "providers": args.providers, **report}, indent=2))


if __name__ == "__main__":
    main()

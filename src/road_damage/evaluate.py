"""Cross-country evaluation: every model on every country's held-out test block.

    rdd-eval --data-dir $DATA/rdd2022_yolo/data --out $RUNS/cross_country.json \\
        --models all=$RUNS/all_yolo26s/weights/best.pt no_norway=$RUNS/no_norway_yolo26s/weights/best.pt

Prints a markdown table of mAP50 (rows: models, columns: test countries).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--models", nargs="+", required=True, metavar="NAME=WEIGHTS")
    parser.add_argument("--data-dir", type=Path, required=True, help="folder with test_<Country>.yaml")
    parser.add_argument(
        "--countries", nargs="+", default=["Norway", "Czech", "Japan", "India", "United_States"]
    )
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--out", type=Path, default=Path("cross_country.json"))
    parser.add_argument(
        "--nms-free",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="score YOLO26's one-to-one head (the exported ONNX path) instead of the one-to-many head + NMS",
    )
    args = parser.parse_args(argv)

    from ultralytics import YOLO

    results: dict[str, dict] = {}
    for spec in args.models:
        name, weights = spec.split("=", 1)
        model = YOLO(os.path.expandvars(weights))
        results[name] = {}
        for country in args.countries:
            data = Path(os.path.expandvars(str(args.data_dir))) / f"test_{country}.yaml"
            if not data.exists():
                continue
            m = model.val(
                data=str(data),
                imgsz=args.imgsz,
                split="val",
                plots=False,
                verbose=False,
                nms=False if args.nms_free else None,
            )
            results[name][country] = {
                "map50": float(m.box.map50),
                "map50_95": float(m.box.map),
                "per_class_map50_95": [float(v) for v in m.box.maps],
            }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))

    countries = [c for c in args.countries if any(c in r for r in results.values())]
    print("| model | " + " | ".join(countries) + " |")
    print("|---" * (len(countries) + 1) + "|")
    for name, row in results.items():
        print(
            f"| {name} | "
            + " | ".join(f"{row[c]['map50']:.3f}" if c in row else "–" for c in countries)
            + " |"
        )


if __name__ == "__main__":
    main()

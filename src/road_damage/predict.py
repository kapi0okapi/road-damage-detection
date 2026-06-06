"""Detect road damage in photos with the exported ONNX model.

rdd-predict --model models/road_damage.onnx --source photos/ --out predictions/ --tiles 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from road_damage.detector import OnnxDetector, draw

SUFFIXES = {".jpg", ".jpeg", ".png"}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("predictions"))
    parser.add_argument("--tiles", type=int, default=1, help="n x n tiles plus the full frame")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args(argv)

    detector = OnnxDetector(args.model, conf=args.conf)
    paths = (
        sorted(p for p in args.source.iterdir() if p.suffix.lower() in SUFFIXES)
        if args.source.is_dir()
        else [args.source]
    )
    args.out.mkdir(parents=True, exist_ok=True)
    report = {}
    for path in paths:
        bgr = cv2.imread(str(path))
        detections = detector(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), tiles=(args.tiles, args.tiles))
        cv2.imwrite(str(args.out / path.name), draw(bgr, detections))
        report[path.name] = [d.as_dict() for d in detections]
    (args.out / "detections.json").write_text(json.dumps(report, indent=2))
    print(f"{len(paths)} images, {sum(map(len, report.values()))} detections -> {args.out}")


if __name__ == "__main__":
    main()

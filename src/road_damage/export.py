"""Export a trained model to ONNX and check the ONNX Runtime detector against Ultralytics.

rdd-export --weights $RUNS/all_yolo26s/weights/best.pt --imgsz 1024 --out models/road_damage.onnx
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area = lambda x: np.prod(x[:, 2:] - x[:, :2], axis=1)  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--weights", required=True)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--out", type=Path, default=Path("models/road_damage.onnx"))
    parser.add_argument("--check-image", type=Path, help="image to compare ONNX and PyTorch detections on")
    args = parser.parse_args(argv)

    from ultralytics import YOLO

    from road_damage.detector import OnnxDetector

    model = YOLO(args.weights)
    # nms=False selects YOLO26's end-to-end head: output (1, 300, 6), no NMS in the graph or after it.
    # Ultralytics 8.4 would otherwise export the one-to-many head, (1, 4 + classes, anchors).
    exported = Path(model.export(format="onnx", imgsz=args.imgsz, dynamic=False, simplify=True, nms=False))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(exported, args.out)
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")

    if args.check_image:
        bgr = cv2.imread(str(args.check_image))
        ours = np.array(
            [d.box for d in OnnxDetector(args.out, conf=0.25)(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))]
        )
        # PyTorch letterboxes to the smallest stride multiple (e.g. 1024 x 768), the ONNX graph is
        # square, so boxes differ slightly; the ONNX file run by Ultralytics must match exactly.
        for name, ref_model, iou in (
            ("PyTorch", model, 0.9),
            ("Ultralytics ONNX", YOLO(str(args.out)), 0.99),
        ):
            result = ref_model.predict(bgr, imgsz=args.imgsz, conf=0.25, nms=False, verbose=False)[0]
            reference = result.boxes.xyxy.cpu().numpy()
            matched = (
                (box_iou(reference, ours).max(1) > iou).mean()
                if len(reference) and len(ours)
                else float(len(reference) == len(ours))
            )
            print(
                f"{name} {len(reference)} boxes, ours {len(ours)} boxes, {matched:.0%} matched at IoU>{iou}"
            )


if __name__ == "__main__":
    main()

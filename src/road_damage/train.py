"""Train or fine-tune a YOLO detector on RDD2022 (Ultralytics), resuming automatically.

    rdd-train --data $DATA/rdd2022_yolo/data/all.yaml --model yolo26s.pt --name all_yolo26s
    rdd-train --data $DATA/rdd2022_yolo/data/norway_25.yaml --model $RUNS/no_norway_yolo26s/weights/best.pt \\
        --name finetune_norway_25 --epochs 40 --lr0 0.002

If ``<project>/<name>/weights/last.pt`` exists, training resumes from it, so a job stopped by
the Slurm time limit continues when it is resubmitted.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", required=True)
    parser.add_argument(
        "--model", default="yolo26s.pt", help="pretrained weights or a checkpoint to fine-tune"
    )
    parser.add_argument("--name", required=True)
    parser.add_argument("--project", default=os.environ.get("RUNS", "runs"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="")
    parser.add_argument(
        "--nms-free",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="validate and pick best.pt with YOLO26's one-to-one head, as deployed (no-op for NMS models)",
    )
    args = parser.parse_args(argv)

    from ultralytics import YOLO

    project = Path(os.path.expandvars(args.project)).resolve()
    last = project / args.name / "weights" / "last.pt"
    if last.exists():
        print(f"resuming from {last}")
        YOLO(str(last)).train(resume=True)
        return
    YOLO(os.path.expandvars(args.model)).train(
        data=os.path.expandvars(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        workers=args.workers,
        device=args.device or None,
        project=str(project),
        name=args.name,
        exist_ok=True,
        cos_lr=True,
        patience=25,
        close_mosaic=10,
        plots=True,
        nms=False if args.nms_free else None,  # Ultralytics 8.4 validates the NMS head unless told otherwise
    )


if __name__ == "__main__":
    main()

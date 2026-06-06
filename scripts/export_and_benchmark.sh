#!/bin/bash
# ONNX export (+ parity check against PyTorch), ONNX Runtime CPU latency and a TensorRT FP16 run.
set -euo pipefail
WEIGHTS=$1
IMAGE=$(head -n 1 "$DATA/rdd2022_yolo/Norway_test.txt")
python -m road_damage.export --weights "$WEIGHTS" --imgsz 1024 --out "$RUNS/export/road_damage.onnx" \
    --check-image "$DATA/rdd2022_yolo/${IMAGE#./}"
python -m road_damage.benchmark --model "$RUNS/export/road_damage.onnx" --runs 30
yolo benchmark model="$WEIGHTS" data="$DATA/rdd2022_yolo/data/test_Norway.yaml" imgsz=1024 format=engine quantize=16 device=0

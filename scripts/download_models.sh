#!/bin/bash
# Pre-fetch YOLO weights and plot fonts on a login node; the jobs run with YOLO_OFFLINE=1.
set -euo pipefail
python - <<'PY'
from ultralytics import YOLO
from ultralytics.utils.checks import check_font
for name in ("yolo26n.pt", "yolo26s.pt", "yolo26m.pt"):
    YOLO(name)
check_font("Arial.ttf")
check_font("Arial.Unicode.ttf")
PY

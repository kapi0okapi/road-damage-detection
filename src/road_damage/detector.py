"""ONNX Runtime detector: no PyTorch or Ultralytics at inference time.

Handles the two YOLO export layouts: classic ``(1, 4 + classes, anchors)`` that needs NMS, and
end-to-end ``(1, max_det, 6)`` (YOLO26 / NMS-free). Large photos can be processed in
overlapping tiles: Norwegian RDD images are ~3.6k px wide, and thin cracks vanish when the
whole frame is shrunk to 1024 px.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Detection:
    box: tuple[float, float, float, float]  # x0, y0, x1, y1 in source pixels
    score: float
    class_id: int
    label: str

    def as_dict(self) -> dict:
        return {
            "box": [round(v, 1) for v in self.box],
            "score": round(self.score, 4),
            "class_id": self.class_id,
            "label": self.label,
        }


def letterbox(image: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize keeping aspect ratio and pad to size x size (grey 114, as in YOLO training)."""
    h, w = image.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = round(h * scale), round(w * scale)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[top : top + nh, left : left + nw] = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return canvas, scale, (left, top)


def nms(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, iou: float) -> np.ndarray:
    """Class-aware NMS (offsetting boxes per class) returning kept indices."""
    if len(boxes) == 0:
        return np.zeros(0, int)
    offset = classes[:, None] * (boxes.max() + 1)
    shifted = boxes + offset
    xywh = np.c_[shifted[:, :2], shifted[:, 2:] - shifted[:, :2]]
    keep = cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(), 0.0, iou)
    return np.asarray(keep, dtype=int).reshape(-1)


class OnnxDetector:
    def __init__(
        self, model_path: str | Path, conf: float = 0.25, iou: float = 0.5, providers: list[str] | None = None
    ):
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(model_path), providers=providers or ["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0]
        self.size = int(self.input.shape[2]) if isinstance(self.input.shape[2], int) else 1024
        meta = self.session.get_modelmeta().custom_metadata_map
        names = ast.literal_eval(meta["names"]) if "names" in meta else {}
        self.names = {int(k): v for k, v in names.items()}
        self.conf, self.iou = conf, iou
        # (1, max_det, 6): the model already removed duplicates, so NMS is only needed to merge tiles
        self.end2end = self.session.get_outputs()[0].shape[-1] == 6

    def _infer(self, rgb: np.ndarray, conf: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Boxes (x0, y0, x1, y1) in ``rgb`` pixels, scores and class ids for one image or tile."""
        canvas, scale, (left, top) = letterbox(rgb, self.size)
        x = canvas.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        out = self.session.run(None, {self.input.name: x})[0][0]
        if out.shape[-1] == 6:  # end-to-end export: (max_det, 6) = x0, y0, x1, y1, score, class
            boxes, scores, classes = out[:, :4], out[:, 4], out[:, 5].astype(int)
        else:  # (4 + classes, anchors), centre-size boxes
            out = out.T
            class_scores = out[:, 4:]
            classes = class_scores.argmax(1)
            scores = class_scores[np.arange(len(out)), classes]
            cx, cy, w, h = out[:, 0], out[:, 1], out[:, 2], out[:, 3]
            boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
        keep = scores >= conf
        boxes, scores, classes = boxes[keep], scores[keep], classes[keep]
        boxes = (boxes - [left, top, left, top]) / scale
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, rgb.shape[1])
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, rgb.shape[0])
        return boxes, scores, classes

    def __call__(
        self,
        rgb: np.ndarray,
        tiles: tuple[int, int] = (1, 1),
        overlap: float = 0.2,
        conf: float | None = None,
    ) -> list[Detection]:
        """Detections on the full image, optionally merged from a grid of overlapping tiles plus
        one pass over the whole frame (for large damage that spans tiles)."""
        conf = self.conf if conf is None else conf
        h, w = rgb.shape[:2]
        parts = [self._infer(rgb, conf)]
        cols, rows = tiles
        if cols * rows > 1:
            tw, th = int(w / (cols - (cols - 1) * overlap)), int(h / (rows - (rows - 1) * overlap))
            for r in range(rows):
                for c in range(cols):
                    x0 = min(int(c * tw * (1 - overlap)), w - tw)
                    y0 = min(int(r * th * (1 - overlap)), h - th)
                    boxes, scores, classes = self._infer(rgb[y0 : y0 + th, x0 : x0 + tw], conf)
                    parts.append((boxes + [x0, y0, x0, y0], scores, classes))
        boxes = np.concatenate([p[0] for p in parts])
        scores = np.concatenate([p[1] for p in parts])
        classes = np.concatenate([p[2] for p in parts])
        keep = (
            np.arange(len(boxes))
            if self.end2end and len(parts) == 1
            else nms(boxes, scores, classes, self.iou)
        )
        return [
            Detection(
                tuple(float(v) for v in boxes[i]),
                float(scores[i]),
                int(classes[i]),
                self.names.get(int(classes[i]), str(int(classes[i]))),
            )
            for i in keep
        ]


def draw(bgr: np.ndarray, detections: list[Detection]) -> np.ndarray:
    palette = [(56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255)]
    for d in detections:
        color = palette[d.class_id % len(palette)]
        x0, y0, x1, y1 = map(int, d.box)
        cv2.rectangle(bgr, (x0, y0), (x1, y1), color, 2)
        cv2.putText(
            bgr, f"{d.label} {d.score:.2f}", (x0, max(14, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )
    return bgr

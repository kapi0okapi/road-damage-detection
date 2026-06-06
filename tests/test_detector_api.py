import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from road_damage.detector import OnnxDetector, letterbox, nms

ROOT = Path(__file__).resolve().parents[1]
IMAGE = ROOT / "tests" / "data" / "road.jpg"


@pytest.fixture(scope="module")
def dummy_model(tmp_path_factory) -> Path:
    spec = importlib.util.spec_from_file_location(
        "make_dummy_model", ROOT / "scripts" / "make_dummy_model.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = tmp_path_factory.mktemp("model") / "road_damage.onnx"
    module.build(str(path))
    return path


def test_letterbox_geometry():
    canvas, scale, (left, top) = letterbox(np.zeros((360, 640, 3), np.uint8), 640)
    assert canvas.shape == (640, 640, 3) and scale == 1.0 and (left, top) == (0, 140)


def test_nms_is_class_aware():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [0, 0, 10, 10]], float)
    keep = nms(boxes, np.array([0.9, 0.8, 0.7]), np.array([0, 0, 1]), iou=0.5)
    assert sorted(keep.tolist()) == [0, 2]


def test_detector_maps_boxes_back_to_the_source_image(dummy_model):
    detector = OnnxDetector(dummy_model, conf=0.5)
    rgb = np.zeros((360, 640, 3), np.uint8)  # letterboxed with 140 px padding on top
    detections = sorted(detector(rgb), key=lambda d: d.class_id)
    assert [d.label for d in detections] == ["D00", "D40"]  # duplicate pothole removed by NMS
    pothole = detections[1]
    np.testing.assert_allclose(pothole.box, [270, 155, 370, 205], atol=1e-3)  # (320,320,100,50) - 140 px in y


def test_tiling_adds_no_duplicates_for_global_boxes(dummy_model):
    detector = OnnxDetector(dummy_model, conf=0.5)
    rgb = np.zeros((1280, 2560, 3), np.uint8)
    assert len(detector(rgb, tiles=(2, 2))) >= len(detector(rgb))


def test_api(dummy_model, monkeypatch):
    monkeypatch.setenv("RDD_MODEL", str(dummy_model))
    from road_damage.serve.app import app

    with TestClient(app) as client:
        health = client.get("/health").json()
        assert health["status"] == "ok" and health["classes"]["3"] == "D40"
        with IMAGE.open("rb") as f:
            result = client.post("/detect?tiles=2", files={"image": ("road.jpg", f, "image/jpeg")}).json()
        assert result["image_size"] == [640, 360] and len(result["detections"]) >= 2
        with IMAGE.open("rb") as f:
            response = client.post("/detect/annotated", files={"image": ("road.jpg", f, "image/jpeg")})
        assert response.headers["content-type"] == "image/jpeg"
        assert cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR) is not None
        bad = client.post("/detect", files={"image": ("x.jpg", b"not an image", "image/jpeg")})
        assert bad.status_code == 400

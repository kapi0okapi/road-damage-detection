"""Real YOLO26 export: the ONNX graph must use the NMS-free head and match Ultralytics' boxes.

Needs the ``train`` extra (Ultralytics) and yolo26n.pt (models/ or a 5 MB download); skipped
otherwise, e.g. in the lightweight CI job.
"""

import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

ultralytics = pytest.importorskip("ultralytics")

from road_damage import export  # noqa: E402
from road_damage.detector import OnnxDetector, letterbox  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BUS = Path(ultralytics.__file__).parent / "assets" / "bus.jpg"


@pytest.fixture(scope="module")
def weights(tmp_path_factory) -> Path:
    folder = tmp_path_factory.mktemp("yolo26")
    cached = ROOT / "models" / "yolo26n.pt"
    if cached.exists():
        return Path(shutil.copy(cached, folder / cached.name))
    from ultralytics.utils.downloads import attempt_download_asset

    try:
        return Path(attempt_download_asset(str(folder / "yolo26n.pt")))
    except Exception as err:  # offline
        pytest.skip(f"yolo26n.pt not available: {err}")


def test_export_is_nms_free_and_matches_pytorch(weights, tmp_path):
    out = tmp_path / "yolo26n.onnx"
    export.main(["--weights", str(weights), "--imgsz", "640", "--out", str(out)])
    detector = OnnxDetector(out, conf=0.25)
    assert detector.end2end
    assert detector.session.get_outputs()[0].shape == [1, 300, 6]

    # same letterboxed tensor through both graphs
    import torch
    from ultralytics import YOLO

    rgb = cv2.cvtColor(cv2.imread(str(BUS)), cv2.COLOR_BGR2RGB)
    canvas, _, _ = letterbox(rgb, 640)
    x = canvas.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    onnx_out = detector.session.run(None, {detector.input.name: x})[0][0]
    net = YOLO(weights).model.float().eval()
    net.end2end = True
    with torch.no_grad():
        torch_out = net(torch.from_numpy(x))
    torch_out = (torch_out[0] if isinstance(torch_out, list | tuple) else torch_out)[0].numpy()
    confident = torch_out[:, 4] > 0.25
    assert confident.sum() >= 4  # bus.jpg: a bus and several people
    np.testing.assert_allclose(onnx_out[confident], torch_out[confident], atol=0.05)

    # our decoding (letterbox, rescale, clip) == Ultralytics running the same ONNX file
    reference = YOLO(str(out), task="detect").predict(rgb[..., ::-1], imgsz=640, conf=0.25, verbose=False)
    ref_boxes = reference[0].boxes.xyxy.cpu().numpy()
    ours = detector(rgb)
    assert len(ours) == len(ref_boxes) == confident.sum()
    assert (export.box_iou(ref_boxes, np.array([d.box for d in ours])).max(1) > 0.99).all()

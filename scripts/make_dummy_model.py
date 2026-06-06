"""Write a tiny ONNX model with the YOLO detection output layout (1, 4 + classes, anchors).

It ignores the image content and always returns the same two boxes. That is enough to test
the decoding, NMS, the API and the Docker image without a trained network.

    python scripts/make_dummy_model.py models/road_damage.onnx
"""

import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

SIZE, ANCHORS, NAMES = 640, 8, {0: "D00", 1: "D10", 2: "D20", 3: "D40"}


def build(path: str) -> None:
    out = np.zeros((1, 4 + len(NAMES), ANCHORS), np.float32)
    out[0, :4, 0] = [320, 320, 100, 50]  # cx, cy, w, h of a pothole (class 3)
    out[0, 4 + 3, 0] = 0.9
    out[0, :4, 1] = [322, 321, 100, 50]  # near-duplicate: removed by NMS
    out[0, 4 + 3, 1] = 0.6
    out[0, :4, 2] = [100, 500, 40, 200]  # longitudinal crack (class 0)
    out[0, 4 + 0, 2] = 0.8
    nodes = [
        helper.make_node("ReduceSum", ["images"], ["total"], keepdims=0),
        helper.make_node("Mul", ["total", "zero"], ["nothing"]),
        helper.make_node("Add", ["boxes", "nothing"], ["output0"]),
    ]
    graph = helper.make_graph(
        nodes,
        "dummy_yolo",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, SIZE, SIZE])],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, list(out.shape))],
        initializer=[
            numpy_helper.from_array(out, "boxes"),
            numpy_helper.from_array(np.zeros((), np.float32), "zero"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    helper.set_model_props(model, {"names": str(NAMES)})
    onnx.save(model, path)


if __name__ == "__main__":
    from pathlib import Path

    Path(sys.argv[1]).parent.mkdir(parents=True, exist_ok=True)
    build(sys.argv[1])
    print(f"wrote {sys.argv[1]}")

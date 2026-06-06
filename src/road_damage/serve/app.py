"""HTTP API for road damage detection (FastAPI + ONNX Runtime, CPU).

RDD_MODEL=models/road_damage.onnx uvicorn road_damage.serve.app:app --port 8080
curl -F image=@photo.jpg "localhost:8080/detect?tiles=2"
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from typing import Annotated

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import Response

from road_damage import __version__
from road_damage.detector import OnnxDetector, draw

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
state: dict[str, OnnxDetector] = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    state["detector"] = OnnxDetector(
        os.environ.get("RDD_MODEL", "models/road_damage.onnx"), conf=float(os.environ.get("RDD_CONF", "0.25"))
    )
    yield
    state.clear()


app = FastAPI(title="Road damage detection", version=__version__, lifespan=lifespan)


async def read_image(upload: UploadFile) -> np.ndarray:
    data = await upload.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "image larger than 25 MB")
    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(400, "not a decodable image")
    return bgr


@app.get("/health")
def health() -> dict:
    detector = state["detector"]
    return {"status": "ok", "version": __version__, "input_size": detector.size, "classes": detector.names}


@app.post("/detect")
async def detect(
    image: Annotated[UploadFile, File()],
    tiles: Annotated[int, Query(ge=1, le=4)] = 1,
    conf: Annotated[float | None, Query(ge=0.0, le=1.0)] = None,
) -> dict:
    bgr = await read_image(image)
    started = time.perf_counter()
    detections = state["detector"](cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), tiles=(tiles, tiles), conf=conf)
    return {
        "image_size": [bgr.shape[1], bgr.shape[0]],
        "inference_ms": round(1000 * (time.perf_counter() - started), 1),
        "detections": [d.as_dict() for d in detections],
    }


@app.post("/detect/annotated", response_class=Response)
async def detect_annotated(
    image: Annotated[UploadFile, File()], tiles: Annotated[int, Query(ge=1, le=4)] = 1
) -> Response:
    bgr = await read_image(image)
    detections = state["detector"](cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), tiles=(tiles, tiles))
    ok, jpeg = cv2.imencode(".jpg", draw(bgr, detections), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return Response(content=jpeg.tobytes(), media_type="image/jpeg")

# Inference image: FastAPI + ONNX Runtime, no PyTorch (~300 MB instead of several GB).
# Build:  docker build -t road-damage-api .         (expects models/road_damage.onnx, see rdd-export)
# Run:    docker run -p 8080:8080 road-damage-api
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    RDD_MODEL=/app/models/road_damage.onnx

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 app

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[serve]"
COPY models/road_damage.onnx /app/models/road_damage.onnx

USER app
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health', timeout=4)"
CMD ["uvicorn", "road_damage.serve.app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]

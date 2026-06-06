.PHONY: install dev lint format test jobs serve docker

install:
	pip install -e ".[train,serve]"

dev:
	pip install -e ".[dev]"

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check --fix .

test:
	pytest -q

serve:    ## local API on :8080 (needs models/road_damage.onnx)
	uvicorn road_damage.serve.app:app --port 8080 --reload

docker:   ## build and run the inference container
	docker build -t road-damage-api . && docker run --rm -p 8080:8080 road-damage-api

jobs:
	python3 slurm/generate.py --all

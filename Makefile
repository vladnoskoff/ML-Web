PYTHON ?= python
PIP ?= pip
UVICORN ?= uvicorn

.PHONY: install train eda serve docker-build docker-up docker-down

install:
$(PIP) install -r requirements.txt

train:
$(PYTHON) ml/train_baseline.py

eda:
$(PYTHON) ml/eda.py

serve:
$(UVICORN) backend.app.main:app --reload

docker-build:
docker compose build

docker-up:
docker compose up --build

docker-down:
docker compose down

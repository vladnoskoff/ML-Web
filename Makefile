PYTHON ?= python
PIP ?= pip
UVICORN ?= uvicorn

.PHONY: install train eda serve docker-build docker-up docker-down feedback-export

install:
$(PIP) install -r requirements.txt

train:
$(PYTHON) ml/train_baseline.py

eda:
$(PYTHON) ml/eda.py

feedback-export:
$(PYTHON) ml/feedback_to_dataset.py

serve:
$(UVICORN) backend.app.main:app --reload

docker-build:
docker compose build

docker-up:
docker compose up --build

docker-down:
docker compose down

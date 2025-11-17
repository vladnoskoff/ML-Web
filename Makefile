PYTHON ?= python
PIP ?= pip
UVICORN ?= uvicorn

.PHONY: install train train-transformer eda evaluate serve docker-build docker-up docker-down feedback-export history-report test

install:
$(PIP) install -r requirements.txt

train:
$(PYTHON) ml/train_baseline.py

train-transformer:
$(PYTHON) ml/train_transformer.py

eda:
$(PYTHON) ml/eda.py

evaluate:
$(PYTHON) ml/evaluate.py

feedback-export:
$(PYTHON) ml/feedback_to_dataset.py

history-report:
$(PYTHON) ml/history_report.py

test:
$(PYTHON) -m unittest discover -s tests -p 'test_*.py'

serve:
$(UVICORN) backend.app.main:app --reload

docker-build:
docker compose build

docker-up:
docker compose up --build

docker-down:
docker compose down

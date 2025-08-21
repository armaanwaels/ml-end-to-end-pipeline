.PHONY: init venv deps dvc-push dvc-pull train api test fmt lint

init: ## one-time: install dev deps
	pip install -r requirements.txt

venv:
	python3 -m venv venv && . venv/bin/activate

deps:
	pip install -r requirements.txt

train:
	python -m src.train --params params.yaml

api:
	cd app && uvicorn main:app --host 0.0.0.0 --port 8000 --reload

fmt:
	ruff format .

lint:
	ruff check .

test:
	pytest -q

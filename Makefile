.PHONY: deps data pipeline metrics api test lint

deps:
	pip install -r requirements-dev.txt -r app/requirements.txt

data:
	python scripts/fetch_data.py

pipeline:
	dvc repro

metrics:
	dvc metrics show

api:
	cd app && uvicorn main:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest -q

lint:
	ruff check .

import importlib
import sys
from pathlib import Path

from fastapi.testclient import TestClient

APP_DIR = Path(__file__).resolve().parents[1] / "app"


def load_app(monkeypatch):
    # main.py loads model.joblib and scaler.joblib relative to its working directory.
    monkeypatch.chdir(APP_DIR)
    monkeypatch.syspath_prepend(str(APP_DIR))
    sys.modules.pop("main", None)
    return importlib.import_module("main").app


def test_predict_returns_label_and_probability(monkeypatch):
    client = TestClient(load_app(monkeypatch))
    resp = client.post("/predict", json={"features": [0.0] * 30})
    assert resp.status_code == 200
    body = resp.json()
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["fraud_probability"] <= 1.0

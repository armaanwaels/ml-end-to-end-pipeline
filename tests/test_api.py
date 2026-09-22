import importlib
import sys
from pathlib import Path

from fastapi.testclient import TestClient

APP_DIR = Path(__file__).resolve().parents[1] / "app"


def load_app(monkeypatch):
    # Imported from the repo root, not from app/, so a wrong working directory would fail here.
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
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

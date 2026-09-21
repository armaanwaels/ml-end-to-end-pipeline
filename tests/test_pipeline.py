import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def run(*args: str) -> None:
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def synthetic_csv(path: Path, n: int = 2000, fraud_rate: float = 0.05) -> None:
    rng = np.random.default_rng(0)
    y = (rng.random(n) < fraud_rate).astype(int)
    X = rng.normal(size=(n, 5)) + y[:, None] * 2.0
    df = pd.DataFrame(X, columns=[f"V{i}" for i in range(1, 6)])
    df["Class"] = y
    df.to_csv(path, index=False)


def test_preprocess_train_evaluate(tmp_path):
    raw, processed = tmp_path / "raw.csv", tmp_path / "processed.csv"
    model, scaler = tmp_path / "model.joblib", tmp_path / "scaler.joblib"
    train_metrics, eval_metrics = tmp_path / "train.json", tmp_path / "eval.json"
    synthetic_csv(raw)

    run("src/data.py", "--input", str(raw), "--output", str(processed))
    assert len(pd.read_csv(processed)) == 2000

    run("src/train.py", "--data", str(processed), "--model", str(model),
        "--scaler", str(scaler), "--metrics", str(train_metrics))
    run("src/evaluate.py", "--data", str(processed), "--model", str(model),
        "--scaler", str(scaler), "--metrics", str(eval_metrics))

    m = json.loads(eval_metrics.read_text())
    assert m["n_test"] == 400
    assert 0.0 <= m["precision"] <= 1.0 and 0.0 <= m["recall"] <= 1.0
    # Signal is strong in the synthetic data, so the model should find it.
    assert m["roc_auc"] > 0.9
    assert sum(map(sum, m["confusion_matrix"])) == 400
    # Train and evaluate score the same split, so shared metrics must agree.
    t = json.loads(train_metrics.read_text())
    assert all(m[k] == t[k] for k in t)

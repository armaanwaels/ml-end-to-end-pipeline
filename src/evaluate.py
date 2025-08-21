# src/evaluate.py
import argparse, json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.model_selection import train_test_split

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--scaler", required=True)
    ap.add_argument("--metrics", default="metrics/eval_metrics.json")
    ap.add_argument("--target", default="Class")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--random_state", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    y = df[args.target].astype(int)
    X = df.drop(columns=[args.target])

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    model = joblib.load(args.model)
    scaler = joblib.load(args.scaler)

    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    y_proba = getattr(model, "predict_proba", None)

    roc = roc_auc_score(y_test, y_proba(X_test_scaled)[:, 1]) if y_proba else None
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc) if roc is not None else None,
        "n_test": int(len(y_test)),
        "positive_rate_test": float(y_test.mean()),
    }

    out = Path(args.metrics)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2)

    print("[evaluate.py] metrics ->", out.resolve())
    print("[evaluate.py] metrics:", metrics)

if __name__ == "__main__":
    main()

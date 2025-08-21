# src/train.py
from __future__ import annotations
import argparse, json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

DEFAULTS = {
    "test_size": 0.2,
    "random_state": 42,
    "C": 1.0,
    "max_iter": 2000,
    "solver": "liblinear",    # works well on imbalanced, small-ish data
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="CSV produced by preprocess step")
    ap.add_argument("--model", required=True, help="Path to save model .joblib")
    ap.add_argument("--scaler", default="artifacts/scaler.joblib", help="Path to save StandardScaler")
    ap.add_argument("--metrics", default="metrics/train_metrics.json", help="Where to write metrics JSON")
    ap.add_argument("--params", default="", help="(optional) params.yaml:train not used here, kept for DVC compatibility")
    ap.add_argument("--target", default="Class", help="Target column name in CSV")
    ap.add_argument("--test_size", type=float, default=DEFAULTS["test_size"])
    ap.add_argument("--random_state", type=int, default=DEFAULTS["random_state"])
    ap.add_argument("--C", type=float, default=DEFAULTS["C"])
    ap.add_argument("--max_iter", type=int, default=DEFAULTS["max_iter"])
    ap.add_argument("--solver", default=DEFAULTS["solver"])
    args = ap.parse_args()

    data_path = Path(args.data)
    df = pd.read_csv(data_path)

    # Split features/target
    y = df[args.target].astype(int)
    X = df.drop(columns=[args.target])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    # Scale (fit on train only)
    scaler = StandardScaler(with_mean=False) if not np.issubdtype(X_train.dtypes[0], np.number) else StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # Model
    clf = LogisticRegression(
        C=args.C,
        max_iter=args.max_iter,
        solver=args.solver,
        class_weight="balanced",
        n_jobs=None,
        random_state=args.random_state if "saga" not in args.solver else None,
    )
    clf.fit(X_train_scaled, y_train)

    # Train-split metrics (report on test for quick feedback)
    y_pred = clf.predict(X_test_scaled)
    y_proba = getattr(clf, "predict_proba", None)
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

    # Save artifacts
    model_path  = Path(args.model)
    scaler_path = Path(args.scaler)
    metrics_path = Path(args.metrics)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(clf, model_path)
    joblib.dump(scaler, scaler_path)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("[train.py] saved model ->", model_path.resolve())
    print("[train.py] saved scaler ->", scaler_path.resolve())
    print("[train.py] metrics ->", metrics_path.resolve())
    print("[train.py] metrics:", metrics)

if __name__ == "__main__":
    main()

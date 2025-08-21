# src/model.py
"""
Model utilities for the credit card fraud detection project.

- Logistic Regression baseline with class weights
- Train/evaluate helpers
- Save/load helpers
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


@dataclass
class TrainConfig:
    C: float = 1.0
    max_iter: int = 1000
    solver: str = "liblinear"  # supports class_weight well & works on small datasets
    random_state: int = 42


def build_model(cfg: TrainConfig, class_weights: Dict[int, float]) -> LogisticRegression:
    """
    Create a Logistic Regression classifier.
    """
    model = LogisticRegression(
        C=cfg.C,
        max_iter=cfg.max_iter,
        solver=cfg.solver,
        class_weight=class_weights,
        random_state=cfg.random_state,
    )
    return model


def train(
    model: LogisticRegression,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> LogisticRegression:
    model.fit(X_train, y_train)
    return model


def evaluate(
    model: LogisticRegression,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Any]:
    """
    Return a dictionary of useful metrics.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "pr_auc": float(average_precision_score(y_test, y_prob)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    return metrics


def save_artifacts(
    model: LogisticRegression,
    scaler,
    model_path: Path,
    scaler_path: Path,
    metrics_path: Path | None = None,
    metrics: Dict[str, Any] | None = None,
) -> None:
    """
    Save model/scaler (joblib) and metrics (json).
    """
    model_path.parent.mkdir(parents=True, exist_ok=True)
    scaler_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    if metrics_path and metrics is not None:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

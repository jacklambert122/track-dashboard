from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class LinearJSONProbabilityModel:
    """Small logistic model loaded from a safe, human-readable JSON artifact."""

    def __init__(self, path: str | Path) -> None:
        model_path = Path(path)
        if not model_path.is_file():
            raise ValueError(f"Linear JSON model does not exist: {model_path}")
        with model_path.open(encoding="utf-8") as model_file:
            payload = json.load(model_file)
        weights = payload.get("weights")
        if not isinstance(weights, list) or not weights:
            raise ValueError("Linear JSON model requires a non-empty weights list.")
        self.path = model_path
        self.weights = np.asarray(weights, dtype=float)
        self.intercept = float(payload.get("intercept", 0.0))

    def predict_proba(self, features):
        matrix = np.asarray(features, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.weights):
            raise ValueError(
                f"Linear JSON model expects {len(self.weights)} features; "
                f"received shape {matrix.shape}."
            )
        logits = np.clip(matrix @ self.weights + self.intercept, -30, 30)
        positive = 1.0 / (1.0 + np.exp(-logits))
        return np.column_stack([1.0 - positive, positive])

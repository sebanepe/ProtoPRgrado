import os
import joblib
import numpy as np
from typing import List, Dict, Tuple

DEFAULT_MODELS_DIR = os.path.join("backend", "app", "ml", "saved_models")


def _find_model_file(models_dir: str, model_name: str, version: str) -> str | None:
    if not os.path.isdir(models_dir):
        return None
    for fn in os.listdir(models_dir):
        if not fn.endswith(".pkl"):
            continue
        name_part = fn[:-4]
        if name_part == f"{model_name}_{version}":
            return os.path.join(models_dir, fn)
    return None


def load_model_by_info(model_name: str, version: str, models_dir: str | None = None):
    models_dir = models_dir or DEFAULT_MODELS_DIR
    path = _find_model_file(models_dir, model_name, version)
    if not path:
        raise FileNotFoundError("Model file not found")
    return joblib.load(path)


def risk_score_from_model(model, name: str, X: List[Dict]) -> List[float]:
    # X is list of feature dicts; convert to 2D array preserving key order
    if len(X) == 0:
        return []
    # assume all dicts have same keys and order doesn't matter for model that was trained on same columns
    keys = list(X[0].keys())
    arr = np.array([[float(x.get(k, 0.0)) for k in keys] for x in X])

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(arr)[:, 1]
        return probs.tolist()
    if hasattr(model, "decision_function"):
        scores = model.decision_function(arr)
        # normalize to 0-1
        s = np.array(scores)
        if s.max() == s.min():
            return np.clip((s - s.min()), 0, 1).tolist()
        norm = (s - s.min()) / (s.max() - s.min())
        return norm.tolist()
    # fallback to predict
    preds = model.predict(arr)
    return preds.astype(float).tolist()


def classify_risk(score: float) -> str:
    if score >= 0.7:
        return "HIGH"
    if score >= 0.3:
        return "MEDIUM"
    return "LOW"

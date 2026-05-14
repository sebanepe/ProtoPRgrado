import json
import os
import numpy as np
from backend.app.ml.evaluation import compute_metrics


def test_regression_metrics_against_baseline():
    # deterministic small dataset
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    y_score = np.array([0.1, 0.9, 0.4, 0.2])

    metrics = compute_metrics(y_true, y_pred, y_score)

    base_path = os.path.join(os.path.dirname(__file__), "regression", "baseline_metrics.json")
    with open(base_path, "r", encoding="utf-8") as fh:
        baseline = json.load(fh)

    # allow tiny floating point tolerance
    tol = 1e-8
    assert abs(metrics.get("precision") - baseline["precision"]) < tol
    assert abs(metrics.get("recall") - baseline["recall"]) < tol
    assert abs(metrics.get("f1_score") - baseline["f1_score"]) < tol
    # roc_auc may be None in some edge cases; require close if present
    if metrics.get("roc_auc") is not None:
        assert abs(metrics.get("roc_auc") - baseline["roc_auc"]) < tol

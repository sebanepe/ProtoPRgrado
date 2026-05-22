"""
Unit tests for metric computation helpers (moved to unit folder).

Verifies `compute_metrics()` returns precision, recall, f1 and roc_auc with expected
values for a deterministic small example.
"""

import numpy as np
from backend.app.ml.evaluation import compute_metrics


def test_compute_metrics_basic_moved():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    y_score = np.array([0.1, 0.9, 0.4, 0.2])

    metrics = compute_metrics(y_true, y_pred, y_score)

    assert round(metrics["precision"], 3) == 1.0
    assert round(metrics["recall"], 3) == 0.5  # recall esperado
    assert round(metrics["f1_score"], 3) == round(2 * (1.0 * 0.5) / (1.0 + 0.5), 3)  # f1 calculada a partir de precision y recall
    assert metrics["roc_auc"] is not None  # ROC AUC debe existir con scores válidos
    assert 0.0 <= metrics["roc_auc"] <= 1.0  # ROC AUC debe estar en rango [0,1]

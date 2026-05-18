import numpy as np
from backend.app.ml.evaluation import compute_metrics

"""Unit test for compute_metrics isolated from DB or I/O."""

def test_compute_metrics_values():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    y_score = np.array([0.1, 0.9, 0.4, 0.2])
    metrics = compute_metrics(y_true, y_pred, y_score)
    assert round(metrics['precision'], 3) == 1.0
    assert round(metrics['recall'], 3) == 0.5
    assert 'f1_score' in metrics
    assert metrics['roc_auc'] is not None

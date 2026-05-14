import numpy as np
from backend.app.ml.evaluation import compute_metrics


def test_compute_metrics_basic():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    y_score = np.array([0.1, 0.9, 0.4, 0.2])

    metrics = compute_metrics(y_true, y_pred, y_score)

    # precision = 1.0 (1 tp, 0 fp)
    assert round(metrics["precision"], 3) == 1.0
    # recall = 0.5 (1 tp, 1 fn)
    assert round(metrics["recall"], 3) == 0.5
    # f1 is harmonic mean
    assert round(metrics["f1_score"], 3) == round(2 * (1.0 * 0.5) / (1.0 + 0.5), 3)
    # roc_auc should be not None and between 0 and 1
    assert metrics["roc_auc"] is not None
    assert 0.0 <= metrics["roc_auc"] <= 1.0

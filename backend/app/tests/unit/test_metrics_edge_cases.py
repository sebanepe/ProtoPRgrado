"""
Pruebas unitarias de métricas (casos límite y valores conocidos).
Usan `compute_metrics` y `compute_metrics_extended` para comprobar valores.
"""
import numpy as np
from backend.app.ml.evaluation import compute_metrics, compute_metrics_extended


def test_precision_with_known_values():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    metrics = compute_metrics(y_true, y_pred)
    assert round(metrics["precision"], 3) == 1.0


def test_recall_with_known_values():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    metrics = compute_metrics(y_true, y_pred)
    assert round(metrics["recall"], 3) == 0.5


def test_f1_with_known_values():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    metrics = compute_metrics(y_true, y_pred)
    assert "f1_score" in metrics


def test_metrics_handle_zero_division():
    # no positive predictions => precision zero_division handled
    y_true = np.array([0, 0, 0])
    y_pred = np.array([0, 0, 0])
    metrics = compute_metrics(y_true, y_pred)
    assert metrics["precision"] == 0.0


def test_roc_auc_with_valid_probabilities():
    y_true = np.array([0, 1, 1, 0])
    y_score = np.array([0.1, 0.9, 0.4, 0.2])
    metrics = compute_metrics(y_true, y_pred=np.array([0, 1, 0, 0]), y_score=y_score)
    assert metrics["roc_auc"] is not None


def test_roc_auc_with_single_class_returns_none_or_safe_value():
    y_true = np.array([0, 0, 0])
    y_pred = np.array([0, 0, 0])
    metrics = compute_metrics(y_true, y_pred)
    assert metrics.get("roc_auc") is None or isinstance(metrics.get("roc_auc"), float)


def test_confusion_matrix_values():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    ext = compute_metrics_extended(y_true, y_pred)
    cm = ext.get("confusion_matrix")
    # Expect TN=2, FP=0, FN=1, TP=1 for this case
    assert cm["tn"] == 2
    assert cm["fp"] == 0
    assert cm["fn"] == 1
    assert cm["tp"] == 1


def test_metrics_output_contains_expected_keys():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    ext = compute_metrics_extended(y_true, y_pred)
    for k in ["accuracy", "precision", "recall", "f1_score", "roc_auc", "confusion_matrix"]:
        assert k in ext

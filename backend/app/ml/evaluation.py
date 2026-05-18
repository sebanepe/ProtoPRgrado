from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from typing import Optional, Dict
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix


def compute_metrics(y_true, y_pred, y_score: Optional[np.ndarray] = None) -> Dict:
    metrics = {}
    metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    metrics["f1_score"] = float(f1_score(y_true, y_pred, zero_division=0))
    # roc_auc requires score/probabilities; fall back to predictions if needed
    try:
        if y_score is not None:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        else:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_pred))
    except Exception:
        metrics["roc_auc"] = None
    return metrics


def compute_metrics_extended(y_true, y_pred, y_score: Optional[np.ndarray] = None) -> Dict:
    """Extended metrics helper that includes accuracy and confusion matrix.

    Returns keys: accuracy, precision, recall, f1_score, roc_auc, confusion_matrix
    """
    metrics = compute_metrics(y_true, y_pred, y_score)
    try:
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    except Exception:
        metrics["accuracy"] = None

    try:
        cm = confusion_matrix(y_true, y_pred)
        # return dict for readability: TP, FP, FN, TN (for binary, sklearn gives [[TN, FP],[FN, TP]])
        if cm.size == 4:
            tn, fp, fn, tp = int(cm.ravel()[0]), int(cm.ravel()[1]), int(cm.ravel()[2]), int(cm.ravel()[3])
            metrics["confusion_matrix"] = {"tn": tn, "fp": fp, "fn": fn, "tp": tp}
        else:
            metrics["confusion_matrix"] = {"matrix": cm.tolist()}
    except Exception:
        metrics["confusion_matrix"] = None

    return metrics

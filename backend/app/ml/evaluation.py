from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from typing import Optional, Dict
import numpy as np


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

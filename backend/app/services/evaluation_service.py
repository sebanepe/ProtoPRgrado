import os
import pandas as pd
from typing import List, Dict
from backend.app.ml.evaluator import discover_models, load_model
from backend.app.ml.evaluation import compute_metrics
from backend.app.models.models import ModelResult
from sqlalchemy.orm import Session
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

DEFAULT_MODELS_DIR = os.path.join("backend", "app", "ml", "saved_models")
DEFAULT_OUTPUT = os.path.join("data", "processed", "model_comparison.csv")


def _load_processed(input_path: str):
    if not os.path.exists(input_path):
        raise FileNotFoundError("Processed file not found for evaluation")
    return pd.read_csv(input_path)


def compare_models(db: Session, input_path: str, models_dir: str | None = None, export_path: str | None = None) -> List[Dict]:
    models_dir = models_dir or DEFAULT_MODELS_DIR
    export_path = export_path or DEFAULT_OUTPUT

    df = _load_processed(input_path)
    if df.empty:
        return []
    if "is_fraud" not in df.columns:
        raise ValueError("Processed data must contain 'is_fraud'")

    X = df.drop(columns=["is_fraud"]) if "is_fraud" in df.columns else df.copy()
    y = df["is_fraud"].astype(int)

    # use a test split
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if len(y.unique())>1 else None)

    discovered = discover_models(models_dir)
    results = []
    for model_name, version, path in discovered:
        try:
            model = load_model(path)
        except Exception:
            continue
        # predict
        try:
            if hasattr(model, "predict_proba"):
                y_score = model.predict_proba(X_test)[:, 1]
                y_pred = (y_score >= 0.5).astype(int)
            elif hasattr(model, "decision_function"):
                y_score = model.decision_function(X_test)
                y_pred = (y_score >= 0).astype(int)
            else:
                y_pred = model.predict(X_test)
                y_score = None
        except Exception:
            continue

        metrics = compute_metrics(y_test, y_pred, y_score)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0,1]).ravel() if len(set(y_test))>1 else (0,0,0,0)

        # save to DB
        mr = ModelResult(
            model_name=model_name,
            version=version,
            accuracy=None,
            precision=metrics.get("precision"),
            recall=metrics.get("recall"),
            f1_score=metrics.get("f1_score"),
            roc_auc=metrics.get("roc_auc"),
        )
        db.add(mr)
        db.commit()
        db.refresh(mr)

        results.append(
            {
                "id": mr.id,
                "model_name": model_name,
                "version": version,
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "f1_score": metrics.get("f1_score"),
                "roc_auc": metrics.get("roc_auc"),
                "tp": int(tp),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "path": path,
            }
        )

    # create dataframe and export
    df_out = pd.DataFrame(results)
    if not df_out.empty:
        df_out = df_out.sort_values(by=["recall", "f1_score"], ascending=False)
        os.makedirs(os.path.dirname(export_path), exist_ok=True)
        df_out.to_csv(export_path, index=False)

    return results

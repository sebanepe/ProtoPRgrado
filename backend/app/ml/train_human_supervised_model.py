from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sqlalchemy.orm import Session

from backend.app.database import SessionLocal
from backend.app.ml.supervised_training_preflight import run_training_preflight
from backend.app.services import artifact_registry_service as artifacts
from backend.app.services import model_registry_service


MODEL_TYPES = {"logistic_regression", "random_forest", "gradient_boosting", "mlp"}
FORBIDDEN_COLUMNS = {"is_fraud", "confirmed_fraud", "PAN_TARJETA", "TARJETA", "pan_card", "raw_card"}
METHODOLOGY_WARNING = (
    "Los modelos supervisados fueron entrenados unicamente con etiquetas humanas de revision. "
    "CONFIRMED_FRAUD fue usado como clase positiva y DISMISSED como clase negativa. "
    "El modelo apoya la priorizacion analitica y no reemplaza la decision humana."
)


def _classifier(model_type: str, random_state: int) -> Any:
    if model_type == "logistic_regression":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)),
            ]
        )
    if model_type == "random_forest":
        return RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=random_state)
    if model_type == "gradient_boosting":
        return GradientBoostingClassifier(random_state=random_state)
    raise ValueError(f"Unsupported model_type: {model_type}")


def _feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    drop_columns = FORBIDDEN_COLUMNS | {
        "target_human_label",
        "target_label_source",
        "target_label_meaning",
        "human_review_status",
        "human_review_comment",
        "reviewed_at",
        "reviewed_by",
        "source_run",
        "summary_alert_id",
        "representative_transaction_id",
        "customer_hash",
        "window_start",
        "window_end",
    }
    X = df.drop(columns=[column for column in drop_columns if column in df.columns]).copy()
    for column in X.columns:
        if X[column].dtype == "bool":
            X[column] = X[column].astype(int)
    X = pd.get_dummies(X, dummy_na=True)
    return X.apply(pd.to_numeric, errors="coerce").fillna(0)


def _evaluation_result(y_true: int, y_pred: int) -> str:
    if y_true == 1 and y_pred == 1:
        return "TRUE_POSITIVE"
    if y_true == 0 and y_pred == 0:
        return "TRUE_NEGATIVE"
    if y_true == 0 and y_pred == 1:
        return "FALSE_POSITIVE"
    return "FALSE_NEGATIVE"


def _metrics(y_true: pd.Series, y_pred: list[int], y_proba: Optional[list[float]]) -> dict[str, Any]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }
    if y_proba is not None and len(set(y_true.tolist())) == 2:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except Exception:
            metrics["roc_auc"] = None
    return metrics


def _write_report(path: Path, metadata: dict[str, Any], artifacts_payload: dict[str, str]) -> None:
    metrics = metadata["metrics"]
    lines = [
        "# Reporte entrenamiento supervisado humano",
        "",
        f"- source_run: {metadata['source_run']}",
        f"- run_token: {metadata['run_token']}",
        f"- algorithm: {metadata['algorithm']}",
        f"- total_rows: {metadata['total_rows']}",
        f"- positive_count: {metadata['positive_count']}",
        f"- negative_count: {metadata['negative_count']}",
        f"- accuracy: {metrics.get('accuracy')}",
        f"- precision: {metrics.get('precision')}",
        f"- recall: {metrics.get('recall')}",
        f"- f1_score: {metrics.get('f1_score')}",
        f"- roc_auc: {metrics.get('roc_auc')}",
        f"- confusion_matrix: {metrics.get('confusion_matrix')}",
        "",
        "## Artefactos",
        "",
    ]
    lines.extend(f"- {name}: {value}" for name, value in artifacts_payload.items())
    lines.extend(["", "## Advertencia metodologica", "", METHODOLOGY_WARNING, ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_closure_report(
    *,
    source_run: str,
    run_token: str,
    preflight: dict[str, Any],
    db: Session,
) -> Path:
    path = artifacts.default_processed_dir() / f"supervised_training_closure_report_run_{run_token}.md"
    models = model_registry_service.list_model_registry(db, source_run=source_run, model_family="SUPERVISED_HUMAN")
    all_artifacts = artifacts.list_artifacts(db, source_run=source_run, phase=artifacts.PHASE_C4)
    lines = [
        "# Cierre tecnico entrenamiento supervisado humano C4.4",
        "",
        f"- source_run: {source_run}",
        f"- fecha_ejecucion: {datetime.now(timezone.utc).isoformat()}",
        f"- CONFIRMED_FRAUD: {preflight['human_labels']['confirmed_fraud']}",
        f"- DISMISSED: {preflight['human_labels']['dismissed']}",
        f"- total usable: {preflight['human_labels']['usable_total']}",
        f"- technical_ready: {preflight['human_labels']['technical_ready']}",
        f"- recommended_ready: {preflight['human_labels']['recommended_ready']}",
        f"- strong_ready: {preflight['human_labels']['strong_ready']}",
        f"- dataset archivo: {preflight['dataset']['file']}",
        f"- dataset filas: {preflight['dataset']['rows']}",
        f"- dataset positivos: {preflight['dataset']['positive_count']}",
        f"- dataset negativos: {preflight['dataset']['negative_count']}",
        f"- dataset verdict: {preflight['dataset']['verdict']}",
        "",
        "## Modelos entrenados",
        "",
    ]
    for model in models:
        metrics = model_registry_service.model_registry_to_dict(model).get("metrics_json", {})
        lines.extend(
            [
                f"- {model.algorithm}: status={model.status}, is_active={model.is_active}",
                f"  - accuracy: {metrics.get('accuracy')}",
                f"  - precision: {metrics.get('precision')}",
                f"  - recall: {metrics.get('recall')}",
                f"  - f1_score: {metrics.get('f1_score')}",
                f"  - roc_auc: {metrics.get('roc_auc')}",
                f"  - confusion_matrix: {metrics.get('confusion_matrix')}",
            ]
        )
    lines.extend(["", "## Registros artifact_registry", ""])
    lines.extend(f"- {item.artifact_type}: {item.file_name} ({item.status})" for item in all_artifacts)
    lines.extend(["", "## Registros model_registry", ""])
    lines.extend(f"- {item.algorithm}: {item.status}, active={item.is_active}" for item in models)
    lines.extend(["", "## Advertencia metodologica", "", METHODOLOGY_WARNING, ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    artifacts.register_or_update_artifact(
        db,
        artifact_type=artifacts.ARTIFACT_SUPERVISED_REPORT,
        phase=artifacts.PHASE_C4,
        source_run=source_run,
        run_token=run_token,
        file_path=path,
        metadata={"report_kind": "supervised_training_closure", "model_family": "SUPERVISED_HUMAN"},
    )
    return path


def train_human_supervised_model(
    source_run: str,
    model_type: str,
    *,
    db: Optional[Session] = None,
    test_size: float = 0.25,
    random_state: int = 42,
) -> dict[str, Any]:
    if model_type not in MODEL_TYPES:
        raise ValueError(f"model_type must be one of {sorted(MODEL_TYPES)}")
    own_session = db is None
    session = db or SessionLocal()
    try:
        normalized = artifacts.normalize_source_run(source_run)
        run_token = artifacts.normalize_run_token(normalized)
        preflight = run_training_preflight(normalized, session, build_if_missing=True)
        if model_type == "mlp" and not preflight["human_labels"]["recommended_ready"]:
            return {
                "status": "BLOCKED",
                "verdict": "MLP_REQUIRES_RECOMMENDED_HUMAN_LABELS",
                "source_run": normalized,
                "model_type": model_type,
                "blocking_reason": "INSUFFICIENT_RECOMMENDED_HUMAN_LABELS",
                "preflight": preflight,
            }
        if not preflight["can_train"]:
            return {
                "status": "BLOCKED",
                "verdict": "HUMAN_SUPERVISED_TRAINING_BLOCKED",
                "source_run": normalized,
                "model_type": model_type,
                "blocking_reason": preflight["blocking_reason"],
                "preflight": preflight,
            }

        dataset_path = Path(preflight["dataset"]["path"])
        df = pd.read_csv(dataset_path)
        if FORBIDDEN_COLUMNS.intersection(df.columns):
            return {"status": "BLOCKED", "verdict": "HUMAN_SUPERVISED_TRAINING_BLOCKED", "blocking_reason": "FORBIDDEN_COLUMNS_PRESENT"}

        y = pd.to_numeric(df["target_human_label"], errors="coerce").astype(int)
        X = _feature_frame(df)
        X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
            X,
            y,
            df[["source_run", "summary_alert_id"]].copy(),
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
        model = _classifier(model_type, random_state)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test).astype(int).tolist()
        y_proba = None
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1].astype(float).tolist()
        metrics = _metrics(y_test, y_pred, y_proba)

        processed = artifacts.default_processed_dir()
        models_dir = artifacts.default_models_dir()
        processed.mkdir(parents=True, exist_ok=True)
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / f"supervised_human_{model_type}_run_{run_token}.pkl"
        metadata_path = models_dir / f"supervised_human_{model_type}_run_{run_token}_metadata.json"
        report_path = processed / f"supervised_human_{model_type}_report_run_{run_token}.md"
        predictions_path = processed / f"supervised_human_{model_type}_predictions_run_{run_token}.csv"

        predictions = meta_test.reset_index(drop=True)
        predictions["y_true"] = y_test.reset_index(drop=True).astype(int)
        predictions["y_pred"] = y_pred
        if y_proba is not None:
            predictions["y_proba"] = y_proba
        predictions["prediction_label"] = predictions["y_pred"].map({1: "CONFIRMED_FRAUD", 0: "DISMISSED"})
        predictions["evaluation_result"] = [
            _evaluation_result(int(true), int(pred)) for true, pred in zip(predictions["y_true"], predictions["y_pred"])
        ]
        predictions.to_csv(predictions_path, index=False)

        joblib.dump({"model": model, "feature_columns": X.columns.tolist()}, model_path)
        metadata = {
            "source_run": normalized,
            "run_token": run_token,
            "model_family": "SUPERVISED_HUMAN",
            "algorithm": model_type,
            "target": "target_human_label",
            "label_policy": "HUMAN_REVIEW_CONFIRMED_FRAUD_DISMISSED",
            "positive_label": "CONFIRMED_FRAUD",
            "negative_label": "DISMISSED",
            "total_rows": int(len(df)),
            "positive_count": int((y == 1).sum()),
            "negative_count": int((y == 0).sum()),
            "test_size": test_size,
            "random_state": random_state,
            "metrics": metrics,
            "methodological_warnings": [METHODOLOGY_WARNING],
            "feature_columns": X.columns.tolist(),
        }
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")
        _write_report(
            report_path,
            metadata,
            {
                "model": str(model_path),
                "metadata": str(metadata_path),
                "predictions": str(predictions_path),
                "dataset": str(dataset_path),
            },
        )

        registered_artifacts = {
            "model": artifacts.register_or_update_artifact(
                session,
                artifact_type=artifacts.ARTIFACT_MODEL_PICKLE,
                phase=artifacts.PHASE_C4,
                source_run=normalized,
                run_token=run_token,
                file_path=model_path,
                metadata={"algorithm": model_type, "model_family": "SUPERVISED_HUMAN"},
            ),
            "metadata": artifacts.register_or_update_artifact(
                session,
                artifact_type=artifacts.ARTIFACT_MODEL_METADATA,
                phase=artifacts.PHASE_C4,
                source_run=normalized,
                run_token=run_token,
                file_path=metadata_path,
                metadata={"algorithm": model_type, "model_family": "SUPERVISED_HUMAN"},
            ),
            "report": artifacts.register_or_update_artifact(
                session,
                artifact_type=artifacts.ARTIFACT_MODEL_REPORT,
                phase=artifacts.PHASE_C4,
                source_run=normalized,
                run_token=run_token,
                file_path=report_path,
                metadata={"algorithm": model_type, "model_family": "SUPERVISED_HUMAN"},
            ),
            "predictions": artifacts.register_or_update_artifact(
                session,
                artifact_type=artifacts.ARTIFACT_MODEL_PREDICTIONS,
                phase=artifacts.PHASE_C4,
                source_run=normalized,
                run_token=run_token,
                file_path=predictions_path,
                metadata={"algorithm": model_type, "model_family": "SUPERVISED_HUMAN"},
            ),
        }
        model_registry = model_registry_service.register_supervised_human_model(
            session,
            source_run=normalized,
            algorithm=model_type,
            model_file=str(model_path),
            metadata_file=str(metadata_path),
            report_file=str(report_path),
            predictions_file=str(predictions_path),
            feature_file=str(dataset_path),
            metrics=metrics,
        )
        closure_report = _write_closure_report(source_run=normalized, run_token=run_token, preflight=preflight, db=session)

        return {
            "status": "COMPLETED",
            "verdict": "HUMAN_SUPERVISED_TRAINING_COMPLETED",
            "source_run": normalized,
            "model_type": model_type,
            "metrics": metrics,
            "files": {
                "model": str(model_path),
                "metadata": str(metadata_path),
                "report": str(report_path),
                "predictions": str(predictions_path),
                "closure_report": str(closure_report),
            },
            "artifact_registry": {key: value.id for key, value in registered_artifacts.items()},
            "model_registry_id": model_registry.id,
            "preflight": preflight,
        }
    finally:
        if own_session:
            session.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a human supervised model")
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--model-type", required=True, choices=sorted(MODEL_TYPES))
    args = parser.parse_args()
    result = train_human_supervised_model(args.source_run, args.model_type)
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

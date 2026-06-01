from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from backend.app.database import SessionLocal
from backend.app.ml.scoring import classify_risk_level
from backend.app.models.models import BatchScoringRun, ModelRegistry
from backend.app.services import artifact_registry_service as artifacts

VALID_ALGORITHMS = {"logistic_regression", "random_forest", "gradient_boosting"}

FORBIDDEN_OUTPUT_COLUMNS = {
    "is_fraud",
    "confirmed_fraud",
    "PAN_TARJETA",
    "TARJETA",
    "pan_card",
    "raw_card",
    "target_human_label",
    "target_label_source",
    "target_label_meaning",
    "human_review_comment",
    "reviewed_by",
}

_TRAINING_DROP_COLUMNS = FORBIDDEN_OUTPUT_COLUMNS | {
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

_IDENTITY_COLUMNS = [
    "source_run",
    "summary_alert_id",
    "representative_transaction_id",
    "customer_hash",
    "rule_code",
    "rule_name",
    "risk_level",
    "max_score",
    "transactions_detected",
    "countries_detected",
    "merchant_rubro_proxy",
    "window_start",
    "window_end",
    "duration_minutes",
    "countries_count",
]

LOW_MEDIUM_THRESHOLD = 0.5
MEDIUM_HIGH_THRESHOLD = 0.75

METHODOLOGY_WARNING = (
    "Este scoring genera predicciones de apoyo analítico. "
    "No confirma fraude automáticamente y no reemplaza la revisión humana."
)


def _feature_frame_for_scoring(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    X = df.drop(columns=[c for c in _TRAINING_DROP_COLUMNS if c in df.columns]).copy()
    for col in X.columns:
        if X[col].dtype == "bool":
            X[col] = X[col].astype(int)
    X = pd.get_dummies(X, dummy_na=True)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    return X.reindex(columns=feature_columns, fill_value=0)


def _load_model(
    db: Session,
    source_run: str,
    algorithm: str,
) -> Tuple[Any, list[str], ModelRegistry]:
    normalized = artifacts.normalize_source_run(source_run)
    row = (
        db.query(ModelRegistry)
        .filter(
            ModelRegistry.model_family == "SUPERVISED_HUMAN",
            ModelRegistry.algorithm == algorithm,
            ModelRegistry.source_run == normalized,
            ModelRegistry.status == "AVAILABLE",
        )
        .order_by(ModelRegistry.created_at.desc())
        .first()
    )
    if row is None or not row.model_file:
        raise FileNotFoundError(
            f"No AVAILABLE SUPERVISED_HUMAN model for source_run={normalized} algorithm={algorithm}"
        )
    artifact_dict = joblib.load(row.model_file)
    model = artifact_dict["model"]
    feature_columns: list[str] = artifact_dict["feature_columns"]
    return model, feature_columns, row


def _predict(model: Any, X: pd.DataFrame) -> Tuple[list[int], Optional[list[float]]]:
    X_arr = X.values
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_arr)[:, 1].tolist()
        preds = [1 if p >= LOW_MEDIUM_THRESHOLD else 0 for p in proba]
        return preds, proba
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X_arr)
        s = np.array(scores, dtype=float)
        if s.max() == s.min():
            proba = np.clip(s - s.min(), 0.0, 1.0).tolist()
        else:
            proba = ((s - s.min()) / (s.max() - s.min())).tolist()
        preds = [1 if p >= LOW_MEDIUM_THRESHOLD else 0 for p in proba]
        return preds, proba
    preds = model.predict(X_arr).astype(int).tolist()
    return preds, None


def _build_results_df(
    df: pd.DataFrame,
    model: Any,
    feature_columns: list[str],
    source_run: str,
    algorithm: str,
) -> pd.DataFrame:
    X = _feature_frame_for_scoring(df, feature_columns)
    y_pred, y_proba = _predict(model, X)

    identity = {col: df[col].tolist() if col in df.columns else [None] * len(df) for col in _IDENTITY_COLUMNS}
    now_iso = datetime.now(timezone.utc).isoformat()

    if y_proba is not None:
        risk_levels = [
            classify_risk_level(p, LOW_MEDIUM_THRESHOLD, MEDIUM_HIGH_THRESHOLD) for p in y_proba
        ]
        scores = [round(float(p), 6) for p in y_proba]
    else:
        risk_levels = ["HIGH" if p == 1 else "LOW" for p in y_pred]
        scores = [float(p) for p in y_pred]

    result = dict(identity)
    result["source_run"] = source_run
    result["ml_risk_score"] = scores
    result["ml_risk_level"] = risk_levels
    result["algorithm"] = algorithm
    result["scored_at"] = now_iso

    out = pd.DataFrame(result)

    forbidden_present = FORBIDDEN_OUTPUT_COLUMNS.intersection(out.columns)
    if forbidden_present:
        out = out.drop(columns=list(forbidden_present))

    return out


def _write_report(path: Path, metadata: dict[str, Any]) -> None:
    high = metadata.get("high_count", 0)
    medium = metadata.get("medium_count", 0)
    low = metadata.get("low_count", 0)
    total = metadata.get("total_scored", 0)
    lines = [
        "# Reporte Scoring por Lotes D1",
        "",
        f"- source_run: {metadata.get('source_run')}",
        f"- algorithm: {metadata.get('algorithm')}",
        f"- model_family: SUPERVISED_HUMAN",
        f"- input_file: {metadata.get('input_file')}",
        f"- total_registros_evaluados: {total}",
        "",
        "## Distribución por ml_risk_level",
        "",
        f"- HIGH: {high} ({round(high / total * 100, 1) if total else 0}%)",
        f"- MEDIUM: {medium} ({round(medium / total * 100, 1) if total else 0}%)",
        f"- LOW: {low} ({round(low / total * 100, 1) if total else 0}%)",
        "",
        "## Thresholds aplicados",
        "",
        f"- LOW/MEDIUM: {LOW_MEDIUM_THRESHOLD}",
        f"- MEDIUM/HIGH: {MEDIUM_HIGH_THRESHOLD}",
        "",
        "## Archivos generados",
        "",
        f"- results_file: {metadata.get('results_file')}",
        f"- report_file: {metadata.get('report_file')}",
        f"- metadata_file: {metadata.get('metadata_file')}",
        "",
        "## Advertencia metodológica",
        "",
        f"> {METHODOLOGY_WARNING}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_batch_scoring(
    source_run: str,
    algorithm: str,
    *,
    db: Optional[Session] = None,
    input_dataset_path: Optional[str] = None,
) -> dict[str, Any]:
    if algorithm not in VALID_ALGORITHMS:
        raise ValueError(f"algorithm must be one of {VALID_ALGORITHMS}, got {algorithm!r}")

    own_session = db is None
    session = db if db is not None else SessionLocal()
    try:
        return _run_batch_scoring_inner(
            source_run,
            algorithm,
            session=session,
            input_dataset_path=input_dataset_path,
        )
    finally:
        if own_session:
            session.close()


def _run_batch_scoring_inner(
    source_run: str,
    algorithm: str,
    *,
    session: Session,
    input_dataset_path: Optional[str],
) -> dict[str, Any]:
    normalized = artifacts.normalize_source_run(source_run)
    run_token = artifacts.normalize_run_token(normalized)

    # Resolve dataset path
    dataset_path: Optional[Path] = None
    if input_dataset_path:
        dataset_path = Path(input_dataset_path)
    else:
        artifact_row = artifacts.get_artifact_by_type(
            session, normalized, artifacts.ARTIFACT_SUPERVISED_DATASET
        )
        if artifact_row and artifact_row.file_path:
            dataset_path = Path(artifact_row.file_path)
        else:
            dataset_path = (
                artifacts.default_processed_dir()
                / f"supervised_human_alert_dataset_run_{run_token}.csv"
            )

    if not dataset_path.exists():
        return {
            "status": "BLOCKED",
            "verdict": "SCORING_BLOCKED_DATASET_NOT_FOUND",
            "source_run": normalized,
            "algorithm": algorithm,
            "dataset_path": str(dataset_path),
            "warnings": [METHODOLOGY_WARNING],
        }

    # Safety check: no forbidden columns in input
    try:
        header_cols = set(pd.read_csv(dataset_path, nrows=0).columns.tolist())
    except Exception as exc:
        return {
            "status": "BLOCKED",
            "verdict": "SCORING_BLOCKED_DATASET_READ_ERROR",
            "source_run": normalized,
            "algorithm": algorithm,
            "error": str(exc),
            "warnings": [METHODOLOGY_WARNING],
        }

    forbidden_in_input = {"is_fraud", "confirmed_fraud"}.intersection(header_cols)
    if forbidden_in_input:
        return {
            "status": "BLOCKED",
            "verdict": "SCORING_BLOCKED_FORBIDDEN_COLUMNS",
            "source_run": normalized,
            "algorithm": algorithm,
            "forbidden_found": sorted(forbidden_in_input),
            "warnings": [METHODOLOGY_WARNING],
        }

    # Create DB run record
    db_run = BatchScoringRun(
        source_run=normalized,
        run_token=run_token,
        model_family="SUPERVISED_HUMAN",
        algorithm=algorithm,
        input_file=str(dataset_path),
        status="RUNNING",
        started_at=datetime.now(timezone.utc),
        params_json=json.dumps({"algorithm": algorithm, "source_run": normalized}),
    )
    session.add(db_run)
    session.commit()
    session.refresh(db_run)

    # Load model
    try:
        model, feature_columns, registry_row = _load_model(session, normalized, algorithm)
        db_run.model_registry_id = registry_row.id
        db_run.model_file = registry_row.model_file
        session.commit()
    except FileNotFoundError as exc:
        db_run.status = "FAILED"
        db_run.error_message = str(exc)
        session.commit()
        return {
            "status": "BLOCKED",
            "verdict": "SCORING_BLOCKED_MODEL_NOT_FOUND",
            "source_run": normalized,
            "algorithm": algorithm,
            "error": str(exc),
            "warnings": [METHODOLOGY_WARNING],
        }

    # Execute scoring
    try:
        df = pd.read_csv(dataset_path)
        results_df = _build_results_df(df, model, feature_columns, normalized, algorithm)

        high_count = int((results_df["ml_risk_level"] == "HIGH").sum())
        medium_count = int((results_df["ml_risk_level"] == "MEDIUM").sum())
        low_count = int((results_df["ml_risk_level"] == "LOW").sum())
        total_scored = len(results_df)

        processed_dir = artifacts.default_processed_dir()
        processed_dir.mkdir(parents=True, exist_ok=True)

        results_path = processed_dir / f"scoring_results_run_{run_token}_{algorithm}.csv"
        report_path = processed_dir / f"scoring_report_run_{run_token}_{algorithm}.md"
        metadata_path = processed_dir / f"scoring_metadata_run_{run_token}_{algorithm}.json"

        results_df.to_csv(results_path, index=False)

        metadata_payload = {
            "source_run": normalized,
            "run_token": run_token,
            "algorithm": algorithm,
            "model_family": "SUPERVISED_HUMAN",
            "total_scored": total_scored,
            "high_count": high_count,
            "medium_count": medium_count,
            "low_count": low_count,
            "low_medium_threshold": LOW_MEDIUM_THRESHOLD,
            "medium_high_threshold": MEDIUM_HIGH_THRESHOLD,
            "input_file": str(dataset_path),
            "results_file": str(results_path),
            "report_file": str(report_path),
            "metadata_file": str(metadata_path),
            "scored_at": datetime.now(timezone.utc).isoformat(),
            "methodology_warning": METHODOLOGY_WARNING,
        }
        metadata_path.write_text(
            json.dumps(metadata_payload, ensure_ascii=True, indent=2), encoding="utf-8"
        )

        _write_report(report_path, metadata_payload)

        # Register artifacts
        for artifact_type, file_path in [
            (artifacts.ARTIFACT_SCORING_RESULTS, results_path),
            (artifacts.ARTIFACT_SCORING_REPORT, report_path),
            (artifacts.ARTIFACT_SCORING_METADATA, metadata_path),
        ]:
            artifacts.register_or_update_artifact(
                session,
                artifact_type=artifact_type,
                phase=artifacts.PHASE_D1,
                source_run=normalized,
                run_token=run_token,
                file_path=file_path,
                metadata={"algorithm": algorithm},
            )

        # Update DB record
        db_run.status = "COMPLETED"
        db_run.total_scored = total_scored
        db_run.high_count = high_count
        db_run.medium_count = medium_count
        db_run.low_count = low_count
        db_run.results_file = str(results_path)
        db_run.report_file = str(report_path)
        db_run.metadata_file = str(metadata_path)
        db_run.finished_at = datetime.now(timezone.utc)
        session.commit()
        session.refresh(db_run)

        return {
            "status": "COMPLETED",
            "batch_scoring_run_id": db_run.id,
            "source_run": normalized,
            "algorithm": algorithm,
            "model_family": "SUPERVISED_HUMAN",
            "total_scored": total_scored,
            "high_count": high_count,
            "medium_count": medium_count,
            "low_count": low_count,
            "low_medium_threshold": LOW_MEDIUM_THRESHOLD,
            "medium_high_threshold": MEDIUM_HIGH_THRESHOLD,
            "results_file": str(results_path),
            "report_file": str(report_path),
            "metadata_file": str(metadata_path),
            "warnings": [METHODOLOGY_WARNING],
        }

    except Exception as exc:
        db_run.status = "FAILED"
        db_run.error_message = str(exc)
        session.commit()
        raise RuntimeError(f"Batch scoring failed: {exc}") from exc

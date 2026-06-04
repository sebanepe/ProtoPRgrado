"""Service for applying trained supervised models to new datasets.

Models are LOADED from ModelRegistry (model_family='SUPERVISED_HUMAN') — never retrained.
No is_fraud or confirmed_fraud is generated.
Outputs prediction_label (0/1), prediction_probability, and priority_level (HIGH/MEDIUM/LOW).
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from backend.app.models.models import (
    ArtifactRegistry,
    ModelRegistry,
    PreprocessingRun,
    SupervisedInferenceRun,
)
from backend.app.services import artifact_registry_service as artifacts
from backend.app.ml.unsupervised_feature_builder import (
    default_processed_dir,
    normalize_run_token,
)

METHODOLOGY_WARNING = (
    "Las predicciones generadas por modelos supervisados son apoyo analítico "
    "y no constituyen fraude confirmado automático."
)

# Priority thresholds based on predicted probability
PRIORITY_HIGH_THRESHOLD = 0.7
PRIORITY_MEDIUM_THRESHOLD = 0.4

FORBIDDEN_OUTPUT_COLUMNS = {
    "is_fraud",
    "confirmed_fraud",
    "target_is_fraud",
    "target_human_label",
    "analyst_label",
    "human_label",
    "review_status",
    "reviewed_by",
    "reviewed_at",
    "rule_code",
    "rule_name",
    "alert_reason",
    "risk_score",
    "behavioral_risk_score",
    "pan_tarjeta",
    "tarjeta",
    "pan_card",
    "raw_card",
    "masked_card",
    "PAN_TARJETA",
    "TARJETA",
    "password",
    "password_hash",
}


# Columns kept in the result CSV for analyst context (not used as model features)
_CONTEXT_OUTPUT_COLS = [
    "summary_alert_id", "customer_hash", "representative_transaction_id",
    "window_start", "window_end", "rule_code", "rule_name",
]

# Columns dropped before passing data to the model (metadata, labels, sensitive)
_DROP_FOR_MODEL = frozenset({
    "is_fraud", "confirmed_fraud", "target_human_label", "target_label_source",
    "target_label_meaning", "human_review_status", "human_review_comment",
    "reviewed_at", "reviewed_by", "source_run", "summary_alert_id",
    "representative_transaction_id", "customer_hash", "window_start", "window_end",
    "PAN_TARJETA", "TARJETA", "pan_card", "raw_card", "masked_card",
    "password", "password_hash",
})


_ALERT_REQUIRED_COLS = {"rule_code", "rule_name"}
_TRANSACTION_ONLY_COLS = {"transaction_id", "amount_log", "hour_of_day"}


def validate_alert_schema_csv(file_path: str) -> None:
    """Raise ValueError if the CSV looks like raw transactions, not alert summaries."""
    df = pd.read_csv(file_path, nrows=5)
    cols_lower = {c.lower() for c in df.columns}
    has_alert = bool(_ALERT_REQUIRED_COLS & cols_lower)
    looks_raw = bool(_TRANSACTION_ONLY_COLS & cols_lower) and not has_alert
    if looks_raw:
        raise ValueError(
            "El CSV parece contener transacciones crudas. "
            "Se requiere un CSV de resumen de alertas con columnas como "
            "rule_code, rule_name, summary_alert_id, countries_detected."
        )


def _truthy_contains(row: Any, cols: List[str], terms: str) -> bool:
    haystack = " ".join(str(row.get(c) or "") for c in cols if c in row).upper()
    return any(t in haystack for t in terms.split("|"))


def _build_alert_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Replicate training's _build_rows() + _feature_frame() pipeline.

    Input: rule summary CSV (ARTIFACT_RULE_SUMMARY_CSV schema).
    Derives the same boolean flags and applies pd.get_dummies exactly as training.
    """
    df = df.copy()

    # Derive countries_count from pipe-separated countries_detected
    if "countries_count" not in df.columns:
        if "countries_detected" in df.columns:
            df["countries_count"] = df["countries_detected"].apply(
                lambda v: len([x for x in str(v or "").split("|") if x.strip()])
            )
        else:
            df["countries_count"] = 0

    if "has_multiple_countries" not in df.columns:
        df["has_multiple_countries"] = df["countries_count"] > 1

    if "is_high_risk_rule" not in df.columns:
        rl = df.get("risk_level", pd.Series("", index=df.index)).fillna("")
        df["is_high_risk_rule"] = rl.str.upper() == "HIGH"

    if "duration_minutes" not in df.columns and "window_start" in df.columns and "window_end" in df.columns:
        def _dur(row: Any) -> Optional[float]:
            s = pd.to_datetime(row.get("window_start"), errors="coerce", utc=True)
            e = pd.to_datetime(row.get("window_end"), errors="coerce", utc=True)
            if pd.isna(s) or pd.isna(e):
                return None
            return round(float((e - s).total_seconds() / 60.0), 3)
        df["duration_minutes"] = df.apply(_dur, axis=1)

    _flag_defs = [
        ("is_velocity_rule",        ["rule_code", "rule_name"],                          "VELOCITY|FREQUENCY|BURST|RAPID"),
        ("is_double_country_rule",  ["rule_code", "rule_name"],                          "DOUBLE_COUNTRY|MULTIPLE_COUNTRY|COUNTRY"),
        ("is_mcc_risk_rule",        ["rule_code", "rule_name"],                          "MCC|RUBRO|MERCHANT"),
        ("is_card_present_rule",    ["rule_code", "rule_name"],                          "CARD_PRESENT|PRESENT"),
        ("is_card_absent_rule",     ["rule_code", "rule_name"],                          "CARD_ABSENT|NOT_PRESENT|ABSENT"),
        ("is_internet_related",     ["rule_code", "rule_name", "merchant_rubro_values"], "INTERNET|ONLINE|WEB|ECOM"),
        ("is_atm_or_cash_related",  ["rule_code", "rule_name", "merchant_rubro_values"], "ATM|CASH|CAJERO"),
    ]
    for flag, cols, terms in _flag_defs:
        if flag not in df.columns:
            df[flag] = df.apply(lambda r, c=cols, t=terms: _truthy_contains(r, c, t), axis=1)

    # Mirror _feature_frame() from train_human_supervised_model.py
    X = df.drop(columns=[c for c in _DROP_FOR_MODEL if c in df.columns]).copy()
    for col in X.columns:
        if X[col].dtype == "bool":
            X[col] = X[col].astype(int)
    X = pd.get_dummies(X, dummy_na=True)
    return X.apply(pd.to_numeric, errors="coerce").fillna(0)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _processed_dir() -> Path:
    return default_processed_dir()


# ── List helpers ──────────────────────────────────────────────────────────────

def list_trained_models(db: Session) -> List[Dict[str, Any]]:
    """Return SUPERVISED_HUMAN models with status AVAILABLE from model_registry."""
    rows = (
        db.query(ModelRegistry)
        .filter(
            ModelRegistry.model_family == "SUPERVISED_HUMAN",
            ModelRegistry.status == "AVAILABLE",
        )
        .order_by(ModelRegistry.created_at.desc())
        .all()
    )
    result = []
    for row in rows:
        metrics: Dict[str, Any] = {}
        if row.metrics_json:
            try:
                raw = json.loads(row.metrics_json)
                metrics = raw.get("metrics", raw)
            except Exception:
                pass
        result.append({
            "id": row.id,
            "algorithm": row.algorithm,
            "source_run": row.source_run,
            "status": row.status,
            "is_active": row.is_active,
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "f1_score": metrics.get("f1_score"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "roc_auc": metrics.get("roc_auc"),
            "accuracy": metrics.get("accuracy"),
        })
    return result


def list_preprocessed_runs(db: Session) -> List[Dict[str, Any]]:
    """Return completed preprocessing runs (kept for compatibility)."""
    rows = (
        db.query(PreprocessingRun)
        .filter(PreprocessingRun.status == "COMPLETED")
        .order_by(PreprocessingRun.finished_at.desc())
        .all()
    )
    return [
        {
            "id": row.id,
            "source_run": f"preprocessed_run_{row.id}",
            "output_file_path": row.output_file_path,
            "total_records": row.total_records,
            "finished_at": row.finished_at.isoformat() if row.finished_at else None,
            "status": row.status,
        }
        for row in rows
    ]


def list_rule_summaries(db: Session) -> List[Dict[str, Any]]:
    """Return available RULE_SUMMARY_CSV artifacts usable as supervised inference input."""
    rows = (
        db.query(ArtifactRegistry)
        .filter(
            ArtifactRegistry.artifact_type == artifacts.ARTIFACT_RULE_SUMMARY_CSV,
            ArtifactRegistry.status == "AVAILABLE",
        )
        .order_by(ArtifactRegistry.updated_at.desc())
        .all()
    )
    result = []
    for row in rows:
        file_exists = row.file_path and Path(row.file_path).exists()
        result.append({
            "id": row.id,
            "source_run": row.source_run,
            "file_path": row.file_path,
            "file_name": row.file_name,
            "row_count": row.row_count,
            "status": row.status,
            "file_exists": file_exists,
            "created_at": row.created_at.isoformat() if row.created_at else None,
        })
    return result


def list_prediction_runs(db: Session) -> List[Dict[str, Any]]:
    """Return all supervised inference runs ordered by creation date."""
    rows = (
        db.query(SupervisedInferenceRun)
        .order_by(SupervisedInferenceRun.created_at.desc())
        .all()
    )
    return [_run_to_dict(r) for r in rows]


def _run_to_dict(row: SupervisedInferenceRun) -> Dict[str, Any]:
    return {
        "id": row.id,
        "algorithm": row.algorithm,
        "model_source_run": row.model_source_run,
        "input_type": row.input_type,
        "input_source": row.input_source,
        "total_analyzed": row.total_analyzed,
        "high_count": row.high_count,
        "medium_count": row.medium_count,
        "low_count": row.low_count,
        "status": row.status,
        "error_message": row.error_message,
        "started_at": row.started_at.isoformat() if row.started_at else None,
        "finished_at": row.finished_at.isoformat() if row.finished_at else None,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "model_registry_id": row.model_registry_id,
    }


# ── Async pattern (matching unsupervised_inference_service) ───────────────────

def create_pending_run(
    db: Session,
    model_registry_id: int,
    input_file_path: str,
    input_type: str,
    input_source: str,
) -> SupervisedInferenceRun:
    """Create a SupervisedInferenceRun in PENDING state and return it."""
    run = SupervisedInferenceRun(
        model_registry_id=model_registry_id,
        algorithm="unknown",
        model_source_run="",
        input_type=input_type,
        input_source=input_source,
        input_file=input_file_path,
        status="PENDING",
        started_at=_utc_now(),
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


def run_inference_background(run_id: int) -> None:
    """Background task: runs supervised inference and updates the run record."""
    from backend.app.database import SessionLocal  # local import to avoid circular deps

    db = SessionLocal()
    run_record = None
    try:
        run_record = db.query(SupervisedInferenceRun).filter(SupervisedInferenceRun.id == run_id).first()
        if run_record is None:
            return
        run_record.status = "RUNNING"
        db.commit()

        result = _run_inference(
            db,
            run_record.model_registry_id,
            run_record.input_file,
            run_record.input_type,
            run_record.input_source,
        )
        run_record.algorithm = result["algorithm"]
        run_record.model_source_run = result["model_source_run"]
        run_record.results_file = result["results_file"]
        run_record.metadata_file = result.get("metadata_file", "")
        run_record.total_analyzed = result["total_analyzed"]
        run_record.high_count = result["high_count"]
        run_record.medium_count = result["medium_count"]
        run_record.low_count = result["low_count"]
        run_record.params_json = json.dumps({"same_run_warning": result.get("same_run_warning")})
        run_record.status = "COMPLETED"
        run_record.finished_at = _utc_now()
        db.commit()
        db.refresh(run_record)

        artifacts.register_or_update_artifact(
            db,
            artifact_type=artifacts.ARTIFACT_SUPERVISED_INFERENCE_SCORES,
            phase=artifacts.PHASE_C4,
            source_run=f"supervised_inference_run_{run_record.id}",
            run_token=str(run_record.id),
            file_path=result["results_file"],
            metadata={
                "algorithm": result["algorithm"],
                "model_registry_id": run_record.model_registry_id,
                "input_type": run_record.input_type,
                "methodology_warning": METHODOLOGY_WARNING,
            },
        )
        db.commit()

    except Exception as exc:
        try:
            if run_record is None:
                run_record = db.query(SupervisedInferenceRun).filter(SupervisedInferenceRun.id == run_id).first()
            if run_record:
                run_record.status = "FAILED"
                run_record.error_message = str(exc)[:2000]
                run_record.finished_at = _utc_now()
                db.commit()
        except Exception:
            db.rollback()
    finally:
        db.close()


def _load_same_run_warning(run: SupervisedInferenceRun) -> Optional[str]:
    if run.params_json:
        try:
            return json.loads(run.params_json).get("same_run_warning")
        except Exception:
            pass
    return None


def get_run_status(db: Session, run_id: int) -> Dict[str, Any]:
    """Return current status of an inference run (used for polling)."""
    run = db.query(SupervisedInferenceRun).filter(SupervisedInferenceRun.id == run_id).first()
    if run is None:
        raise ValueError(f"Supervised inference run {run_id} not found")
    return {
        "id": run.id,
        "status": run.status,
        "error_message": run.error_message,
        "total_analyzed": run.total_analyzed,
        "high_count": run.high_count,
        "medium_count": run.medium_count,
        "low_count": run.low_count,
        "algorithm": run.algorithm,
        "input_source": run.input_source,
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "finished_at": run.finished_at.isoformat() if run.finished_at else None,
        "methodology_warning": METHODOLOGY_WARNING,
        "same_run_warning": _load_same_run_warning(run),
    }


# ── Core inference ────────────────────────────────────────────────────────────

def _assign_priority(prob: Optional[float], label: int) -> str:
    """Assign HIGH/MEDIUM/LOW based on probability or fallback to label."""
    if prob is None:
        return "HIGH" if label == 1 else "LOW"
    if prob >= PRIORITY_HIGH_THRESHOLD:
        return "HIGH"
    if prob >= PRIORITY_MEDIUM_THRESHOLD:
        return "MEDIUM"
    return "LOW"


def _run_inference(
    db: Session,
    model_registry_id: int,
    input_file_path: str,
    input_type: str,
    input_source: str,
) -> Dict[str, Any]:
    reg = db.query(ModelRegistry).filter(ModelRegistry.id == model_registry_id).first()
    if reg is None:
        raise ValueError(f"Model registry entry {model_registry_id} not found")
    if reg.status != "AVAILABLE":
        raise ValueError(f"Model {model_registry_id} is not AVAILABLE (status={reg.status})")
    if not reg.model_file or not Path(reg.model_file).exists():
        raise FileNotFoundError(f"Model file not found: {reg.model_file}")

    # Load model artifact — saved as {"model": pipeline, "feature_columns": [...]}
    artifact = joblib.load(reg.model_file)
    if isinstance(artifact, dict):
        pipeline = artifact["model"]
        saved_feature_columns: Optional[List[str]] = artifact.get("feature_columns")
    else:
        pipeline = artifact
        saved_feature_columns = None

    processed_dir = _processed_dir()

    raw_df = pd.read_csv(input_file_path)
    if raw_df.empty:
        raise ValueError("El dataset de entrada está vacío.")

    # Build alert-level feature frame using same pipeline as training
    feature_frame = _build_alert_feature_frame(raw_df)
    if feature_frame.empty:
        raise ValueError("No se pudieron generar features del dataset de entrada.")

    # Reindex to exactly the columns the model was trained on, filling unseen one-hot cols with 0
    if saved_feature_columns:
        X = feature_frame.reindex(columns=saved_feature_columns, fill_value=0)
    elif hasattr(pipeline, "feature_names_in_"):
        X = feature_frame.reindex(columns=list(pipeline.feature_names_in_), fill_value=0)
    else:
        X = feature_frame
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Predict
    labels = pipeline.predict(X)

    # Probabilities (class=1 column)
    probs: Optional[np.ndarray] = None
    if hasattr(pipeline, "predict_proba"):
        try:
            proba_matrix = pipeline.predict_proba(X)
            if proba_matrix.shape[1] >= 2:
                probs = proba_matrix[:, 1]
        except Exception:
            pass

    n_rows = len(labels)
    # Preserve alert-level context columns in result (analyst identification)
    result = pd.DataFrame(index=range(n_rows))
    for col in _CONTEXT_OUTPUT_COLS:
        if col in raw_df.columns:
            vals = raw_df[col].values
            result[col] = vals[:n_rows] if len(vals) >= n_rows else list(vals) + [None] * (n_rows - len(vals))

    result["prediction_label"] = labels.astype(int)
    result["prediction_probability"] = probs.round(6).astype(float) if probs is not None else None
    result["priority_level"] = [
        _assign_priority(float(probs[i]) if probs is not None else None, int(labels[i]))
        for i in range(n_rows)
    ]
    result["model_name"] = reg.algorithm
    result["source_run"] = reg.source_run

    # Enforce no forbidden columns
    for col in list(FORBIDDEN_OUTPUT_COLUMNS):
        if col in result.columns:
            result = result.drop(columns=[col])

    high_count = int((result["priority_level"] == "HIGH").sum())
    medium_count = int((result["priority_level"] == "MEDIUM").sum())
    low_count = int((result["priority_level"] == "LOW").sum())

    run_token_val = normalize_run_token(f"run_{int(_utc_now().timestamp())}")
    output_path = processed_dir / f"supervised_inference_{reg.algorithm}_{run_token_val}.csv"
    result.to_csv(output_path, index=False)

    # Warn if inference runs on the same source_run used to train the model
    model_sr = normalize_run_token(reg.source_run or "")
    infer_sr = normalize_run_token(input_source or "")
    same_run_warning: Optional[str] = None
    if model_sr and infer_sr and model_sr == infer_sr:
        same_run_warning = (
            "Este run coincide con el usado para entrenar el modelo; "
            "los resultados sirven como validación técnica, no como predicción sobre datos nuevos."
        )

    return {
        "algorithm": reg.algorithm,
        "model_source_run": reg.source_run,
        "results_file": str(output_path),
        "total_analyzed": n_rows,
        "high_count": high_count,
        "medium_count": medium_count,
        "low_count": low_count,
        "same_run_warning": same_run_warning,
    }


# ── Read results ──────────────────────────────────────────────────────────────

def get_prediction_results(
    db: Session,
    run_id: int,
    page: int = 1,
    page_size: int = 50,
    priority_filter: str = "ALL",
    sort_by: str = "prediction_probability",
) -> Dict[str, Any]:
    run = db.query(SupervisedInferenceRun).filter(SupervisedInferenceRun.id == run_id).first()
    if run is None:
        raise ValueError(f"Supervised inference run {run_id} not found")
    if run.status != "COMPLETED" or not run.results_file:
        return {"run_id": run_id, "status": run.status, "rows": [], "total": 0, "page": page, "page_size": page_size}

    df = pd.read_csv(run.results_file)

    for col in list(FORBIDDEN_OUTPUT_COLUMNS):
        if col in df.columns:
            df = df.drop(columns=[col])

    if priority_filter and priority_filter != "ALL" and "priority_level" in df.columns:
        df = df[df["priority_level"] == priority_filter]

    if sort_by in df.columns:
        ascending = sort_by not in ("prediction_probability",)
        df = df.sort_values(sort_by, ascending=ascending, na_position="last")

    total = len(df)
    page = max(1, page)
    offset = (page - 1) * page_size
    page_df = df.iloc[offset: offset + page_size]

    return {
        "run_id": run_id,
        "status": run.status,
        "total": total,
        "page": page,
        "page_size": page_size,
        "rows": page_df.where(pd.notnull(page_df), None).to_dict(orient="records"),
        "methodology_warning": METHODOLOGY_WARNING,
    }


def get_prediction_metadata(db: Session, run_id: int) -> Dict[str, Any]:
    run = db.query(SupervisedInferenceRun).filter(SupervisedInferenceRun.id == run_id).first()
    if run is None:
        raise ValueError(f"Supervised inference run {run_id} not found")

    model_info: Dict[str, Any] = {}
    if run.model_registry_id:
        reg = db.query(ModelRegistry).filter(ModelRegistry.id == run.model_registry_id).first()
        if reg:
            metrics: Dict[str, Any] = {}
            if reg.metrics_json:
                try:
                    raw = json.loads(reg.metrics_json)
                    metrics = raw.get("metrics", raw)
                except Exception:
                    pass
            model_info = {
                "model_registry_id": reg.id,
                "algorithm": reg.algorithm,
                "source_run": reg.source_run,
                "status": reg.status,
                "metrics": metrics,
            }

    return {
        "run": _run_to_dict(run),
        "model": model_info,
        "methodology_warning": METHODOLOGY_WARNING,
        "same_run_warning": _load_same_run_warning(run),
    }


def get_prediction_report(db: Session, run_id: int) -> Dict[str, Any]:
    run = db.query(SupervisedInferenceRun).filter(SupervisedInferenceRun.id == run_id).first()
    if run is None:
        raise ValueError(f"Supervised inference run {run_id} not found")

    priority_distribution: List[Dict[str, Any]] = []
    prob_distribution: List[Dict[str, Any]] = []

    if run.status == "COMPLETED" and run.results_file and Path(run.results_file).exists():
        try:
            df = pd.read_csv(run.results_file)
            if "priority_level" in df.columns:
                counts = df["priority_level"].value_counts()
                for level in ("HIGH", "MEDIUM", "LOW"):
                    priority_distribution.append({"level": level, "count": int(counts.get(level, 0))})
            if "prediction_probability" in df.columns:
                valid = df["prediction_probability"].dropna()
                if len(valid) > 0:
                    bins = np.histogram(valid, bins=10, range=(0.0, 1.0))
                    prob_distribution = [
                        {
                            "bucket": f"{round(float(bins[1][i]), 2)}–{round(float(bins[1][i+1]), 2)}",
                            "count": int(bins[0][i]),
                        }
                        for i in range(len(bins[0]))
                    ]
        except Exception:
            pass

    return {
        "run_id": run_id,
        "algorithm": run.algorithm,
        "model_source_run": run.model_source_run,
        "input_type": run.input_type,
        "input_source": run.input_source,
        "total_analyzed": run.total_analyzed,
        "high_count": run.high_count,
        "medium_count": run.medium_count,
        "low_count": run.low_count,
        "status": run.status,
        "priority_distribution": priority_distribution,
        "prob_distribution": prob_distribution,
        "methodology_warning": METHODOLOGY_WARNING,
    }

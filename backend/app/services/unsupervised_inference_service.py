"""Service for applying trained unsupervised models to new datasets.

Models are LOADED from ModelRegistry — never retrained.
No is_fraud or confirmed_fraud is generated.
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
    UnsupervisedInferenceRun,
)
from backend.app.services import artifact_registry_service as artifacts
from backend.app.ml.unsupervised_feature_builder import (
    build_unsupervised_features,
    default_models_dir,
    default_processed_dir,
    normalize_run_token,
)

ARTIFACT_UNSUPERVISED_INFERENCE_SCORES = "UNSUPERVISED_INFERENCE_SCORES"

FORBIDDEN_OUTPUT_COLUMNS = {
    "is_fraud",
    "confirmed_fraud",
    "target_is_fraud",
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
}

METHODOLOGY_WARNING = (
    "Las anomalías detectadas por modelos no supervisados representan comportamientos "
    "atípicos y no constituyen fraude confirmado."
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _processed_dir() -> Path:
    return default_processed_dir()


def _models_dir() -> Path:
    return default_models_dir()


# ── List helpers ──────────────────────────────────────────────────────────────

def list_trained_models(db: Session) -> List[Dict[str, Any]]:
    """Return UNSUPERVISED models with status AVAILABLE from model_registry."""
    rows = (
        db.query(ModelRegistry)
        .filter(
            ModelRegistry.model_family == "UNSUPERVISED",
            ModelRegistry.status == "AVAILABLE",
        )
        .order_by(ModelRegistry.created_at.desc())
        .all()
    )
    result = []
    for row in rows:
        metrics = {}
        if row.metrics_json:
            try:
                raw = json.loads(row.metrics_json)
                # IF stores metrics nested under "metadata"; AE stores them at top level
                metrics = raw.get("metadata", raw) if isinstance(raw.get("metadata"), dict) else raw
            except Exception:
                pass
        result.append({
            "id": row.id,
            "algorithm": row.algorithm,
            "source_run": row.source_run,
            "status": row.status,
            "is_active": row.is_active,
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "anomaly_rate": metrics.get("anomaly_rate"),
            "contamination": metrics.get("contamination"),
            "total_records": metrics.get("total_records"),
        })
    return result


def list_preprocessed_runs(db: Session) -> List[Dict[str, Any]]:
    """Return completed preprocessing runs available as input datasets."""
    rows = (
        db.query(PreprocessingRun)
        .filter(PreprocessingRun.status == "COMPLETED")
        .order_by(PreprocessingRun.finished_at.desc())
        .all()
    )
    result = []
    for row in rows:
        result.append({
            "id": row.id,
            "output_file_path": row.output_file_path,
            "total_records": row.total_records,
            "finished_at": row.finished_at.isoformat() if row.finished_at else None,
            "status": row.status,
        })
    return result


def list_prediction_runs(db: Session) -> List[Dict[str, Any]]:
    """Return all unsupervised inference runs ordered by creation date."""
    rows = (
        db.query(UnsupervisedInferenceRun)
        .order_by(UnsupervisedInferenceRun.created_at.desc())
        .all()
    )
    return [_run_to_dict(r) for r in rows]


def _run_to_dict(row: UnsupervisedInferenceRun) -> Dict[str, Any]:
    return {
        "id": row.id,
        "algorithm": row.algorithm,
        "model_source_run": row.model_source_run,
        "input_type": row.input_type,
        "input_source": row.input_source,
        "total_analyzed": row.total_analyzed,
        "anomaly_count": row.anomaly_count,
        "anomaly_rate": row.anomaly_rate,
        "status": row.status,
        "error_message": row.error_message,
        "started_at": row.started_at.isoformat() if row.started_at else None,
        "finished_at": row.finished_at.isoformat() if row.finished_at else None,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "model_registry_id": row.model_registry_id,
    }


# ── Apply model ───────────────────────────────────────────────────────────────

def apply_model(
    db: Session,
    model_registry_id: int,
    input_file_path: str,
    input_type: str,
    input_source: str,
) -> Dict[str, Any]:
    """Apply a trained unsupervised model to a dataset. Never retrains.

    Returns a dict with the UnsupervisedInferenceRun id and summary stats.
    """
    started_at = _utc_now()

    run_record = UnsupervisedInferenceRun(
        model_registry_id=model_registry_id,
        algorithm="unknown",
        model_source_run="",
        input_type=input_type,
        input_source=input_source,
        input_file=input_file_path,
        status="RUNNING",
        started_at=started_at,
    )
    db.add(run_record)
    db.commit()
    db.refresh(run_record)

    try:
        result = _run_inference(db, model_registry_id, input_file_path, input_type, input_source)
        run_record.algorithm = result["algorithm"]
        run_record.model_source_run = result["model_source_run"]
        run_record.results_file = result["results_file"]
        run_record.metadata_file = result.get("metadata_file", "")
        run_record.total_analyzed = result["total_analyzed"]
        run_record.anomaly_count = result["anomaly_count"]
        run_record.anomaly_rate = result["anomaly_rate"]
        run_record.status = "COMPLETED"
        run_record.finished_at = _utc_now()
        db.commit()
        db.refresh(run_record)

        # Register result artifact
        run_token = normalize_run_token(f"inference_run_{run_record.id}")
        artifacts.register_or_update_artifact(
            db,
            artifact_type=ARTIFACT_UNSUPERVISED_INFERENCE_SCORES,
            phase=artifacts.PHASE_C3,
            source_run=f"unsupervised_inference_run_{run_record.id}",
            run_token=str(run_record.id),
            file_path=result["results_file"],
            metadata={
                "algorithm": result["algorithm"],
                "model_registry_id": model_registry_id,
                "input_type": input_type,
                "methodology_warning": METHODOLOGY_WARNING,
            },
        )
        db.commit()

        return {**_run_to_dict(run_record), "methodology_warning": METHODOLOGY_WARNING}

    except Exception as exc:
        run_record.status = "FAILED"
        run_record.error_message = str(exc)[:2000]
        run_record.finished_at = _utc_now()
        try:
            db.commit()
        except Exception:
            db.rollback()
        raise


def create_pending_run(
    db: Session,
    model_registry_id: int,
    input_file_path: str,
    input_type: str,
    input_source: str,
) -> UnsupervisedInferenceRun:
    """Create an UnsupervisedInferenceRun in PENDING state and return it."""
    run = UnsupervisedInferenceRun(
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
    """Background task: runs inference and updates the run record. Uses its own DB session."""
    from backend.app.database import SessionLocal  # local import to avoid circular deps

    db = SessionLocal()
    run_record = None
    try:
        run_record = db.query(UnsupervisedInferenceRun).filter(UnsupervisedInferenceRun.id == run_id).first()
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
        run_record.anomaly_count = result["anomaly_count"]
        run_record.anomaly_rate = result["anomaly_rate"]
        run_record.status = "COMPLETED"
        run_record.finished_at = _utc_now()
        db.commit()
        db.refresh(run_record)

        artifacts.register_or_update_artifact(
            db,
            artifact_type=ARTIFACT_UNSUPERVISED_INFERENCE_SCORES,
            phase=artifacts.PHASE_C3,
            source_run=f"unsupervised_inference_run_{run_record.id}",
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
                run_record = db.query(UnsupervisedInferenceRun).filter(UnsupervisedInferenceRun.id == run_id).first()
            if run_record:
                run_record.status = "FAILED"
                run_record.error_message = str(exc)[:2000]
                run_record.finished_at = _utc_now()
                db.commit()
        except Exception:
            db.rollback()
    finally:
        db.close()


def get_run_status(db: Session, run_id: int) -> Dict[str, Any]:
    """Return current status of an inference run (used for polling)."""
    run = db.query(UnsupervisedInferenceRun).filter(UnsupervisedInferenceRun.id == run_id).first()
    if run is None:
        raise ValueError(f"Inference run {run_id} not found")
    return {
        "id": run.id,
        "status": run.status,
        "error_message": run.error_message,
        "total_analyzed": run.total_analyzed,
        "anomaly_count": run.anomaly_count,
        "anomaly_rate": run.anomaly_rate,
        "algorithm": run.algorithm,
        "input_source": run.input_source,
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "finished_at": run.finished_at.isoformat() if run.finished_at else None,
        "methodology_warning": METHODOLOGY_WARNING,
    }


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

    algorithm = reg.algorithm.lower()

    metadata: Dict[str, Any] = {}
    if reg.metrics_json:
        try:
            metadata = json.loads(reg.metrics_json)
        except Exception:
            pass

    contamination = float(metadata.get("contamination", 0.01))

    # ── Load and validate input CSV ───────────────────────────────────────────
    raw_df = pd.read_csv(input_file_path)
    if raw_df.empty:
        raise ValueError("El dataset de entrada está vacío.")

    # Build unsupervised features (reuses the same pipeline as training)
    processed_dir = _processed_dir()
    inference_run_label = f"inference_{Path(input_file_path).stem}"
    try:
        feature_file, _ = build_unsupervised_features(
            input_file_path, inference_run_label, processed_dir
        )
    except Exception as exc:
        raise ValueError(f"No se pudieron construir las features del dataset: {exc}") from exc

    feature_frame = pd.read_csv(feature_file)
    if feature_frame.empty:
        raise ValueError("No se pudieron generar features del dataset de entrada.")

    # ── Run inference per algorithm ───────────────────────────────────────────
    if algorithm == "isolation_forest":
        score_frame = _apply_isolation_forest(reg, feature_frame, raw_df, contamination)
    elif algorithm == "autoencoder_pytorch":
        score_frame = _apply_autoencoder(db, reg, feature_frame, raw_df, contamination)
    else:
        raise ValueError(f"Algoritmo no soportado para inferencia: {algorithm}")

    # Enforce no forbidden columns in output
    for col in list(FORBIDDEN_OUTPUT_COLUMNS):
        if col in score_frame.columns:
            score_frame = score_frame.drop(columns=[col])

    total = len(score_frame)
    flag_col = "anomaly_flag" if "anomaly_flag" in score_frame.columns else "autoencoder_anomaly_flag"
    anomaly_count = int(score_frame[flag_col].sum()) if flag_col in score_frame.columns else 0
    anomaly_rate = round(anomaly_count / total, 6) if total > 0 else 0.0

    # Save results CSV
    run_token_val = normalize_run_token(f"run_{int(_utc_now().timestamp())}")
    output_path = processed_dir / f"unsupervised_inference_{algorithm}_{run_token_val}.csv"
    score_frame.to_csv(output_path, index=False)

    return {
        "algorithm": reg.algorithm,
        "model_source_run": reg.source_run,
        "results_file": str(output_path),
        "total_analyzed": total,
        "anomaly_count": anomaly_count,
        "anomaly_rate": anomaly_rate,
    }


def _apply_isolation_forest(
    reg: ModelRegistry,
    feature_frame: pd.DataFrame,
    raw_df: pd.DataFrame,
    contamination: float,
) -> pd.DataFrame:
    pipeline = joblib.load(reg.model_file)

    try:
        raw_scores = -pipeline.decision_function(feature_frame)
    except Exception:
        try:
            raw_scores = -pipeline.score_samples(feature_frame)
        except Exception:
            preds = pipeline.predict(feature_frame)
            raw_scores = np.where(preds == -1, 1.0, 0.0).astype(float)

    threshold = float(np.percentile(raw_scores, 100.0 * (1.0 - contamination)))
    flags = (raw_scores >= threshold).astype(int)
    if flags.sum() == 0 and len(flags) > 0:
        flags[int(np.argmax(raw_scores))] = 1

    ranks = pd.Series(raw_scores).rank(method="first", ascending=False).astype(int).to_numpy()

    result = _build_context_frame(raw_df, len(raw_scores))
    result["anomaly_score"] = raw_scores.astype(float)
    result["anomaly_flag"] = flags.astype(int)
    result["anomaly_rank"] = ranks.astype(int)
    return result.sort_values("anomaly_rank").reset_index(drop=True)


def _apply_autoencoder(
    db: Session,
    reg: ModelRegistry,
    feature_frame: pd.DataFrame,
    raw_df: pd.DataFrame,
    contamination: float,
) -> pd.DataFrame:
    from backend.app.ml.autoencoder_anomaly import (
        AutoencoderTabular,
        AutoencoderDependencyError,
        _require_torch,
        FORBIDDEN_COLUMNS,
        CONTEXT_COLUMNS,
    )

    torch, _, _, _ = _require_torch()

    # Load scaler bundle from artifact_registry
    scaler_artifact = db.query(ArtifactRegistry).filter(
        ArtifactRegistry.source_run == reg.source_run,
        ArtifactRegistry.artifact_type == artifacts.ARTIFACT_MODEL_SCALER,
    ).first()
    if scaler_artifact is None or not Path(scaler_artifact.file_path).exists():
        raise FileNotFoundError(
            f"Scaler artifact not found for model source_run={reg.source_run}. "
            "Ensure the autoencoder was trained with artifact registration."
        )
    scaler_bundle = joblib.load(scaler_artifact.file_path)
    imputer = scaler_bundle["imputer"]
    scaler = scaler_bundle["scaler"]

    # Prepare feature matrix using same column selection as training
    numeric_cols = [
        col for col in feature_frame.select_dtypes(include=[np.number, "bool"]).columns
        if col.lower() not in {c.lower() for c in FORBIDDEN_COLUMNS}
        and col.lower() not in {c.lower() for c in CONTEXT_COLUMNS}
        and not col.lower().endswith("_id")
        and col.lower() != "id"
        and "fraud" not in col.lower()
        and "review" not in col.lower()
        and not col.lower().startswith("rule_")
    ]
    if not numeric_cols:
        raise ValueError("No se encontraron columnas numéricas válidas para el autoencoder.")

    matrix = feature_frame[numeric_cols].replace([np.inf, -np.inf], np.nan)
    try:
        X_imputed = imputer.transform(matrix)
        X_scaled = scaler.transform(X_imputed).astype(np.float32)
    except Exception as exc:
        raise ValueError(f"Error al aplicar scaler del modelo: {exc}") from exc

    # Load checkpoint first to read saved architecture params
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(reg.model_file, map_location=device_name)
    # Saved as bundle {"state_dict": ..., "input_dim": ..., "latent_dim": ..., "feature_columns": ...}
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        actual_state = checkpoint["state_dict"]
        latent_dim = int(checkpoint.get("latent_dim", 16))
    else:
        # Fallback: plain state_dict; derive latent_dim from metrics_json
        actual_state = checkpoint
        metadata_info: Dict[str, Any] = {}
        if reg.metrics_json:
            try:
                metadata_info = json.loads(reg.metrics_json)
            except Exception:
                pass
        latent_dim = int(metadata_info.get("latent_dim", 16))

    model = AutoencoderTabular(input_dim=X_scaled.shape[1], latent_dim=latent_dim)
    model.to(device_name)
    model.module.load_state_dict(actual_state)

    from backend.app.ml.autoencoder_anomaly import compute_reconstruction_errors
    score_parts = compute_reconstruction_errors(model, X_scaled, contamination=contamination, device=device_name)

    result = _build_context_frame(raw_df, len(X_scaled))
    result["reconstruction_error"] = score_parts["reconstruction_error"].astype(float)
    result["autoencoder_anomaly_score"] = score_parts["autoencoder_anomaly_score"].astype(float)
    result["autoencoder_anomaly_flag"] = score_parts["autoencoder_anomaly_flag"].astype(int)
    result["anomaly_rank"] = score_parts["anomaly_rank"].astype(int)
    return result.sort_values("anomaly_rank").reset_index(drop=True)


def _build_context_frame(raw_df: pd.DataFrame, n_rows: int) -> pd.DataFrame:
    """Build a context DataFrame with safe identifier columns."""
    context_cols = ["transaction_id", "customer_hash", "transaction_datetime", "amount",
                    "country_code", "pos_entry_mode", "merchant_rubro_proxy"]
    frame = pd.DataFrame(index=range(n_rows))
    for col in context_cols:
        if col in raw_df.columns:
            values = raw_df[col].values
            frame[col] = values[:n_rows] if len(values) >= n_rows else list(values) + [None] * (n_rows - len(values))
    return frame


# ── Read results ──────────────────────────────────────────────────────────────

def get_prediction_results(
    db: Session,
    run_id: int,
    page: int = 1,
    page_size: int = 50,
    anomaly_only: bool = False,
    sort_by: str = "anomaly_rank",
) -> Dict[str, Any]:
    run = db.query(UnsupervisedInferenceRun).filter(UnsupervisedInferenceRun.id == run_id).first()
    if run is None:
        raise ValueError(f"Inference run {run_id} not found")
    if run.status != "COMPLETED" or not run.results_file:
        return {"run_id": run_id, "status": run.status, "rows": [], "total": 0, "page": page, "page_size": page_size}

    df = pd.read_csv(run.results_file)

    # Strip forbidden columns
    for col in list(FORBIDDEN_OUTPUT_COLUMNS):
        if col in df.columns:
            df = df.drop(columns=[col])

    # Filter
    flag_col = "anomaly_flag" if "anomaly_flag" in df.columns else "autoencoder_anomaly_flag"
    if anomaly_only and flag_col in df.columns:
        df = df[df[flag_col] == 1]

    # Sort
    if sort_by in df.columns:
        ascending = sort_by not in ("anomaly_score", "reconstruction_error", "autoencoder_anomaly_score")
        df = df.sort_values(sort_by, ascending=ascending)
    elif "anomaly_rank" in df.columns:
        df = df.sort_values("anomaly_rank")

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
    run = db.query(UnsupervisedInferenceRun).filter(UnsupervisedInferenceRun.id == run_id).first()
    if run is None:
        raise ValueError(f"Inference run {run_id} not found")

    model_info: Dict[str, Any] = {}
    if run.model_registry_id:
        reg = db.query(ModelRegistry).filter(ModelRegistry.id == run.model_registry_id).first()
        if reg:
            metrics = {}
            if reg.metrics_json:
                try:
                    metrics = json.loads(reg.metrics_json)
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
    }


def get_prediction_report(db: Session, run_id: int) -> Dict[str, Any]:
    run = db.query(UnsupervisedInferenceRun).filter(UnsupervisedInferenceRun.id == run_id).first()
    if run is None:
        raise ValueError(f"Inference run {run_id} not found")

    score_distribution: List[Dict[str, Any]] = []
    if run.status == "COMPLETED" and run.results_file and Path(run.results_file).exists():
        try:
            df = pd.read_csv(run.results_file)
            score_col = next(
                (c for c in ["anomaly_score", "reconstruction_error", "autoencoder_anomaly_score"] if c in df.columns),
                None,
            )
            if score_col:
                bins = np.histogram(df[score_col].dropna(), bins=10)
                score_distribution = [
                    {"bucket": f"{round(float(bins[1][i]), 4)}–{round(float(bins[1][i+1]), 4)}", "count": int(bins[0][i])}
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
        "anomaly_count": run.anomaly_count,
        "anomaly_rate": run.anomaly_rate,
        "status": run.status,
        "score_distribution": score_distribution,
        "methodology_warning": METHODOLOGY_WARNING,
    }


def compare_inference_runs(db: Session, run_id_a: int, run_id_b: int) -> Dict[str, Any]:
    """Compare two completed inference runs. Returns intersection of flagged anomalies.

    Intended for fraud analysts: transactions flagged by BOTH models are stronger signals.
    """
    run_a = db.query(UnsupervisedInferenceRun).filter(UnsupervisedInferenceRun.id == run_id_a).first()
    run_b = db.query(UnsupervisedInferenceRun).filter(UnsupervisedInferenceRun.id == run_id_b).first()
    if run_a is None:
        raise ValueError(f"Run {run_id_a} not found")
    if run_b is None:
        raise ValueError(f"Run {run_id_b} not found")
    if run_a.status != "COMPLETED":
        raise ValueError(f"Run {run_id_a} no está COMPLETED (status={run_a.status})")
    if run_b.status != "COMPLETED":
        raise ValueError(f"Run {run_id_b} no está COMPLETED (status={run_b.status})")
    if not run_a.results_file or not Path(run_a.results_file).exists():
        raise FileNotFoundError(f"Archivo de resultados no encontrado para run {run_id_a}")
    if not run_b.results_file or not Path(run_b.results_file).exists():
        raise FileNotFoundError(f"Archivo de resultados no encontrado para run {run_id_b}")

    df_a = pd.read_csv(run_a.results_file)
    df_b = pd.read_csv(run_b.results_file)

    for col in list(FORBIDDEN_OUTPUT_COLUMNS):
        df_a = df_a.drop(columns=[col], errors="ignore")
        df_b = df_b.drop(columns=[col], errors="ignore")

    if "transaction_id" not in df_a.columns or "transaction_id" not in df_b.columns:
        raise ValueError("Ambos resultados deben tener columna 'transaction_id' para la comparación.")

    flag_a = "anomaly_flag" if "anomaly_flag" in df_a.columns else "autoencoder_anomaly_flag"
    flag_b = "anomaly_flag" if "anomaly_flag" in df_b.columns else "autoencoder_anomaly_flag"

    if flag_a not in df_a.columns or flag_b not in df_b.columns:
        raise ValueError("Ambos resultados deben tener columna de flag de anomalía.")

    anom_a = df_a[df_a[flag_a] == 1].copy()
    anom_b = df_b[df_b[flag_b] == 1].copy()

    set_a = set(anom_a["transaction_id"].astype(str))
    set_b = set(anom_b["transaction_id"].astype(str))
    intersection = set_a & set_b

    alg_a = run_a.algorithm
    alg_b = run_b.algorithm

    # Pick the best score column for each model
    score_col_a = next((c for c in ["anomaly_score", "reconstruction_error", "autoencoder_anomaly_score"] if c in anom_a.columns), None)
    score_col_b = next((c for c in ["anomaly_score", "reconstruction_error", "autoencoder_anomaly_score"] if c in anom_b.columns), None)

    context_cols = ["transaction_id", "customer_hash", "transaction_datetime", "amount", "country_code", "merchant_rubro_proxy"]

    # Build subset A: context + score + rank
    cols_a = [c for c in context_cols if c in anom_a.columns]
    if score_col_a:
        cols_a.append(score_col_a)
    if "anomaly_rank" in anom_a.columns:
        cols_a.append("anomaly_rank")

    rename_a = {}
    if score_col_a:
        rename_a[score_col_a] = f"score_{alg_a}"
    if "anomaly_rank" in anom_a.columns:
        rename_a["anomaly_rank"] = f"rank_{alg_a}"

    sub_a = anom_a[cols_a].rename(columns=rename_a)
    sub_a["transaction_id"] = sub_a["transaction_id"].astype(str)

    # Build subset B: only score + rank (context comes from A)
    cols_b = ["transaction_id"]
    if score_col_b:
        cols_b.append(score_col_b)
    if "anomaly_rank" in anom_b.columns:
        cols_b.append("anomaly_rank")

    rename_b = {}
    if score_col_b:
        rename_b[score_col_b] = f"score_{alg_b}"
    if "anomaly_rank" in anom_b.columns:
        rename_b["anomaly_rank"] = f"rank_{alg_b}"

    sub_b = anom_b[[c for c in cols_b if c in anom_b.columns]].rename(columns=rename_b)
    sub_b["transaction_id"] = sub_b["transaction_id"].astype(str)

    # Inner merge on transaction_id
    merged = pd.merge(sub_a, sub_b, on="transaction_id", how="inner")

    rank_a = f"rank_{alg_a}"
    rank_b = f"rank_{alg_b}"
    if rank_a in merged.columns and rank_b in merged.columns:
        merged["_priority"] = merged[rank_a].fillna(99999) + merged[rank_b].fillna(99999)
        merged = merged.sort_values("_priority").drop(columns=["_priority"])
    merged = merged.reset_index(drop=True)
    merged["consensus_priority"] = merged.index + 1

    rows = (
        merged
        .replace({float("nan"): None, float("inf"): None, float("-inf"): None})
        .to_dict(orient="records")
    )

    return {
        "run_a": {
            "id": run_a.id,
            "algorithm": alg_a,
            "input_source": run_a.input_source,
            "anomaly_count": run_a.anomaly_count,
        },
        "run_b": {
            "id": run_b.id,
            "algorithm": alg_b,
            "input_source": run_b.input_source,
            "anomaly_count": run_b.anomaly_count,
        },
        "total_analyzed": run_a.total_analyzed,
        "intersection_count": len(intersection),
        "only_in_a": len(set_a - set_b),
        "only_in_b": len(set_b - set_a),
        "agreement_rate_pct": round(len(intersection) / max(len(set_a), len(set_b), 1) * 100, 1),
        "rows": rows[:500],
        "methodology_warning": METHODOLOGY_WARNING,
    }

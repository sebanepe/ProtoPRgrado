from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session
import pandas as pd

from backend.app.database import get_db
from backend.app.models.models import ModelRegistry
from backend.app.ml.supervised_dataset_builder import (
    build_human_supervised_alert_dataset,
    get_latest_supervised_dataset_run,
    supervised_dataset_run_to_dict,
)
from backend.app.ml.validate_human_supervised_dataset import validate_human_supervised_dataset
from backend.app.ml.supervised_training_preflight import run_training_preflight
from backend.app.ml.train_human_supervised_model import train_human_supervised_model
from backend.app.services import artifact_registry_service as artifacts
from backend.app.services import model_registry_service
from backend.app.services import supervised_service


router = APIRouter(prefix="/api/supervised", tags=["supervised"])


class BuildHumanDatasetRequest(BaseModel):
    source_run: str
    force: bool = False


class TrainHumanModelRequest(BaseModel):
    source_run: str
    model_type: str
    test_size: float = 0.25
    random_state: int = 42
    use_smote: bool = False


FORBIDDEN_COLUMNS = {"is_fraud", "confirmed_fraud", "PAN_TARJETA", "TARJETA", "pan_card", "raw_card"}


def _normalize_model_type(model_type: str) -> str:
    value = str(model_type or "").strip()
    return "mlp" if value == "mlp_classifier" else value


def _find_supervised_model(db: Session, source_run: str, model_type: str) -> ModelRegistry:
    normalized = artifacts.normalize_source_run(source_run)
    algorithm = _normalize_model_type(model_type)
    model = (
        db.query(ModelRegistry)
        .filter(
            ModelRegistry.model_family == "SUPERVISED_HUMAN",
            ModelRegistry.algorithm == algorithm,
            ModelRegistry.source_run == normalized,
        )
        .order_by(ModelRegistry.updated_at.desc(), ModelRegistry.id.desc())
        .first()
    )
    if model is None:
        raise HTTPException(status_code=404, detail="No existe modelo supervisado entrenado para los parametros indicados.")
    return model


def _load_metrics(model: ModelRegistry) -> dict[str, Any]:
    if not model.metrics_json:
        return {}
    try:
        parsed = json.loads(model.metrics_json)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


@router.get("/human-label-summary")
def get_human_label_summary(
    source_run: Optional[str] = Query(None),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    return supervised_service.get_human_label_summary(db, source_run=source_run)


@router.get("/human-readiness")
def get_human_readiness(
    source_run: Optional[str] = Query(None),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    return supervised_service.get_human_readiness(db, source_run=source_run)


@router.post("/build-human-dataset")
def build_human_dataset(
    request: BuildHumanDatasetRequest,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    return build_human_supervised_alert_dataset(request.source_run, db=db, force=request.force)


@router.get("/human-dataset-summary")
def get_human_dataset_summary(
    source_run: str = Query(...),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    normalized = artifacts.normalize_source_run(source_run)
    run = get_latest_supervised_dataset_run(db, normalized)
    dataset_artifact = artifacts.get_artifact_by_type(db, normalized, artifacts.ARTIFACT_SUPERVISED_DATASET)
    report_artifact = artifacts.get_artifact_by_type(db, normalized, artifacts.ARTIFACT_SUPERVISED_REPORT)
    if run is None:
        return {
            "source_run": normalized,
            "exists": False,
            "verdict": "HUMAN_SUPERVISED_DATASET_NOT_FOUND",
            "status": "NOT_CREATED",
        }
    payload = supervised_dataset_run_to_dict(run)
    return {
        "source_run": normalized,
        "exists": bool(run.dataset_file),
        "dataset_file": run.dataset_file,
        "report_file": run.report_file,
        "rows": run.usable_total_count,
        "positives": run.positive_count,
        "negatives": run.negative_count,
        "technical_ready": run.technical_ready,
        "recommended_ready": run.recommended_ready,
        "strong_ready": run.strong_ready,
        "status": run.status,
        "artifact_registry_dataset_id": dataset_artifact.id if dataset_artifact else None,
        "artifact_registry_report_id": report_artifact.id if report_artifact else None,
        "supervised_dataset_run_id": run.id,
        "created_at": payload["created_at"],
        "updated_at": payload["updated_at"],
        "verdict": "HUMAN_SUPERVISED_DATASET_AVAILABLE" if run.dataset_file else "HUMAN_SUPERVISED_DATASET_NOT_CREATED",
        "metadata_json": payload["metadata_json"],
    }


@router.get("/human-dataset-report")
def get_human_dataset_report(
    source_run: str = Query(...),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    run = get_latest_supervised_dataset_run(db, source_run)
    if run is None or not run.report_file:
        raise HTTPException(status_code=404, detail="No existe reporte supervisado humano para el source_run indicado.")
    from pathlib import Path

    path = Path(run.report_file)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Reporte no encontrado en disco: {path}")
    return {"source_run": run.source_run, "report_file": path.name, "markdown": path.read_text(encoding="utf-8")}


@router.get("/human-dataset-preview")
def get_human_dataset_preview(
    source_run: str = Query(...),
    limit: int = Query(20, ge=1, le=200),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    from pathlib import Path

    import pandas as pd

    run = get_latest_supervised_dataset_run(db, source_run)
    if run is None or not run.dataset_file:
        return {"source_run": artifacts.normalize_source_run(source_run), "rows": [], "verdict": "HUMAN_SUPERVISED_DATASET_NOT_FOUND"}
    path = Path(run.dataset_file)
    if not path.exists():
        return {"source_run": run.source_run, "rows": [], "verdict": "HUMAN_SUPERVISED_DATASET_FILE_NOT_FOUND"}
    df = pd.read_csv(path, nrows=limit)
    sensitive = {"PAN_TARJETA", "TARJETA", "pan_card", "raw_card", "masked_card"}
    df = df.drop(columns=[column for column in sensitive if column in df.columns])
    df = df.astype(object).where(pd.notna(df), None)
    return {"source_run": run.source_run, "dataset_file": path.name, "limit": limit, "rows": df.to_dict(orient="records")}


@router.get("/human-dataset-validate")
def validate_human_dataset(
    source_run: str = Query(...),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    from pathlib import Path

    normalized = artifacts.normalize_source_run(source_run)
    run = get_latest_supervised_dataset_run(db, normalized)
    data_path = Path(run.dataset_file) if run and run.dataset_file else artifacts.default_processed_dir() / (
        f"supervised_human_alert_dataset_run_{artifacts.normalize_run_token(normalized)}.csv"
    )
    result = validate_human_supervised_dataset(data_path)
    result["source_run"] = normalized
    result["dataset_file"] = str(data_path)
    return result


@router.get("/training-preflight")
def get_training_preflight(
    source_run: str = Query(...),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    return run_training_preflight(source_run, db, build_if_missing=False)


@router.post("/train-human-model")
def train_human_model(
    request: TrainHumanModelRequest,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    result = train_human_supervised_model(
        request.source_run,
        _normalize_model_type(request.model_type),
        db=db,
        test_size=request.test_size,
        random_state=request.random_state,
    )
    if result.get("status") == "BLOCKED":
        raise HTTPException(status_code=409, detail=result)
    return result


@router.get("/training-runs")
def get_training_runs(
    source_run: str = Query(...),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    normalized = artifacts.normalize_source_run(source_run)
    models = model_registry_service.list_model_registry(db, source_run=normalized, model_family="SUPERVISED_HUMAN")
    items = []
    for model in models:
        payload = model_registry_service.model_registry_to_dict(model)
        payload["metrics"] = _load_metrics(model)
        items.append(payload)
    return {"source_run": normalized, "count": len(items), "items": items}


@router.get("/model-metadata")
def get_supervised_model_metadata(
    source_run: str = Query(...),
    model_type: str = Query(...),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    model = _find_supervised_model(db, source_run, model_type)
    if not model.metadata_file:
        raise HTTPException(status_code=404, detail="No existe metadata para este modelo.")
    path = Path(model.metadata_file)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Archivo de metadata no encontrado.")
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"No se pudo leer metadata del modelo: {exc}") from exc
    metadata.update(
        {
            "model_file": model.model_file,
            "metadata_file": model.metadata_file,
            "report_file": model.report_file,
            "predictions_file": model.scores_file,
            "created_at": model.created_at.isoformat() if model.created_at else None,
            "is_active": model.is_active,
            "status": model.status,
        }
    )
    return {"source_run": model.source_run, "model_type": model.algorithm, "metadata": metadata}


@router.get("/model-report")
def get_supervised_model_report(
    source_run: str = Query(...),
    model_type: str = Query(...),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    model = _find_supervised_model(db, source_run, model_type)
    if not model.report_file:
        raise HTTPException(status_code=404, detail="No existe reporte para este modelo.")
    path = Path(model.report_file)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Reporte no encontrado.")
    return {"source_run": model.source_run, "model_type": model.algorithm, "report_file": path.name, "markdown": path.read_text(encoding="utf-8")}


@router.get("/model-predictions")
def get_supervised_model_predictions(
    source_run: str = Query(...),
    model_type: str = Query(...),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
    evaluation_result: Optional[str] = Query(None),
    y_true: Optional[int] = Query(None),
    y_pred: Optional[int] = Query(None),
    prediction_label: Optional[str] = Query(None),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    model = _find_supervised_model(db, source_run, model_type)
    if not model.scores_file:
        raise HTTPException(status_code=404, detail="No existen predicciones para este modelo.")
    path = Path(model.scores_file)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Predicciones no encontradas.")
    df = pd.read_csv(path)
    df = df.drop(columns=[column for column in FORBIDDEN_COLUMNS if column in df.columns])
    if evaluation_result:
        df = df[df.get("evaluation_result", pd.Series(dtype=str)).astype(str) == evaluation_result]
    if y_true is not None:
        df = df[pd.to_numeric(df.get("y_true", pd.Series(dtype=int)), errors="coerce") == y_true]
    if y_pred is not None:
        df = df[pd.to_numeric(df.get("y_pred", pd.Series(dtype=int)), errors="coerce") == y_pred]
    if prediction_label:
        df = df[df.get("prediction_label", pd.Series(dtype=str)).astype(str) == prediction_label]
    total = len(df)
    total_pages = max(1, int((total + page_size - 1) / page_size))
    page = min(page, total_pages)
    start = (page - 1) * page_size
    rows = df.iloc[start : start + page_size].astype(object).where(pd.notna(df.iloc[start : start + page_size]), None)
    return {
        "source_run": model.source_run,
        "model_type": model.algorithm,
        "predictions_file": path.name,
        "page": page,
        "page_size": page_size,
        "total": total,
        "total_pages": total_pages,
        "rows": rows.to_dict(orient="records"),
    }

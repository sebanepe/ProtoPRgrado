from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.ml.supervised_dataset_builder import (
    build_human_supervised_alert_dataset,
    get_latest_supervised_dataset_run,
    supervised_dataset_run_to_dict,
)
from backend.app.ml.validate_human_supervised_dataset import validate_human_supervised_dataset
from backend.app.services import artifact_registry_service as artifacts
from backend.app.services import supervised_service


router = APIRouter(prefix="/api/supervised", tags=["supervised"])


class BuildHumanDatasetRequest(BaseModel):
    source_run: str


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
    return build_human_supervised_alert_dataset(request.source_run, db=db)


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

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.ml.batch_scoring_service import FORBIDDEN_OUTPUT_COLUMNS, run_batch_scoring
from backend.app.models.models import BatchScoringRun
from backend.app.services.permission_service import require_permission

router = APIRouter(prefix="/api/scoring", tags=["scoring", "batch_scoring", "d1"])


class BatchScoringRequest(BaseModel):
    source_run: str
    algorithm: str
    input_dataset_path: Optional[str] = None


def _run_to_dict(run: BatchScoringRun) -> dict[str, Any]:
    return {
        "id": run.id,
        "source_run": run.source_run,
        "run_token": run.run_token,
        "model_family": run.model_family,
        "algorithm": run.algorithm,
        "model_registry_id": run.model_registry_id,
        "total_scored": run.total_scored,
        "high_count": run.high_count,
        "medium_count": run.medium_count,
        "low_count": run.low_count,
        "status": run.status,
        "error_message": run.error_message,
        "results_file": run.results_file,
        "report_file": run.report_file,
        "metadata_file": run.metadata_file,
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "finished_at": run.finished_at.isoformat() if run.finished_at else None,
        "created_at": run.created_at.isoformat() if run.created_at else None,
    }


@router.post("/batch-run")
def start_batch_scoring(
    request: BatchScoringRequest,
    db: Session = Depends(get_db),
    _auth=Depends(require_permission("scoring")),
) -> dict[str, Any]:
    from backend.app.ml.batch_scoring_service import VALID_ALGORITHMS

    if request.algorithm not in VALID_ALGORITHMS:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "algorithm_invalid",
                "algorithm": request.algorithm,
                "valid_algorithms": sorted(VALID_ALGORITHMS),
            },
        )

    try:
        result = run_batch_scoring(
            request.source_run,
            request.algorithm,
            db=db,
            input_dataset_path=request.input_dataset_path,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail={"error": "scoring_error", "message": str(exc)})

    if result.get("status") == "BLOCKED":
        raise HTTPException(status_code=409, detail=result)

    if result.get("status") == "FAILED":
        raise HTTPException(status_code=500, detail=result)

    return result


@router.get("/runs")
def list_scoring_runs(
    source_run: Optional[str] = Query(None),
    algorithm: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    _auth=Depends(require_permission("scoring")),
) -> dict[str, Any]:
    q = db.query(BatchScoringRun)
    if source_run:
        q = q.filter(BatchScoringRun.source_run == source_run)
    if algorithm:
        q = q.filter(BatchScoringRun.algorithm == algorithm)
    if status:
        q = q.filter(BatchScoringRun.status == status)
    runs = q.order_by(BatchScoringRun.created_at.desc()).all()
    return {"count": len(runs), "items": [_run_to_dict(r) for r in runs]}


@router.get("/runs/{run_id}")
def get_scoring_run(
    run_id: int,
    db: Session = Depends(get_db),
    _auth=Depends(require_permission("scoring")),
) -> dict[str, Any]:
    run = db.query(BatchScoringRun).filter(BatchScoringRun.id == run_id).first()
    if run is None:
        raise HTTPException(status_code=404, detail=f"BatchScoringRun id={run_id} no encontrado")
    return _run_to_dict(run)


@router.get("/results")
def get_scoring_results(
    source_run: str = Query(...),
    algorithm: str = Query(...),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
    ml_risk_level: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    _auth=Depends(require_permission("scoring")),
) -> dict[str, Any]:
    run = (
        db.query(BatchScoringRun)
        .filter(
            BatchScoringRun.source_run == source_run,
            BatchScoringRun.algorithm == algorithm,
            BatchScoringRun.status == "COMPLETED",
        )
        .order_by(BatchScoringRun.created_at.desc())
        .first()
    )
    if run is None or not run.results_file:
        raise HTTPException(
            status_code=404,
            detail=f"No hay scoring completado para source_run={source_run} algorithm={algorithm}",
        )

    results_path = Path(run.results_file)
    if not results_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"results_file no existe en disco: {run.results_file}",
        )

    try:
        df = pd.read_csv(results_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error leyendo results_file: {exc}")

    # Safety guard: drop forbidden columns before serving
    forbidden_present = FORBIDDEN_OUTPUT_COLUMNS.intersection(df.columns)
    if forbidden_present:
        df = df.drop(columns=list(forbidden_present))

    if ml_risk_level:
        df = df[df["ml_risk_level"] == ml_risk_level.upper()]

    total = len(df)
    total_pages = max(1, (total + page_size - 1) // page_size)
    start = (page - 1) * page_size
    page_df = df.iloc[start : start + page_size]

    return {
        "source_run": source_run,
        "algorithm": algorithm,
        "batch_scoring_run_id": run.id,
        "page": page,
        "page_size": page_size,
        "total": total,
        "total_pages": total_pages,
        "rows": page_df.to_dict(orient="records"),
    }


@router.get("/report")
def get_scoring_report(
    source_run: str = Query(...),
    algorithm: str = Query(...),
    db: Session = Depends(get_db),
    _auth=Depends(require_permission("scoring")),
) -> dict[str, Any]:
    run = (
        db.query(BatchScoringRun)
        .filter(
            BatchScoringRun.source_run == source_run,
            BatchScoringRun.algorithm == algorithm,
            BatchScoringRun.status == "COMPLETED",
        )
        .order_by(BatchScoringRun.created_at.desc())
        .first()
    )
    if run is None or not run.report_file:
        raise HTTPException(
            status_code=404,
            detail=f"No hay reporte para source_run={source_run} algorithm={algorithm}",
        )

    report_path = Path(run.report_file)
    if not report_path.exists():
        raise HTTPException(status_code=404, detail=f"report_file no existe: {run.report_file}")

    markdown = report_path.read_text(encoding="utf-8")
    return {
        "source_run": source_run,
        "algorithm": algorithm,
        "batch_scoring_run_id": run.id,
        "report_file": run.report_file,
        "markdown": markdown,
    }

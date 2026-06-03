"""Routes for applying trained unsupervised models to new datasets (Fase C)."""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.services import unsupervised_inference_service as svc

router = APIRouter(prefix="/api/unsupervised", tags=["unsupervised-inference"])


def _http_error(exc: Exception, status: int = 400) -> HTTPException:
    return HTTPException(status_code=status, detail=str(exc))


# ── Discovery ─────────────────────────────────────────────────────────────────

@router.get("/trained-models")
def get_trained_models(db: Session = Depends(get_db)):
    """List unsupervised models with status AVAILABLE."""
    try:
        return svc.list_trained_models(db)
    except Exception as exc:
        raise _http_error(exc, 500)


@router.get("/preprocessed-runs")
def get_preprocessed_runs(db: Session = Depends(get_db)):
    """List completed preprocessing runs available as input datasets."""
    try:
        return svc.list_preprocessed_runs(db)
    except Exception as exc:
        raise _http_error(exc, 500)


# ── Apply model ───────────────────────────────────────────────────────────────

@router.post("/apply-trained-model")
def apply_trained_model(
    background_tasks: BackgroundTasks,
    model_registry_id: int = Form(...),
    input_type: str = Form(...),
    preprocessed_run_id: Optional[int] = Form(None),
    file: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db),
):
    """Start an unsupervised model inference run asynchronously.

    Returns 202 immediately with the run_id. Poll GET /inference-status/{run_id}
    until status is COMPLETED or FAILED.
    Does NOT retrain the model.
    Does NOT generate is_fraud or confirmed_fraud.
    """
    if input_type not in ("csv_upload", "preprocessed_run"):
        raise HTTPException(status_code=422, detail="input_type must be 'csv_upload' or 'preprocessed_run'")

    # Validate and prepare input FIRST (fail-fast on bad input before touching DB)
    input_file_path: str
    input_source: str

    if input_type == "csv_upload":
        if file is None:
            raise HTTPException(status_code=422, detail="Se requiere un archivo CSV cuando input_type=csv_upload")
        if not file.filename.lower().endswith(".csv"):
            raise HTTPException(status_code=422, detail="El archivo debe ser un CSV (.csv)")

        upload_dir = Path(os.environ.get("PROJECT_PROCESSED_DIR", "")) / "uploads"
        if not upload_dir.parent.exists():
            upload_dir = Path(tempfile.gettempdir()) / "unsupervised_uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)

        dest = upload_dir / file.filename
        try:
            with open(dest, "wb") as fh:
                shutil.copyfileobj(file.file, fh)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Error al guardar el archivo: {exc}")

        input_file_path = str(dest)
        input_source = file.filename

    else:  # preprocessed_run
        if preprocessed_run_id is None:
            raise HTTPException(status_code=422, detail="Se requiere preprocessed_run_id cuando input_type=preprocessed_run")

        runs = svc.list_preprocessed_runs(db)
        run_info = next((r for r in runs if r["id"] == preprocessed_run_id), None)
        if run_info is None:
            raise HTTPException(status_code=404, detail=f"preprocessing_run {preprocessed_run_id} not found or not COMPLETED")

        input_file_path = run_info["output_file_path"]
        if not input_file_path or not Path(input_file_path).exists():
            raise HTTPException(
                status_code=404,
                detail=f"El archivo del preprocessing_run {preprocessed_run_id} no existe en disco: {input_file_path}",
            )
        input_source = f"preprocessed_run_{preprocessed_run_id}"

    # Validate model exists and is AVAILABLE (after input checks to preserve 422 priority)
    trained = svc.list_trained_models(db)
    if not any(m["id"] == model_registry_id for m in trained):
        raise HTTPException(status_code=404, detail=f"Model registry entry {model_registry_id} not found or not AVAILABLE")

    try:
        run_record = svc.create_pending_run(
            db=db,
            model_registry_id=model_registry_id,
            input_file_path=input_file_path,
            input_type=input_type,
            input_source=input_source,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    background_tasks.add_task(svc.run_inference_background, run_record.id)
    return JSONResponse(
        status_code=202,
        content={"run_id": run_record.id, "status": "PENDING", "message": "Inferencia iniciada en segundo plano."},
    )


@router.get("/inference-status/{run_id}")
def get_inference_status(run_id: int, db: Session = Depends(get_db)):
    """Poll the status of an inference run. Returns status, progress counts, and error if any."""
    try:
        return svc.get_run_status(db, run_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise _http_error(exc, 500)


@router.get("/compare-runs")
def compare_runs(
    run_id_a: int = Query(...),
    run_id_b: int = Query(...),
    db: Session = Depends(get_db),
):
    """Compare two completed inference runs. Returns intersection of anomalies flagged by both models.

    Designed for fraud analysts: transactions in the intersection are stronger anomaly signals.
    """
    if run_id_a == run_id_b:
        raise HTTPException(status_code=422, detail="run_id_a y run_id_b deben ser distintos.")
    try:
        return svc.compare_inference_runs(db, run_id_a, run_id_b)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise _http_error(exc, 500)


# ── Read results ──────────────────────────────────────────────────────────────

@router.get("/prediction-runs")
def get_prediction_runs(db: Session = Depends(get_db)):
    """List all unsupervised inference runs."""
    try:
        return svc.list_prediction_runs(db)
    except Exception as exc:
        raise _http_error(exc, 500)


@router.get("/prediction-results")
def get_prediction_results(
    run_id: int = Query(...),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    anomaly_only: bool = Query(False),
    sort_by: str = Query("anomaly_rank"),
    db: Session = Depends(get_db),
):
    """Paginated results for an inference run. Strips forbidden columns."""
    try:
        return svc.get_prediction_results(
            db,
            run_id=run_id,
            page=page,
            page_size=page_size,
            anomaly_only=anomaly_only,
            sort_by=sort_by,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise _http_error(exc, 500)


@router.get("/prediction-report")
def get_prediction_report(
    run_id: int = Query(...),
    db: Session = Depends(get_db),
):
    """Summary report for an inference run including score distribution."""
    try:
        return svc.get_prediction_report(db, run_id=run_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise _http_error(exc, 500)


@router.get("/prediction-metadata")
def get_prediction_metadata(
    run_id: int = Query(...),
    db: Session = Depends(get_db),
):
    """Metadata for an inference run and the model used."""
    try:
        return svc.get_prediction_metadata(db, run_id=run_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise _http_error(exc, 500)

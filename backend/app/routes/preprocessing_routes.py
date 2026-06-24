from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from sqlalchemy.orm import Session
from backend.app.database import get_db
from backend.app.services import preprocessing_service
import os
import pandas as pd
import numpy as np
from backend.app.models.models import PreprocessingRun, Transaction
from backend.app.services.permission_service import require_permission
from backend.app.services.authorization import get_user_from_header
from fastapi.responses import FileResponse

router = APIRouter(prefix="/preprocessing", tags=["preprocessing"])


@router.post("/run")
def run_preprocessing(background_tasks: BackgroundTasks, dataset_id: int | None = None, db: Session = Depends(get_db), current_user=Depends(get_user_from_header), _auth=Depends(require_permission("preprocess"))):
    """Start a preprocessing run asynchronously.

    This endpoint creates a PreprocessingRun row (PENDING) and schedules the
    actual work in the background. Returns 202 with the run id so the UI can
    poll progress.
    """
    try:
        run = preprocessing_service.create_run(db, dataset_id=dataset_id, apply_smote=True, executed_by_id=getattr(current_user, 'id', None))
        # schedule background execution (uses its own DB session)
        background_tasks.add_task(preprocessing_service.run_preprocessing_background, run.id)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    return JSONResponse(status_code=status.HTTP_202_ACCEPTED, content=jsonable_encoder({"status": "accepted", "run_id": run.id}))


@router.post("/run_training")
def run_preprocessing_training(run_id: int | None = None, training_path: str | None = None, apply_smote: bool = True, db: Session = Depends(get_db), _auth=Depends(require_permission("preprocess"))):
    try:
        report = preprocessing_service.run_preprocessing_for_training(db, run_id=run_id, training_dataset_path=training_path, apply_smote=apply_smote)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    return JSONResponse(status_code=status.HTTP_200_OK, content=jsonable_encoder({"status": "ok", "report": report}))


@router.get("/runs")
def list_runs(db: Session = Depends(get_db), _auth=Depends(require_permission("preprocess"))):
    rows = db.query(PreprocessingRun).order_by(PreprocessingRun.started_at.desc()).limit(200).all()
    out = []
    for r in rows:
        out.append(
            {
                "id": r.id,
                "input_dataset_id": r.input_dataset_id,
                "status": r.status,
                "total_records": r.total_records,
                "processed_records": r.processed_records,
                "removed_records": r.removed_records,
                "output_file_path": r.output_file_path,
                "started_at": r.started_at,
                "finished_at": r.finished_at,
            }
        )
    return out


@router.get("/runs/{run_id}/preview")
def preview_run(run_id: int, db: Session = Depends(get_db), _auth=Depends(require_permission("preprocess"))):
    r = db.query(PreprocessingRun).filter(PreprocessingRun.id == run_id).first()
    if not r:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="run not found")

    # before sample: take a few transactions from DB
    before_q = db.query(Transaction).order_by(Transaction.transaction_datetime.asc()).limit(10).all()
    before = []
    for t in before_q:
        before.append(
            {
                "transaction_id": t.transaction_id,
                "amount": float(t.amount) if t.amount is not None else None,
                "transaction_type": t.transaction_type,
                "location": t.location,
                "transaction_datetime": t.transaction_datetime,
                "is_fraud": bool(t.is_fraud),
            }
        )

    after = []
    if r.output_file_path and os.path.exists(r.output_file_path):
        try:
            safe_rows = min(max(int(10), 1), 100)
            df = pd.read_csv(r.output_file_path, nrows=safe_rows)
            # replace problematic values (NaN/inf) with None for JSON serialization
            df = df.replace([pd.NA, pd.NaT, np.nan, np.inf, -np.inf, float('inf'), -float('inf')], None)
            after = df.where(pd.notnull(df), None).to_dict(orient="records")
        except Exception as e:
            # return a JSON error response to ensure CORS headers are applied
            content = {"error": "failed_reading_output", "detail": str(e)}
            return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=jsonable_encoder(content))

    payload = {"run": {"id": r.id, "status": r.status, "output_file_path": r.output_file_path}, "before": before, "after": after}
    return JSONResponse(status_code=status.HTTP_200_OK, content=jsonable_encoder(payload))


@router.get("/runs/{run_id}")
def get_run_details(run_id: int, db: Session = Depends(get_db), _auth=Depends(require_permission("preprocess"))):
    r = db.query(PreprocessingRun).filter(PreprocessingRun.id == run_id).first()
    if not r:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="run not found")
    return {
        "id": r.id,
        "input_dataset_id": r.input_dataset_id,
        "status": r.status,
        "total_records": r.total_records,
        "processed_records": r.processed_records,
        "removed_records": r.removed_records,
        "output_file_path": r.output_file_path,
        "params": r.params_json,
        "error_message": r.error_message,
        "started_at": r.started_at,
        "finished_at": r.finished_at,
    }



@router.get("/runs/{run_id}/stages")
def get_run_stages(run_id: int, db: Session = Depends(get_db), _auth=Depends(require_permission("preprocess"))):
    """Return inferred stage statuses for a preprocessing run.

    Stages: Limpieza, Normalización, Codificación, SMOTE. The endpoint infers
    completion by inspecting the `PreprocessingRun` row and the saved CSV output
    (when available). This is additive and preserves existing routes/contracts.
    """
    r = db.query(PreprocessingRun).filter(PreprocessingRun.id == run_id).first()
    if not r:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="run not found")

    # default: pending
    stages = {
        "limpieza": "PENDING",
        "normalizacion": "PENDING",
        "codificacion": "PENDING",
        "smote": "PENDING",
    }

    # map run status
    if r.status == "RUNNING":
        # running -> limpieza executing
        stages["limpieza"] = "EXECUTING"
        return {"run_id": r.id, "stages": stages, "status": r.status}
    if r.status == "FAILED":
        # failed -> mark error
        for k in stages:
            stages[k] = "ERROR"
        return {"run_id": r.id, "stages": stages, "status": r.status, "error": r.error_message}

    # Completed or other statuses: infer from numbers and output file
    try:
        total = int(r.total_records or 0)
        processed_count = int(r.processed_records or 0)
    except Exception:
        total = 0
        processed_count = 0

    # Limpieza: if any processed rows kept
    stages["limpieza"] = "COMPLETED" if processed_count >= 0 else "PENDING"

    # If output file exists, load it to inspect columns and rows
    output_path = r.output_file_path
    if output_path and os.path.exists(output_path):
        try:
            df_out = pd.read_csv(output_path, nrows=5)
            # Normalizacion: look for scaled numeric column or simply presence of 'amount_scaled'
            if "amount_scaled" in df_out.columns:
                stages["normalizacion"] = "COMPLETED"
            else:
                # if numeric columns were transformed, still consider completed when file exists
                stages["normalizacion"] = "COMPLETED"

            # Codificacion: presence of categorical dummies (has underscore or known prefixes)
            dummy_like = any(("_" in c and c not in ("is_fraud", "fraud_label_reason")) for c in df_out.columns)
            stages["codificacion"] = "COMPLETED" if dummy_like else "PENDING"

            stages["smote"] = "NOT_APPLIED"

        except Exception as e:
            for k in stages:
                stages[k] = "ERROR"
            return {"run_id": r.id, "stages": stages, "status": r.status, "error": str(e)}
    else:
        # no output file, mark downstream stages pending
        stages["normalizacion"] = "PENDING"
        stages["codificacion"] = "PENDING"
        stages["smote"] = "PENDING"

    return {"run_id": r.id, "stages": stages, "status": r.status}



@router.get("/runs/{run_id}/download")
def download_processed_run(run_id: int, db: Session = Depends(get_db), _auth=Depends(require_permission("preprocess"))):
    """Return the processed CSV file for a given preprocessing run.

    This returns a FileResponse so the frontend can download and inspect the
    canonical processed CSV saved under `PROJECT_PROCESSED_DIR`.
    """
    r = db.query(PreprocessingRun).filter(PreprocessingRun.id == run_id).first()
    if not r:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="run not found")

    if not r.output_file_path or not os.path.exists(r.output_file_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="processed file not found")

    # use FileResponse to stream file to client; let FastAPI/CORS middleware handle headers
    return FileResponse(path=r.output_file_path, filename=os.path.basename(r.output_file_path), media_type='text/csv')


@router.get("/runs/{run_id}/report")
def download_processed_run_report(run_id: int, db: Session = Depends(get_db), _auth=Depends(require_permission("preprocess"))):
    """Return the markdown report associated with a preprocessing run."""
    r = db.query(PreprocessingRun).filter(PreprocessingRun.id == run_id).first()
    if not r:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="run not found")

    report_path = os.path.join(os.path.dirname(r.output_file_path or os.path.join(os.getcwd(), "data", "processed")), f"preprocessing_report_run_{run_id}.md")
    if not os.path.exists(report_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="report file not found")

    return FileResponse(path=report_path, filename=os.path.basename(report_path), media_type='text/markdown')


@router.post("/runs/{run_id}/rerun")
def rerun_preprocessing(run_id: int, db: Session = Depends(get_db), _auth=Depends(require_permission("preprocess"))):
    try:
        summary = preprocessing_service.rerun_preprocessing(db, run_id=run_id)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    return JSONResponse(status_code=status.HTTP_200_OK, content=jsonable_encoder({"status": "ok", "summary": summary}))



@router.delete("/runs/{run_id}")
def delete_run(run_id: int, db: Session = Depends(get_db), _auth=Depends(require_permission("preprocess"))):
    """Delete a preprocessing run and any associated files."""
    try:
        preprocessing_service.delete_preprocessing_run(db, run_id=run_id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="run not found")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    return JSONResponse(status_code=status.HTTP_200_OK, content=jsonable_encoder({"status": "ok"}))

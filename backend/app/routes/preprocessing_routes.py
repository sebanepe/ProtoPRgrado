from fastapi import APIRouter, Depends, HTTPException, status
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

router = APIRouter(prefix="/preprocessing", tags=["preprocessing"])


@router.post("/run")
def run_preprocessing(db: Session = Depends(get_db), _auth=Depends(require_permission("preprocess"))):
    try:
        summary = preprocessing_service.run_preprocessing(db)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    return {"status": "ok", "summary": summary}


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
            df = pd.read_csv(r.output_file_path)
            # replace problematic values (NaN/inf) with None for JSON serialization
            df = df.replace([pd.NA, pd.NaT, np.nan, np.inf, -np.inf, float('inf'), -float('inf')], None)
            sample = df.head(10)
            after = sample.where(pd.notnull(sample), None).to_dict(orient="records")
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

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.orm import Session
from backend.app.database import get_db
from backend.app.services import dataset_service
from backend.app.services.permission_service import require_permission
from backend.app.services.authorization import get_user_from_header
from backend.app.repositories import dataset_repository
import pandas as pd
import os
import json
from uuid import uuid4
from datetime import datetime

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.post("/import")
def import_dataset(file: UploadFile = File(...), background_tasks: BackgroundTasks = None, db: Session = Depends(get_db), current_user=Depends(get_user_from_header), _auth=Depends(require_permission("import_data"))):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='Only CSV files are supported')

    # Persist upload quickly to disk and create dataset record; schedule background processing
    storage_dir = os.environ.get('DATASET_STORAGE', os.path.join(os.getcwd(), 'data', 'uploads'))
    try:
        os.makedirs(storage_dir, exist_ok=True)
    except Exception:
        pass
    timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
    safe_name = f"{timestamp}_{os.path.basename(file.filename)}"
    dest_path = os.path.join(storage_dir, safe_name)
    try:
        file.file.seek(0)
        with open(dest_path, 'wb') as out:
            chunk = file.file.read(1024 * 1024)
            while chunk:
                out.write(chunk)
                chunk = file.file.read(1024 * 1024)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed saving uploaded file: {e}")

    # Light upfront validation: read a small sample to detect empty files or
    # missing required columns. Return 400 for invalid uploads before creating
    # the DB dataset record or scheduling background work.
    try:
        sample = pd.read_csv(dest_path, nrows=5)
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='Uploaded CSV is empty')
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f'Failed reading uploaded CSV sample: {e}')

    # map columns and check required ones
    try:
        mapped = dataset_service._map_columns(sample)
    except Exception:
        mapped = {}

    missing = []
    for rc in dataset_service.REQUIRED_COLUMNS:
        if rc == 'is_fraud':
            # is_fraud is optional; will be filled with False if missing
            continue
        if mapped.get(rc) is None:
            missing.append(rc)
    if missing:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f'Missing required columns: {missing}')

    # try to compute total rows for small uploads so tests / clients can get a quick summary
    total_rows = None
    try:
        # for small files this is cheap; for very large files this may be slow — acceptable for test environment
        max_count_bytes = int(os.environ.get("DATASET_IMPORT_COUNT_MAX_BYTES", str(5 * 1024 * 1024)))
        if os.path.getsize(dest_path) <= max_count_bytes:
            total_rows = int(pd.read_csv(dest_path).shape[0])
    except Exception:
        total_rows = None

    # create dataset record in DB with status 'importing'
    try:
        dataset = dataset_repository.create_dataset(db, name=file.filename, file_name=safe_name, file_path=dest_path, original_filename=file.filename, total_records=0, valid_records=0, invalid_records=0, status='importing', uploaded_by_id=getattr(current_user, 'id', None))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed creating dataset record: {e}")

    # decide whether to process synchronously for very small uploads
    try:
        sync_limit = int(os.environ.get('DATASET_IMPORT_SYNC_ROWS', '1000'))
    except Exception:
        sync_limit = 1000

    job_id = str(uuid4())
    meta = {'job_id': job_id, 'submitted_at': datetime.utcnow().isoformat()}
    dataset.metadata_json = json.dumps(meta)
    db.add(dataset)
    db.commit()

    try:
        try:
            bind = db.get_bind()
        except Exception:
            bind = None

        if total_rows is not None and total_rows <= sync_limit:
            # small file: process synchronously to give immediate feedback
            dataset_service.import_dataset_background(dest_path, dataset.id, file.filename, safe_name, bind)
            ds = dataset_repository.get_dataset(db, dataset.id)
            out = {"accepted": True, "dataset_id": dataset.id, "job_id": job_id, "status": ds.status.lower() if ds and ds.status else 'imported', "message": "imported"}
            if ds:
                out['details'] = {"total": ds.total_records or total_rows, "valid": ds.valid_records or 0, "invalid": ds.invalid_records or 0}
            return out
        else:
            # schedule background processing
            background_tasks.add_task(dataset_service.import_dataset_background, dest_path, dataset.id, file.filename, safe_name, bind)
            out = {"accepted": True, "dataset_id": dataset.id, "job_id": job_id, "status": "PROCESSING", "message": "Dataset import accepted for background processing"}
            if total_rows is not None:
                out['details'] = {"total": total_rows}
            return out
    except Exception as e:
        # if background scheduling or sync processing fails, mark dataset as failed
        dataset.status = 'failed'
        dataset.error_message = f"Failed scheduling or processing import: {e}"
        db.add(dataset)
        db.commit()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed scheduling background processing: {e}")


@router.post('/import-background')
def import_dataset_background_route(file: UploadFile = File(...), background_tasks: BackgroundTasks = None, db: Session = Depends(get_db), _auth=Depends(require_permission("import_data"))):
    # New explicit background import endpoint. Behaves like /import but named for clarity.
    if not file.filename.lower().endswith(('.csv', '.xlsx', '.xlsb', '.xls')):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='Only CSV/XLSX/XLSB/XLS files are supported')

    storage_dir = os.environ.get('DATASET_STORAGE', os.path.join(os.getcwd(), 'data', 'uploads'))
    try:
        os.makedirs(storage_dir, exist_ok=True)
    except Exception:
        pass
    timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
    safe_name = f"{timestamp}_{os.path.basename(file.filename)}"
    dest_path = os.path.join(storage_dir, safe_name)
    try:
        file.file.seek(0)
        with open(dest_path, 'wb') as out:
            chunk = file.file.read(1024 * 1024)
            while chunk:
                out.write(chunk)
                chunk = file.file.read(1024 * 1024)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed saving uploaded file: {e}")

    try:
        dataset = dataset_repository.create_dataset(db, name=file.filename, file_name=safe_name, file_path=dest_path, original_filename=file.filename, total_records=0, valid_records=0, invalid_records=0, status='UPLOADED')
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed creating dataset record: {e}")

    job_id = str(uuid4())
    meta = {'job_id': job_id, 'submitted_at': datetime.utcnow().isoformat()}
    dataset.metadata_json = json.dumps(meta)
    db.add(dataset)
    db.commit()

    try:
        try:
            bind = db.get_bind()
        except Exception:
            bind = None
        background_tasks.add_task(dataset_service.import_dataset_background, dest_path, dataset.id, file.filename, safe_name, bind)
    except Exception as e:
        dataset.status = 'FAILED'
        dataset.error_message = f"Failed scheduling background processing: {e}"
        db.add(dataset)
        db.commit()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed scheduling background processing: {e}")

    return {"accepted": True, "dataset_id": dataset.id, "job_id": job_id, "status": "PROCESSING", "message": "Dataset import accepted for background processing"}


@router.get('/{dataset_id}/status')
def dataset_status(dataset_id: int, db: Session = Depends(get_db), _auth=Depends(require_permission('preprocess'))):
    ds = dataset_repository.get_dataset(db, dataset_id)
    if not ds:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Dataset not found')
    # compute progress
    total = ds.total_records or 0
    processed = (ds.valid_records or 0) + (ds.invalid_records or 0)
    inserted = ds.valid_records or 0
    progress = 0
    if total > 0:
        progress = int(processed * 100 / total)
    # attempt to extract job_id
    job_id = None
    try:
        if ds.metadata_json:
            meta = json.loads(ds.metadata_json)
            job_id = meta.get('job_id')
    except Exception:
        job_id = None

    return {
        'dataset_id': ds.id,
        'status': ds.status,
        'total_rows': total,
        'processed_rows': processed,
        'valid_rows': ds.valid_records or 0,
        'invalid_rows': ds.invalid_records or 0,
        'inserted_rows': inserted,
        'progress_percent': progress,
        'error_message': ds.error_message,
        'started_at': ds.started_at.isoformat() if getattr(ds, 'started_at', None) else None,
        'finished_at': ds.finished_at.isoformat() if getattr(ds, 'finished_at', None) else None,
        'job_id': job_id,
    }



@router.get("")
def list_datasets(limit: int = 50, offset: int = 0, db: Session = Depends(get_db), _auth=Depends(require_permission("preprocess"))):
    ds = dataset_repository.list_datasets(db, limit=limit, offset=offset)
    out = []
    for d in ds:
        out.append({
            "id": d.id,
            "name": d.name,
            "original_filename": d.original_filename,
            "file_name": d.file_name,
            "file_path": d.file_path,
            "total_records": d.total_records,
            "valid_records": d.valid_records,
            "invalid_records": d.invalid_records,
            "status": d.status,
            "created_at": d.created_at.isoformat() if getattr(d, 'created_at', None) else None,
        })
    return {"datasets": out}


@router.get("/{dataset_id}/preview")
def preview_dataset(dataset_id: int, rows: int = 10, db: Session = Depends(get_db), _auth=Depends(require_permission("preprocess"))):
    d = dataset_repository.get_dataset(db, dataset_id)
    if not d:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")
    if not d.file_path or not os.path.exists(d.file_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Raw file not found on server")
    try:
        df = pd.read_csv(d.file_path, nrows=rows)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error reading CSV: {e}")
    # sanitize values: replace inf/-inf and NaN with None for JSON serialization
    try:
        import math
        records = df.to_dict(orient="records")
        sanitized = []
        for r in records:
            nr = {}
            for k, v in r.items():
                try:
                    if isinstance(v, float) and (math.isinf(v) or math.isnan(v)):
                        nr[k] = None
                    else:
                        nr[k] = v
                except Exception:
                    nr[k] = None
            sanitized.append(nr)
    except Exception:
        sanitized = df.fillna('').astype(str).to_dict(orient='records')

    return {"dataset_id": d.id, "rows": sanitized, "columns": list(df.columns)}


@router.delete("/{dataset_id}")
def delete_dataset(dataset_id: int, db: Session = Depends(get_db), _auth=Depends(require_permission("preprocess"))):
    d = dataset_repository.get_dataset(db, dataset_id)
    if not d:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")
    try:
        ok = dataset_repository.delete_dataset(db, dataset_id)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    if not ok:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to delete dataset")
    return {"message": "deleted", "dataset_id": dataset_id}

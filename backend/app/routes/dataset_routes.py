from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status
from sqlalchemy.orm import Session
from backend.app.database import get_db
from backend.app.services import dataset_service
from backend.app.services.permission_service import require_permission
from backend.app.repositories import dataset_repository
import pandas as pd
import os

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.post("/import")
def import_dataset(file: UploadFile = File(...), db: Session = Depends(get_db), _auth=Depends(require_permission("import_data"))):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only CSV files are supported")
    try:
        result = dataset_service.import_dataset(db, file.file, name=file.filename, file_name=file.filename)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    ds = result.get('dataset')
    details = {"inserted": result["inserted"], "total": result["total"], "valid": result["valid"], "invalid": result["invalid"]}
    if ds:
        details.update({"dataset_id": ds.id, "file_path": ds.file_path, "original_filename": ds.original_filename})
    return {"message": "imported", "details": details}



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
    return {"dataset_id": d.id, "rows": df.to_dict(orient="records"), "columns": list(df.columns)}


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

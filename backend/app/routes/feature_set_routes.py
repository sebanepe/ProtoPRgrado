from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.encoders import jsonable_encoder
from sqlalchemy.orm import Session
from backend.app.database import get_db
from backend.app.services.permission_service import require_permission
from backend.app.services.authorization import get_user_from_header
from backend.app.models.models import FeatureSet, SystemLog
import os
import pandas as pd
import numpy as np

router = APIRouter(prefix="/feature_sets", tags=["feature_sets"])


@router.get("/{fs_id}/preview")
def preview_feature_set(fs_id: int, rows: int = 10, db: Session = Depends(get_db), _auth=Depends(require_permission("preprocess"))):
    fs = db.query(FeatureSet).filter(FeatureSet.id == fs_id).first()
    if not fs:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="feature_set not found")

    if not fs.file_path or not os.path.exists(fs.file_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="feature set file not found")

    try:
        df = pd.read_csv(fs.file_path)
        df = df.replace([pd.NA, pd.NaT, np.nan, np.inf, -np.inf, float('inf'), -float('inf')], None)
        sample = df.head(rows)
        rows_out = sample.where(pd.notnull(sample), None).to_dict(orient="records")
        payload = {"id": fs.id, "name": fs.name, "rows": rows_out, "columns": list(df.columns)}
        return JSONResponse(status_code=status.HTTP_200_OK, content=jsonable_encoder(payload))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/{fs_id}/download")
def download_feature_set(fs_id: int, db: Session = Depends(get_db), _auth=Depends(require_permission("preprocess"))):
    fs = db.query(FeatureSet).filter(FeatureSet.id == fs_id).first()
    if not fs:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="feature_set not found")
    if not fs.file_path or not os.path.exists(fs.file_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="feature set file not found")
    return FileResponse(path=fs.file_path, filename=os.path.basename(fs.file_path), media_type='text/csv')


@router.delete("/{fs_id}")
def delete_feature_set(fs_id: int, db: Session = Depends(get_db), user=Depends(get_user_from_header), request: Request = None, _auth=Depends(require_permission("preprocess"))):
    fs = db.query(FeatureSet).filter(FeatureSet.id == fs_id).first()
    if not fs:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="feature_set not found")

    # try removing the file and pipeline artifact
    try:
        if fs.file_path and os.path.exists(fs.file_path):
            os.remove(fs.file_path)
    except Exception:
        pass
    try:
        if fs.pipeline_path and os.path.exists(fs.pipeline_path):
            os.remove(fs.pipeline_path)
    except Exception:
        pass

    try:
        # create audit log
        try:
            ip = None
            ua = None
            try:
                if request and hasattr(request, 'client') and request.client:
                    ip = request.client.host
            except Exception:
                ip = None
            try:
                ua = request.headers.get('user-agent') if request else None
            except Exception:
                ua = None
            log = SystemLog(action="delete_feature_set", description=f"Deleted feature_set {fs.id}", user_id=getattr(user, 'id', None), ip=ip, user_agent=ua)
            db.add(log)
            db.commit()
        except Exception:
            db.rollback()
        db.delete(fs)
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    return JSONResponse(status_code=status.HTTP_200_OK, content=jsonable_encoder({"status": "ok"}))

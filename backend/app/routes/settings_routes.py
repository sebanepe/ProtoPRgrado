from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from backend.app.database import get_db
from backend.app.services import settings_service

router = APIRouter(prefix="/settings", tags=["settings"])


@router.get("/model-config")
def get_model_config(db: Session = Depends(get_db)):
    cfg = settings_service.get_active_config(db)
    if not cfg:
        return {"model_config": None}
    return {"model_config": {"id": cfg.id, "active_model_id": cfg.active_model_id, "alert_threshold": cfg.alert_threshold, "updated_by": cfg.updated_by, "updated_at": cfg.updated_at}}


@router.post("/model-config")
def post_model_config(payload: dict, db: Session = Depends(get_db)):
    # payload may contain: active_model_id, alert_threshold, updated_by
    active_model_id = payload.get("active_model_id")
    alert_threshold = payload.get("alert_threshold")
    updated_by = payload.get("updated_by")
    try:
        cfg = settings_service.set_model_config(db, active_model_id=active_model_id, alert_threshold=alert_threshold, updated_by=updated_by)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    return {"model_config": {"id": cfg.id, "active_model_id": cfg.active_model_id, "alert_threshold": cfg.alert_threshold, "updated_by": cfg.updated_by, "updated_at": cfg.updated_at}}

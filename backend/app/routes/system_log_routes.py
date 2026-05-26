from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from backend.app.database import get_db
from backend.app.models.models import SystemLog
from backend.app.services.permission_service import require_permission

router = APIRouter(prefix="/system_logs", tags=["system_logs"])


@router.get("")
def list_system_logs(page: int = 1, per_page: int = 20, action: str | None = None, db: Session = Depends(get_db), _auth=Depends(require_permission("preprocess"))):
    if page < 1:
        page = 1
    if per_page < 1 or per_page > 200:
        per_page = 20

    q = db.query(SystemLog)
    if action:
        q = q.filter(SystemLog.action == action)

    total = q.count()
    items = q.order_by(SystemLog.created_at.desc()).offset((page - 1) * per_page).limit(per_page).all()

    out = []
    for it in items:
        out.append({
            "id": it.id,
            "action": it.action,
            "description": it.description,
            "user_id": it.user_id,
            "ip": it.ip,
            "user_agent": it.user_agent,
            "created_at": it.created_at,
        })

    return JSONResponse(status_code=status.HTTP_200_OK, content=jsonable_encoder({"total": total, "page": page, "per_page": per_page, "items": out}))

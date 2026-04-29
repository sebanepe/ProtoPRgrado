from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from backend.app.database import get_db
from backend.app.services import alert_service
from typing import List, Optional

router = APIRouter(prefix="/alerts", tags=["alerts"])


@router.post("/generate")
def generate_alerts(transactions: List[dict], db: Session = Depends(get_db)):
    try:
        created = alert_service.generate_alerts_from_batch(db, transactions)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    return {"created": created}


@router.get("")
def list_alerts(
    status: Optional[str] = Query(None),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    risk_level: Optional[str] = Query(None),
    channel: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    filters = {"status": status, "start_date": start_date, "end_date": end_date, "risk_level": risk_level, "channel": channel}
    res = alert_service.list_alerts(db, filters)
    return {"alerts": [ {"id": a.id, "transaction_id": a.transaction_id, "risk_score": a.risk_score, "risk_level": a.risk_level, "status": a.status, "created_at": a.created_at} for a in res ]}


@router.get("/{id}")
def get_alert(id: int, db: Session = Depends(get_db)):
    a = alert_service.get_alert(db, id)
    if not a:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    return {"id": a.id, "transaction_id": a.transaction_id, "risk_score": a.risk_score, "risk_level": a.risk_level, "status": a.status, "created_at": a.created_at}


@router.patch("/{id}/status")
def patch_status(id: int, status: str, db: Session = Depends(get_db)):
    try:
        a = alert_service.patch_alert_status(db, id, status)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    return {"id": a.id, "status": a.status}

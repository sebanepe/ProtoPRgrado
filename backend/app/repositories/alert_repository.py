from sqlalchemy.orm import Session
from backend.app.models.models import FraudAlert
from datetime import datetime
from typing import List, Dict, Any


def create_alert(db: Session, *, transaction_id: int, risk_score: float, risk_level: str, model_name: str, status: str = "NEW") -> FraudAlert:
    alert = FraudAlert(
        transaction_id=transaction_id,
        risk_score=risk_score,
        risk_level=risk_level,
        model_name=model_name,
        status=status,
    )
    db.add(alert)
    db.commit()
    db.refresh(alert)
    return alert


def get_alert(db: Session, alert_id: int) -> FraudAlert | None:
    return db.query(FraudAlert).filter(FraudAlert.id == alert_id).first()


def query_alerts(db: Session, filters: Dict[str, Any]) -> List[FraudAlert]:
    q = db.query(FraudAlert)
    if "status" in filters and filters["status"]:
        q = q.filter(FraudAlert.status == filters["status"])
    if "risk_level" in filters and filters["risk_level"]:
        q = q.filter(FraudAlert.risk_level == filters["risk_level"])
    if "channel" in filters and filters["channel"]:
        # channel is on Transaction; join
        q = q.join(FraudAlert.transaction).filter_by(channel=filters["channel"])
    if "start_date" in filters and filters["start_date"]:
        q = q.filter(FraudAlert.created_at >= filters["start_date"])
    if "end_date" in filters and filters["end_date"]:
        q = q.filter(FraudAlert.created_at <= filters["end_date"])
    return q.order_by(FraudAlert.created_at.desc()).all()


def update_alert_status(db: Session, alert_id: int, status: str) -> FraudAlert | None:
    alert = get_alert(db, alert_id)
    if not alert:
        return None
    alert.status = status
    db.add(alert)
    db.commit()
    db.refresh(alert)
    return alert

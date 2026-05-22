from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func
from backend.app.database import get_db
from backend.app.models.models import Transaction, FraudAlert, ModelResult

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get('/summary')
def dashboard_summary(db: Session = Depends(get_db)):
    # Transactions analyzed
    tx_count = db.query(func.count(Transaction.id)).scalar() or 0

    # Alerts (total) and average risk
    alerts_q = db.query(FraudAlert)
    alerts_count = db.query(func.count(FraudAlert.id)).scalar() or 0
    avg_risk = db.query(func.avg(FraudAlert.risk_score)).scalar() or 0.0

    # Active model
    active = db.query(ModelResult).filter(ModelResult.is_active == True).order_by(ModelResult.created_at.desc()).first()
    active_model = active.model_name if active else None

    # Alert trend (last 7 days)
    trend_rows = (
        db.query(func.date(FraudAlert.created_at).label('d'), func.count(FraudAlert.id).label('c'))
        .group_by(func.date(FraudAlert.created_at))
        .order_by(func.date(FraudAlert.created_at))
        .limit(14)
        .all()
    )
    alert_trend = [{"date": str(r.d), "count": int(r.c)} for r in trend_rows]

    # Fraud ratio from transactions
    fraud_count = db.query(func.count(Transaction.id)).filter(Transaction.is_fraud == True).scalar() or 0
    total_tx = tx_count or 1
    fraud_pct = int((fraud_count / total_tx) * 100) if total_tx else 0
    normal_pct = 100 - fraud_pct

    # Recent alerts
    recent = (
        db.query(FraudAlert).order_by(FraudAlert.created_at.desc()).limit(10).all()
    )
    recent_list = [
        {
            "alert_id": a.id,
            "transaction_id": a.transaction_id,
            "score": a.risk_score,
            "channel": a.transaction.channel if a.transaction else None,
            "amount": float(a.transaction.amount) if a.transaction and a.transaction.amount is not None else None,
            "status": a.status,
            "date": a.created_at.isoformat() if a.created_at else None,
        }
        for a in recent
    ]

    return {
        "transactions": int(tx_count),
        "alerts": int(alerts_count),
        "risk": float(avg_risk) if avg_risk is not None else 0.0,
        "model": active_model or "--",
        "alertTrend": alert_trend,
        "fraudRatio": {"fraud": fraud_pct, "normal": normal_pct},
        "recentAlerts": recent_list,
        # Backwards-compat keys expected by frontend mock
        "recent_alerts": recent_list,
        # Backwards-compatible keys expected by integration tests
        "total_transactions": int(tx_count),
        "active_alerts": int(alerts_count),
        "average_risk": float(avg_risk) if avg_risk is not None else 0.0,
        "active_model": active_model or "--",
    }

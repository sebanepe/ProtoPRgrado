import os
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from backend.app.repositories import alert_repository
from backend.app.models.models import ModelResult, Transaction
from backend.app.ml.scoring import load_model_by_info, risk_score_from_model, classify_risk
import os

DEFAULT_MODELS_DIR = os.path.join("backend", "app", "ml", "saved_models")
ALERT_THRESHOLD = float(os.getenv("ALERT_THRESHOLD", "0.7"))


def _get_active_model_row(db: Session) -> ModelResult | None:
    return db.query(ModelResult).filter(ModelResult.is_active == True).order_by(ModelResult.created_at.desc()).first()


def generate_alerts_from_batch(db: Session, transactions: List[Dict[str, Any]], models_dir: str | None = None) -> List[Dict]:
    # fetch active model
    mr = _get_active_model_row(db)
    if not mr:
        raise ValueError("No active model available")

    model = load_model_by_info(mr.model_name, mr.version, models_dir=models_dir)

    # prepare feature dicts for scoring: expect transactions contain feature columns used in model
    features = [ {k: v for k, v in t.items() if k not in ("transaction_id", "transaction_datetime", "device_id", "customer_hash")} for t in transactions ]

    scores = risk_score_from_model(model, mr.model_name, features)
    created = []
    for tx, score in zip(transactions, scores):
        level = classify_risk(float(score))
        tx_id = tx.get("id") or tx.get("transaction_id")
        # only create alert if score >= ALERT_THRESHOLD
        if float(score) >= ALERT_THRESHOLD:
            # If transaction id refers to actual Transaction PK, use it; otherwise assume absent and set to None
            try:
                trans_pk = int(tx_id) if tx_id is not None and str(tx_id).isdigit() else None
            except Exception:
                trans_pk = None
            alert = alert_repository.create_alert(db, transaction_id=trans_pk or 0, risk_score=float(score), risk_level=level, model_name=mr.model_name, status="NEW")
            created.append({"alert_id": alert.id, "transaction_id": alert.transaction_id, "risk_score": alert.risk_score, "risk_level": alert.risk_level})

    return created


def list_alerts(db: Session, filters: Dict[str, Any]):
    return alert_repository.query_alerts(db, filters)


def get_alert(db: Session, alert_id: int):
    return alert_repository.get_alert(db, alert_id)


def patch_alert_status(db: Session, alert_id: int, status: str):
    allowed = {"NEW", "REVIEWED", "DISMISSED", "CONFIRMED"}
    if status not in allowed:
        raise ValueError("Invalid status")
    res = alert_repository.update_alert_status(db, alert_id, status)
    if not res:
        raise ValueError("Alert not found")
    return res

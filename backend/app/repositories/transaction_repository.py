from sqlalchemy.orm import Session
from backend.app.models.models import Transaction
from typing import List, Dict
from sqlalchemy.exc import SQLAlchemyError


def insert_transactions(db: Session, records: List[Dict], dataset_id: int | None = None):
    inserted = 0
    for r in records:
        t = Transaction(
            transaction_id=r.get("transaction_id"),
            amount=r.get("amount"),
            transaction_type=r.get("transaction_type"),
            channel=r.get("channel"),
            location=r.get("location"),
            device_id=r.get("device_id"),
            customer_hash=r.get("customer_hash"),
            transaction_datetime=r.get("transaction_datetime"),
            is_fraud=r.get("is_fraud", False),
            dataset_id=dataset_id,
        )
        try:
            db.add(t)
            db.commit()
            inserted += 1
        except SQLAlchemyError:
            db.rollback()
            # skip problematic record (likely duplicate key) and continue
            continue
    return inserted

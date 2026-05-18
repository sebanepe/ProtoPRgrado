from backend.app.services import preprocessing_service
from backend.app.models.models import Transaction
from datetime import datetime


def test_preprocessing_pipeline_with_sample_transactions(db_session, sample_transactions):
    # insert sample transactions into DB
    for t in sample_transactions:
        tx = Transaction(
            transaction_id=t["transaction_id"],
            amount=float(t["amount"]),
            transaction_type=t.get("transaction_type"),
            channel=t.get("channel"),
            location=t.get("location"),
            device_id=t.get("device_id"),
            customer_hash=t.get("customer_hash"),
            transaction_datetime=datetime.fromisoformat(t["transaction_datetime"]),
            is_fraud=bool(t["is_fraud"]),
        )
        db_session.add(tx)
    db_session.commit()

    summary = preprocessing_service.run_preprocessing(db_session, output_path="", apply_smote=False)
    # summary should contain counts
    assert "after_clean" in summary
    assert summary["after_clean"] >= 0


def test_preprocessing_endpoint_if_exists(test_client):
    # endpoint exists and responds
    r = test_client.post("/preprocessing/run")
    assert r.status_code == 200
    data = r.json()
    assert data.get("status") == "ok"

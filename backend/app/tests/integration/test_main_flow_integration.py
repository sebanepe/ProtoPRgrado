from backend.app.models.models import Transaction, ModelResult, ModelConfig
from backend.app.services import preprocessing_service


def test_minimal_fraud_detection_flow(db_session, test_client, monkeypatch):
    # insert transactions
    t = Transaction(
        transaction_id="flow1",
        amount=50.0,
        transaction_type="purchase",
        channel="web",
        location="here",
        device_id="d1",
        customer_hash="c1",
        transaction_datetime=None,
        is_fraud=False,
    )
    # give a valid datetime to avoid being dropped
    import datetime

    t.transaction_datetime = datetime.datetime.utcnow()
    db_session.add(t)
    db_session.commit()

    # create dummy model result and active config
    mr = ModelResult(model_name="mf", version="v1", is_active=True)
    db_session.add(mr)
    db_session.commit()
    cfg = ModelConfig(active_model_id=mr.id, alert_threshold=0.1, updated_by="test")
    db_session.add(cfg)
    db_session.commit()

    # monkeypatch model loader to return model that predicts high score
    class Dummy:
        def predict_proba(self, arr):
            import numpy as np

            return np.vstack([arr.sum(axis=1) * 0 + 0.1, arr.sum(axis=1) * 0 + 0.95]).T

    from backend.app.services import alert_service

    monkeypatch.setattr(alert_service, "load_model_by_info", lambda *a, **k: Dummy())

    # run preprocessing service (lightweight)
    summary = preprocessing_service.run_preprocessing(db_session, output_path="", apply_smote=False)
    assert "after_clean" in summary

    # generate alerts from batch using the alert endpoint
    txs = [{"transaction_id": t.transaction_id, "amount": float(t.amount)}]
    r = test_client.post("/alerts/generate", json=txs)
    assert r.status_code in (200, 201)

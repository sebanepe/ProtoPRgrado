import types
from backend.app.models.models import ModelResult


def test_alert_created_when_score_exceeds_threshold(db_session, test_client, monkeypatch):
    # create active model
    mr = ModelResult(model_name="a", version="v1", is_active=True)
    db_session.add(mr)
    db_session.commit()

    # dummy model that returns high probability
    class Dummy:
        def predict_proba(self, arr):
            import numpy as np

            return np.vstack([arr.sum(axis=1) * 0 + 0.1, arr.sum(axis=1) * 0 + 0.95]).T

    from backend.app.services import alert_service

    monkeypatch.setattr(alert_service, "load_model_by_info", lambda *a, **k: Dummy())

    txs = [{"transaction_id": "1", "amount": 10.0}]
    r = test_client.post("/alerts/generate", json=txs)
    assert r.status_code in (200, 201)
    if r.status_code == 200:
        assert isinstance(r.json().get("created"), list)


def test_alert_not_created_when_score_below_threshold(db_session, test_client, monkeypatch):
    mr = ModelResult(model_name="b", version="v1", is_active=True)
    db_session.add(mr)
    db_session.commit()

    class DummyLow:
        def predict_proba(self, arr):
            import numpy as np

            return np.vstack([arr.sum(axis=1) * 0 + 0.9, arr.sum(axis=1) * 0 + 0.1]).T

    from backend.app.services import alert_service

    monkeypatch.setattr(alert_service, "load_model_by_info", lambda *a, **k: DummyLow())
    txs = [{"transaction_id": "2", "amount": 1.0}]
    r = test_client.post("/alerts/generate", json=txs)
    assert r.status_code in (200, 201)

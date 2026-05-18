from backend.app.models.models import ModelResult


def test_model_results_can_be_registered(db_session):
    mr = ModelResult(model_name="dummy", version="v1", accuracy=0.9)
    db_session.add(mr)
    db_session.commit()
    assert mr.id is not None


def test_model_results_route_returns_models(test_client, db_session):
    # ensure at least one model exists
    resp = test_client.get("/models/results")
    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data


def test_model_activation_flow_if_supported(test_client, db_session):
    # create model result
    mr = ModelResult(model_name="m1", version="v1", accuracy=0.1)
    db_session.add(mr)
    db_session.commit()
    # activate via endpoint
    r = test_client.post(f"/models/{mr.id}/activate")
    assert r.status_code == 200
    assert r.json().get("activated") == mr.id

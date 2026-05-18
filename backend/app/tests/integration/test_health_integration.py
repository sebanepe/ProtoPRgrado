def test_health_endpoint_returns_ok(test_client):
    r = test_client.get("/health")
    assert r.status_code == 200
    assert r.json().get("status") == "ok"


def test_openapi_schema_available(test_client):
    r = test_client.get("/openapi.json")
    assert r.status_code == 200
    assert "openapi" in r.json()

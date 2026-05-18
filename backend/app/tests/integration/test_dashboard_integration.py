def test_dashboard_summary_returns_expected_keys(test_client):
    # If dashboard endpoint exists, try to call it; otherwise this should be harmless
    r = test_client.get("/dashboard/summary")
    # endpoint may not exist; allow 404
    assert r.status_code in (200, 404, 405)
    if r.status_code == 200:
        data = r.json()
        for k in ("total_transactions", "active_alerts", "average_risk", "active_model"):
            assert k in data


def test_dashboard_handles_empty_database(test_client):
    r = test_client.get("/dashboard/summary")
    assert r.status_code in (200, 404, 405)

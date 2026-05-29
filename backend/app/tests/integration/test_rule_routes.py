from __future__ import annotations

from pathlib import Path

import pandas as pd

from backend.app.routes import rule_routes


def _write_sample_rule_files(base_dir: Path) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)

    preprocessed = pd.DataFrame(
        [
            {"transaction_id": "tx-1", "customer_hash": "cust-1", "transaction_datetime": "2026-05-28T10:00:00+00:00", "amount": 10.0},
            {"transaction_id": "tx-2", "customer_hash": "cust-1", "transaction_datetime": "2026-05-28T10:05:00+00:00", "amount": 20.0},
        ]
    )
    alerts = pd.DataFrame(
        [
            {
                "alert_id": "26-000001",
                "source_run": "26",
                "transaction_id": "tx-1",
                "customer_hash": "cust-1",
                "transaction_datetime": "2026-05-28T10:00:00+00:00",
                "amount": 10.0,
                "country_code": "BO",
                "pos_entry_mode": "7",
                "merchant_rubro_proxy": "5944",
                "rule_code": "RULE_DOUBLE_COUNTRY_SAME_DAY",
                "rule_name": "Double country same day",
                "risk_level": "HIGH",
                "risk_score": 0.95,
                "status": "NEW",
            },
            {
                "alert_id": "26-000002",
                "source_run": "26",
                "transaction_id": "tx-2",
                "customer_hash": "cust-1",
                "transaction_datetime": "2026-05-28T10:05:00+00:00",
                "amount": 20.0,
                "country_code": "AR",
                "pos_entry_mode": "7",
                "merchant_rubro_proxy": "5944",
                "rule_code": "RULE_DOUBLE_COUNTRY_SAME_DAY",
                "rule_name": "Double country same day",
                "risk_level": "HIGH",
                "risk_score": 0.96,
                "status": "NEW",
            },
        ]
    )
    summary = pd.DataFrame(
        [
            {
                "summary_alert_id": "26-S-000001",
                "source_run": "26",
                "customer_hash": "cust-1",
                "rule_code": "RULE_DOUBLE_COUNTRY_SAME_DAY",
                "rule_name": "Double country same day",
                "risk_level": "HIGH",
                "max_risk_score": 0.96,
                "count_transactions": 2,
                "countries_detected": "AR|BO",
                "window_start": "2026-05-28T10:00:00+00:00",
                "window_end": "2026-05-28T10:05:00+00:00",
                "representative_transaction_id": "tx-1",
                "status": "NEW",
                "created_at": "2026-05-28T10:10:00+00:00",
            }
        ]
    )

    preprocessed.to_csv(base_dir / "preprocessed_run_26.csv", index=False)
    alerts.to_csv(base_dir / "alerts_run_26.csv", index=False)
    summary.to_csv(base_dir / "alerts_summary_run_26.csv", index=False)
    (base_dir / "rules_report_run_26.md").write_text("# Rules Report\n\n- sample report\n", encoding="utf-8")


def test_rule_routes_endpoints(monkeypatch, test_client, tmp_path):
    processed_dir = tmp_path / "processed"
    _write_sample_rule_files(processed_dir)
    monkeypatch.setattr(rule_routes, "_processed_dir", lambda: processed_dir)

    runs_response = test_client.get("/api/rules/preprocessed-runs")
    assert runs_response.status_code == 200
    runs = runs_response.json()
    assert runs and runs[0]["run_id"] == "preprocessed_run_26"
    assert "is_fraud" not in runs_response.text
    assert "confirmed_fraud" not in runs_response.text

    analyze_response = test_client.post("/api/rules/analyze", json={"preprocessed_run_id": "preprocessed_run_26", "force": False, "config": {}})
    assert analyze_response.status_code == 200
    assert analyze_response.json()["status"] == "ALREADY_EXISTS"

    summary_response = test_client.get("/api/rules/summary", params={"run_id": "preprocessed_run_26", "page": 1, "page_size": 1})
    assert summary_response.status_code == 200
    summary_json = summary_response.json()
    assert summary_json["total_items"] == 1
    assert len(summary_json["items"]) == 1
    assert summary_json["page_size"] == 1
    assert "is_fraud" not in summary_response.text
    assert "confirmed_fraud" not in summary_response.text

    alerts_response = test_client.get("/api/rules/alerts", params={"run_id": "preprocessed_run_26", "page": 1, "page_size": 1})
    assert alerts_response.status_code == 200
    alerts_json = alerts_response.json()
    assert alerts_json["total_items"] == 2
    assert len(alerts_json["items"]) == 1
    assert alerts_json["page_size"] == 1

    report_response = test_client.get("/api/rules/report", params={"run_id": "preprocessed_run_26"})
    assert report_response.status_code == 200
    assert "sample report" in report_response.json()["report"]

    metrics_response = test_client.get("/api/rules/metrics", params={"run_id": "preprocessed_run_26"})
    assert metrics_response.status_code == 200
    metrics_json = metrics_response.json()
    assert metrics_json["total_alerts"] == 2
    assert metrics_json["total_summary_alerts"] == 1
    assert metrics_json["alerts_by_rule"]["RULE_DOUBLE_COUNTRY_SAME_DAY"] == 2
    assert "is_fraud" not in metrics_response.text
    assert "confirmed_fraud" not in metrics_response.text

    detail_response = test_client.get("/api/rules/alerts/26-000001", params={"run_id": "preprocessed_run_26"})
    assert detail_response.status_code == 200
    assert detail_response.json()["alert_id"] == "26-000001"

    missing_response = test_client.get("/api/rules/summary", params={"run_id": "preprocessed_run_999", "page": 1, "page_size": 1})
    assert missing_response.status_code == 404

    limited_page_size = test_client.get("/api/rules/alerts", params={"run_id": "preprocessed_run_26", "page_size": 999})
    assert limited_page_size.status_code == 200
    assert limited_page_size.json()["page_size"] == 200
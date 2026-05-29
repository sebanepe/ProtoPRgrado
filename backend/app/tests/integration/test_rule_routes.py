from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

from backend.app.models.models import RuleAlertReview
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
                "rule_code": "RULE_DOUBLE_COUNTRY_CARD_PRESENT_SAME_DAY",
                "rule_name": "Double country card present same day",
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
                "country_code": "US",
                "pos_entry_mode": "7",
                "merchant_rubro_proxy": "6011",
                "rule_code": "RULE_DOUBLE_COUNTRY_CARD_PRESENT_SAME_DAY",
                "rule_name": "Double country card present same day",
                "risk_level": "HIGH",
                "risk_score": 0.96,
                "status": "NEW",
            },
            {
                "alert_id": "26-000003",
                "source_run": "26",
                "transaction_id": "tx-3",
                "customer_hash": "cust-2",
                "transaction_datetime": "2026-05-28T11:00:00+00:00",
                "amount": 50.0,
                "country_code": "BO",
                "pos_entry_mode": "7",
                "merchant_rubro_proxy": "7995",
                "rule_code": "RULE_GAMBLING_MCC",
                "rule_name": "Gambling MCC",
                "risk_level": "HIGH",
                "risk_score": 0.88,
                "status": "NEW",
            },
            {
                "alert_id": "26-000004",
                "source_run": "26",
                "transaction_id": "tx-4",
                "customer_hash": "cust-3",
                "transaction_datetime": "2026-05-28T12:00:00+00:00",
                "amount": 40.0,
                "country_code": "AR",
                "pos_entry_mode": "7",
                "merchant_rubro_proxy": "5411",
                "rule_code": "RULE_VELOCITY_CARD_HOUR",
                "rule_name": "Velocity card hour",
                "risk_level": "MEDIUM",
                "risk_score": 0.73,
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
                "rule_code": "RULE_DOUBLE_COUNTRY_CARD_PRESENT_SAME_DAY",
                "rule_name": "Double country card present same day",
                "risk_level": "HIGH",
                "max_risk_score": 0.96,
                "count_transactions": 2,
                "countries_detected": "[\"BO\",\"US\"]",
                "window_start": "2026-05-28T10:00:00+00:00",
                "window_end": "2026-05-28T10:05:00+00:00",
                "representative_transaction_id": "tx-1",
                "status": "NEW",
                "created_at": "2026-05-28T10:10:00+00:00",
            },
            {
                "summary_alert_id": "26-S-000002",
                "source_run": "26",
                "customer_hash": "cust-2",
                "rule_code": "RULE_GAMBLING_MCC",
                "rule_name": "Gambling MCC",
                "risk_level": "HIGH",
                "max_risk_score": 0.88,
                "count_transactions": 1,
                "countries_detected": "[\"BO\"]",
                "window_start": "2026-05-28T11:00:00+00:00",
                "window_end": "2026-05-28T11:00:00+00:00",
                "representative_transaction_id": "tx-3",
                "status": "NEW",
                "created_at": "2026-05-28T11:05:00+00:00",
            },
            {
                "summary_alert_id": "26-S-000003",
                "source_run": "26",
                "customer_hash": "cust-3",
                "rule_code": "RULE_VELOCITY_CARD_HOUR",
                "rule_name": "Velocity card hour",
                "risk_level": "MEDIUM",
                "max_risk_score": 0.73,
                "count_transactions": 1,
                "countries_detected": "[\"AR\"]",
                "window_start": "2026-05-28T12:00:00+00:00",
                "window_end": "2026-05-28T12:00:00+00:00",
                "representative_transaction_id": "tx-4",
                "status": "NEW",
                "created_at": "2026-05-28T12:05:00+00:00",
            },
        ]
    )

    preprocessed.to_csv(base_dir / "preprocessed_run_26.csv", index=False)
    alerts.to_csv(base_dir / "alerts_run_26.csv", index=False)
    summary.to_csv(base_dir / "alerts_summary_run_26.csv", index=False)
    (base_dir / "rules_report_run_26.md").write_text("# Rules Report\n\n- sample report\n", encoding="utf-8")


def _write_large_summary_rule_files(base_dir: Path) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for index in range(1, 46):
        summary_rows.append(
            {
                "summary_alert_id": f"26-S-{index:06d}",
                "source_run": "26",
                "customer_hash": f"cust-{(index % 3) + 1}",
                "rule_code": "RULE_DOUBLE_COUNTRY_CARD_PRESENT_SAME_DAY",
                "rule_name": "Double country card present same day",
                "risk_level": "HIGH" if index % 2 else "MEDIUM",
                "max_risk_score": 85.0 if index % 2 else 65.0,
                "count_transactions": index,
                "countries_detected": "BO|AR" if index % 2 else "BO",
                "window_start": f"2026-05-28T10:{index:02d}:00+00:00",
                "window_end": f"2026-05-28T10:{index + 1:02d}:00+00:00",
                "representative_transaction_id": f"tx-{index}",
                "status": "NEW",
                "created_at": "2026-05-28T10:10:00+00:00",
            }
        )

    summary_rows[41]["window_start"] = float("nan")
    summary_rows[41]["window_end"] = float("nan")

    pd.DataFrame(summary_rows).to_csv(base_dir / "alerts_summary_run_26.csv", index=False)
    pd.DataFrame([
        {
            "alert_id": f"26-{index:06d}",
            "source_run": "26",
            "transaction_id": f"tx-{index}",
            "customer_hash": f"cust-{(index % 3) + 1}",
            "transaction_datetime": "2026-05-28T10:00:00+00:00",
            "amount": 10.0,
            "country_code": "BO",
            "pos_entry_mode": "7",
            "merchant_rubro_proxy": "5944" if index % 2 else "6011",
            "rule_code": "RULE_DOUBLE_COUNTRY_CARD_PRESENT_SAME_DAY",
            "rule_name": "Double country card present same day",
            "risk_level": "HIGH",
            "risk_score": 0.95,
            "status": "NEW",
        }
        for index in range(1, 46)
    ]).to_csv(base_dir / "alerts_run_26.csv", index=False)
    pd.DataFrame([{ "transaction_id": f"tx-{index}", "customer_hash": f"cust-{(index % 3) + 1}", "transaction_datetime": "2026-05-28T10:00:00+00:00", "amount": 10.0 } for index in range(1, 46)]).to_csv(base_dir / "preprocessed_run_26.csv", index=False)
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
    assert summary_json["total_items"] == 3
    assert len(summary_json["items"]) == 1
    assert summary_json["page_size"] == 1
    assert summary_json["items"][0]["merchant_rubro_proxy"] == "5944"
    assert summary_json["items"][0]["merchant_rubro_values"] == "5944|6011"
    assert summary_json["items"][0]["top_merchant_rubro_proxy"] == "5944"
    assert "is_fraud" not in summary_response.text
    assert "confirmed_fraud" not in summary_response.text

    summary_options = test_client.get("/api/rules/summary-filter-options", params={"run_id": "preprocessed_run_26"})
    assert summary_options.status_code == 200
    options_json = summary_options.json()
    assert "RULE_DOUBLE_COUNTRY_CARD_PRESENT_SAME_DAY" in options_json["rule_code"]
    assert "RULE_GAMBLING_MCC" in options_json["rule_code"]
    assert "RULE_VELOCITY_CARD_HOUR" in options_json["rule_code"]
    assert "BO" in options_json["country_code"]
    assert "AR" in options_json["country_code"]
    assert "6011" in options_json["merchant_rubro_proxy"]
    assert "7995" in options_json["merchant_rubro_proxy"]

    for returned_rule_code in options_json["rule_code"]:
        filtered_summary = test_client.get(
            "/api/rules/summary",
            params={
                "run_id": "preprocessed_run_26",
                "rule_code": returned_rule_code,
                "page": 1,
                "page_size": 20,
            },
        )
        assert filtered_summary.status_code == 200
        assert filtered_summary.json()["total_items"] > 0

    gambling_summary = test_client.get(
        "/api/rules/summary",
        params={"run_id": "preprocessed_run_26", "rule_code": "RULE_GAMBLING_MCC", "page": 1, "page_size": 20},
    )
    assert gambling_summary.status_code == 200
    assert gambling_summary.json()["total_items"] > 0

    velocity_summary = test_client.get(
        "/api/rules/summary",
        params={"run_id": "preprocessed_run_26", "rule_code": "RULE_VELOCITY_CARD_HOUR", "page": 1, "page_size": 20},
    )
    assert velocity_summary.status_code == 200
    assert velocity_summary.json()["total_items"] > 0

    alerts_response = test_client.get("/api/rules/alerts", params={"run_id": "preprocessed_run_26", "page": 1, "page_size": 1})
    assert alerts_response.status_code == 200
    alerts_json = alerts_response.json()
    assert alerts_json["total_items"] == 4
    assert len(alerts_json["items"]) == 1
    assert alerts_json["page_size"] == 1

    report_response = test_client.get("/api/rules/report", params={"run_id": "preprocessed_run_26"})
    assert report_response.status_code == 200
    assert "sample report" in report_response.json()["report"]

    metrics_response = test_client.get("/api/rules/metrics", params={"run_id": "preprocessed_run_26"})
    assert metrics_response.status_code == 200
    metrics_json = metrics_response.json()
    assert metrics_json["total_alerts"] == 4
    assert metrics_json["total_summary_alerts"] == 3
    assert metrics_json["alerts_by_rule"]["RULE_DOUBLE_COUNTRY_CARD_PRESENT_SAME_DAY"] == 2
    assert metrics_json["alerts_by_rule"]["RULE_GAMBLING_MCC"] == 1
    assert metrics_json["alerts_by_rule"]["RULE_VELOCITY_CARD_HOUR"] == 1
    assert "is_fraud" not in metrics_response.text
    assert "confirmed_fraud" not in metrics_response.text

    detail_response = test_client.get("/api/rules/alerts/26-000001", params={"run_id": "preprocessed_run_26"})
    assert detail_response.status_code == 200
    assert detail_response.json()["alert_id"] == "26-000001"

    missing_response = test_client.get("/api/rules/summary", params={"run_id": "preprocessed_run_999", "page": 1, "page_size": 1})
    assert missing_response.status_code == 404

    missing_options = test_client.get("/api/rules/summary-filter-options", params={"run_id": "preprocessed_run_999"})
    assert missing_options.status_code == 404

    limited_page_size = test_client.get("/api/rules/alerts", params={"run_id": "preprocessed_run_26", "page_size": 999})
    assert limited_page_size.status_code == 200
    assert limited_page_size.json()["page_size"] == 200


def test_rule_summary_pagination_and_review_filters(monkeypatch, test_client, db_session, tmp_path):
    processed_dir = tmp_path / "processed"
    _write_large_summary_rule_files(processed_dir)
    monkeypatch.setattr(rule_routes, "_processed_dir", lambda: processed_dir)

    review = RuleAlertReview(
        source_run="preprocessed_run_26",
        summary_alert_id="26-S-000002",
        rule_code="RULE_DOUBLE_COUNTRY_CARD_PRESENT_SAME_DAY",
        new_status="IN_REVIEW",
        reviewed_at=datetime.now(timezone.utc),
    )
    db_session.add(review)
    db_session.commit()

    page1 = test_client.get("/api/rules/summary", params={"run_id": "preprocessed_run_26", "page": 1, "page_size": 20})
    assert page1.status_code == 200
    page1_json = page1.json()
    assert page1_json["total_items"] == 45
    assert page1_json["total_pages"] == 3
    assert len(page1_json["items"]) == 20

    page2 = test_client.get("/api/rules/summary", params={"run_id": "preprocessed_run_26", "page": 2, "page_size": 20})
    assert page2.status_code == 200
    assert len(page2.json()["items"]) == 20

    page3 = test_client.get("/api/rules/summary", params={"run_id": "preprocessed_run_26", "page": 3, "page_size": 20})
    assert page3.status_code == 200
    page3_json = page3.json()
    assert len(page3_json["items"]) == 5
    assert any(item["summary_alert_id"] == "26-S-000042" for item in page3_json["items"])
    affected = next(item for item in page3_json["items"] if item["summary_alert_id"] == "26-S-000042")
    assert affected["window_start"] is None
    assert affected["window_end"] is None
    assert "NaN" not in page3.text

    out_of_range = test_client.get("/api/rules/summary", params={"run_id": "preprocessed_run_26", "page": 99, "page_size": 20})
    assert out_of_range.status_code == 200
    assert out_of_range.json()["items"] == []

    in_review = test_client.get("/api/rules/summary", params={"run_id": "preprocessed_run_26", "status": "IN_REVIEW"})
    assert in_review.status_code == 200
    in_review_json = in_review.json()
    assert in_review_json["total_items"] == 1
    assert in_review_json["items"][0]["summary_alert_id"] == "26-S-000002"
    assert in_review_json["items"][0]["status"] == "IN_REVIEW"

    new_status = test_client.get("/api/rules/summary", params={"run_id": "preprocessed_run_26", "status": "NEW"})
    assert new_status.status_code == 200
    assert all(item["summary_alert_id"] != "26-S-000002" for item in new_status.json()["items"])

    risk_level = test_client.get("/api/rules/summary", params={"run_id": "preprocessed_run_26", "risk_level": "HIGH"})
    assert risk_level.status_code == 200
    assert risk_level.json()["total_items"] > 0

    rule_code = test_client.get(
        "/api/rules/summary",
        params={"run_id": "preprocessed_run_26", "rule_code": "RULE_DOUBLE_COUNTRY_CARD_PRESENT_SAME_DAY"},
    )
    assert rule_code.status_code == 200
    assert rule_code.json()["total_items"] == 45

    country_filter = test_client.get("/api/rules/summary", params={"run_id": "preprocessed_run_26", "country_code": "BO"})
    assert country_filter.status_code == 200
    assert country_filter.json()["total_items"] == 45

    merchant_filter = test_client.get("/api/rules/summary", params={"run_id": "preprocessed_run_26", "merchant_rubro_proxy": "6011"})
    assert merchant_filter.status_code == 200
    assert merchant_filter.json()["total_items"] > 0

    no_merchant_filter = test_client.get("/api/rules/summary", params={"run_id": "preprocessed_run_26", "merchant_rubro_proxy": "NO_EXISTE"})
    assert no_merchant_filter.status_code == 200
    assert no_merchant_filter.json()["items"] == []

    missing_run = test_client.get("/api/rules/summary", params={"run_id": "preprocessed_run_999", "page": 1, "page_size": 20})
    assert missing_run.status_code == 404
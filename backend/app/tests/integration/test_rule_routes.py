from __future__ import annotations

import hashlib
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
                "child_alert_ids": "26-000001|26-000002",
                "child_transaction_ids": "tx-1|tx-2",
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
                "child_alert_ids": "26-000003",
                "child_transaction_ids": "tx-3",
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
                "child_alert_ids": "26-000004",
                "child_transaction_ids": "tx-4",
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
                "child_alert_ids": "|".join([f"26-{index:06d}"] * max(index, 1)),
                "child_transaction_ids": "|".join([f"tx-{index}"] * max(index, 1)),
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


def _write_customer_card_lookup_files(base_dir: Path) -> str:
    base_dir.mkdir(parents=True, exist_ok=True)

    pan_value = "4698261234568047"
    customer_hash = f"cust_{hashlib.sha256(pan_value.encode('utf-8')).hexdigest()[:16]}"
    lookup = pd.DataFrame(
        [
            {
                "transaction_id": "tx-lookup-1",
                "customer_hash": customer_hash,
                "pan_card": pan_value,
                "masked_card": "469826******8047",
                "transaction_datetime": "2026-05-28T10:00:00+00:00",
                "amount": 10.0,
            },
            {
                "transaction_id": "tx-lookup-2",
                "customer_hash": "cust_no_match",
                "pan_card": "4111111111111111",
                "masked_card": "411111******1111",
                "transaction_datetime": "2026-05-28T10:05:00+00:00",
                "amount": 20.0,
            },
        ]
    )
    lookup.to_csv(base_dir / "uploaded_lookup.csv", index=False)
    return customer_hash


def _write_summary_transactions_rule_files(base_dir: Path, customer_hash: str, *, include_alert_rows: bool = True, include_child_ids: bool = True) -> str:
    base_dir.mkdir(parents=True, exist_ok=True)

    summary_alert_id = "26-S-5186b35686f8"
    summary_rows = [
        {
            "summary_alert_id": summary_alert_id,
            "source_run": "26",
            "customer_hash": customer_hash,
            "rule_code": "RULE_DOUBLE_COUNTRY_CARD_PRESENT_SAME_DAY",
            "rule_name": "Double country card present same day",
            "risk_level": "HIGH",
            "max_risk_score": 95.0,
            "count_transactions": 3,
            "countries_detected": "BO|AR",
            **({"child_alert_ids": "26-000101|26-000102|26-000103", "child_transaction_ids": "tx-001|tx-002|tx-003"} if include_child_ids else {}),
            "merchant_rubro_proxy": "5944",
            "window_start": "2026-04-14T00:00:00+00:00",
            "window_end": "2026-04-14T00:30:00+00:00",
            "representative_transaction_id": "tx-001",
            "status": "NEW",
            "created_at": "2026-04-14T01:00:00+00:00",
        }
    ]

    if include_alert_rows:
        alert_rows = [
            {
                "alert_id": "26-000101",
                "source_run": "26",
                "transaction_id": "tx-001",
                "customer_hash": customer_hash,
                "transaction_datetime": "2026-04-14T00:07:46+00:00",
                "amount": 123.45,
                "country_code": "BO",
                "pos_entry_mode": "7",
                "merchant_rubro_proxy": "5944",
                "merchant_name": "Mercado Uno",
                "has_pinblock": 0,
                "risk_score": 85,
                "rule_code": "RULE_DOUBLE_COUNTRY_CARD_PRESENT_SAME_DAY",
                "rule_name": "Double country card present same day",
                "status": "NEW",
                "pan_card": "4698261234568047",
            },
            {
                "alert_id": "26-000102",
                "source_run": "26",
                "transaction_id": "tx-002",
                "customer_hash": customer_hash,
                "transaction_datetime": "2026-04-14T00:15:00+00:00",
                "amount": 88.0,
                "country_code": "AR",
                "pos_entry_mode": "7",
                "merchant_rubro_proxy": "5944",
                "merchant_name": "Mercado Dos",
                "has_pinblock": 1,
                "risk_score": 91,
                "rule_code": "RULE_DOUBLE_COUNTRY_CARD_PRESENT_SAME_DAY",
                "rule_name": "Double country card present same day",
                "status": "NEW",
                "pan_card": "4698261234568047",
            },
            {
                "alert_id": "26-000103",
                "source_run": "26",
                "transaction_id": "tx-003",
                "customer_hash": customer_hash,
                "transaction_datetime": "2026-04-14T00:30:00+00:00",
                "amount": 59.5,
                "country_code": "BO",
                "pos_entry_mode": "5",
                "merchant_rubro_proxy": "5944",
                "merchant_name": None,
                "has_pinblock": 0,
                "risk_score": 95,
                "rule_code": "RULE_DOUBLE_COUNTRY_CARD_PRESENT_SAME_DAY",
                "rule_name": "Double country card present same day",
                "status": "NEW",
                "pan_card": "4698261234568047",
            },
            {
                "alert_id": "26-000104",
                "source_run": "26",
                "transaction_id": "tx-999",
                "customer_hash": "cust_other",
                "transaction_datetime": "2026-04-15T00:30:00+00:00",
                "amount": 11.0,
                "country_code": "US",
                "pos_entry_mode": "7",
                "merchant_rubro_proxy": "6011",
                "merchant_name": "Other",
                "has_pinblock": 0,
                "risk_score": 10,
                "rule_code": "RULE_VELOCITY_CARD_DAY",
                "rule_name": "Velocity card day",
                "status": "NEW",
            },
        ]
        pd.DataFrame(alert_rows).to_csv(base_dir / "alerts_run_26.csv", index=False)
    else:
        pd.DataFrame(columns=["alert_id", "source_run", "transaction_id", "customer_hash", "transaction_datetime", "amount", "country_code", "pos_entry_mode", "merchant_rubro_proxy", "merchant_name", "has_pinblock", "risk_score", "rule_code", "rule_name", "status"]).to_csv(base_dir / "alerts_run_26.csv", index=False)

    pd.DataFrame(summary_rows).to_csv(base_dir / "alerts_summary_run_26.csv", index=False)
    return summary_alert_id


def _write_summary_transactions_double_country_aw_bo_files(
    base_dir: Path,
    customer_hash: str,
    *,
    include_aw_preprocessed: bool = True,
    narrowed_window: bool = False,
) -> str:
    base_dir.mkdir(parents=True, exist_ok=True)

    summary_alert_id = "26-S-AW-BO-0001"
    pd.DataFrame(
        [
            {
                "summary_alert_id": summary_alert_id,
                "source_run": "26",
                "customer_hash": customer_hash,
                "rule_code": "RULE_DOUBLE_COUNTRY_CARD_PRESENT_SAME_DAY",
                "rule_name": "Double country card present same day",
                "risk_level": "HIGH",
                "max_risk_score": 95.0,
                "count_transactions": 4,
                "countries_detected": "AW|BO",
                "alert_reason": "Countries: AW, BO",
                "child_alert_ids": "26-000201|26-000202|26-000203",
                "child_transaction_ids": "tx_dd46d83098108114|tx_66388fe04992501e|tx_8a9f4d611a19303f",
                "merchant_rubro_proxy": "5944",
                "window_start": "2026-04-14T00:00:00+00:00",
                "window_end": "2026-04-14T15:30:57.643000+00:00" if narrowed_window else "2026-04-14T23:59:59+00:00",
                "representative_transaction_id": "tx_8a9f4d611a19303f",
                "status": "NEW",
                "created_at": "2026-04-14T01:00:00+00:00",
            }
        ]
    ).to_csv(base_dir / "alerts_summary_run_26.csv", index=False)

    pd.DataFrame(
        [
            {
                "alert_id": "26-000201",
                "source_run": "26",
                "transaction_id": "tx_dd46d83098108114",
                "customer_hash": customer_hash,
                "transaction_datetime": "2026-04-14T15:30:57.643000+00:00",
                "amount": 41.25,
                "country_code": "BO",
                "pos_entry_mode": "7",
                "merchant_rubro_proxy": "5944",
                "merchant_name": "Mercado Uno",
                "has_pinblock": 0,
                "risk_score": 85,
                "rule_code": "RULE_DOUBLE_COUNTRY_CARD_PRESENT_SAME_DAY",
                "rule_name": "Double country card present same day",
                "status": "NEW",
                "pan_card": "4698261234568047",
            },
            {
                "alert_id": "26-000202",
                "source_run": "26",
                "transaction_id": "tx_66388fe04992501e",
                "customer_hash": customer_hash,
                "transaction_datetime": "2026-04-14T00:17:22.969000+00:00",
                "amount": 88.0,
                "country_code": "BO",
                "pos_entry_mode": "7",
                "merchant_rubro_proxy": "5944",
                "merchant_name": "Mercado Dos",
                "has_pinblock": 1,
                "risk_score": 91,
                "rule_code": "RULE_DOUBLE_COUNTRY_CARD_PRESENT_SAME_DAY",
                "rule_name": "Double country card present same day",
                "status": "NEW",
                "pan_card": "4698261234568047",
            },
            {
                "alert_id": "26-000203",
                "source_run": "26",
                "transaction_id": "tx_8a9f4d611a19303f",
                "customer_hash": customer_hash,
                "transaction_datetime": "2026-04-14T00:07:46.246000+00:00",
                "amount": 59.5,
                "country_code": "BO",
                "pos_entry_mode": "5",
                "merchant_rubro_proxy": "5944",
                "merchant_name": "Mercado Tres",
                "has_pinblock": 0,
                "risk_score": 95,
                "rule_code": "RULE_DOUBLE_COUNTRY_CARD_PRESENT_SAME_DAY",
                "rule_name": "Double country card present same day",
                "status": "NEW",
                "pan_card": "4698261234568047",
            },
        ]
    ).to_csv(base_dir / "alerts_run_26.csv", index=False)

    preprocessed_rows = [
        {
            "transaction_id": "tx_dd46d83098108114",
            "amount": 41.25,
            "transaction_type": 1,
            "channel": "LPZ1POSID4",
            "location": "EL ALTO",
            "device_id": "04401901",
            "customer_hash": customer_hash,
            "merchant_hash": "merch_bo_1",
            "merchant_code": "000000430286",
            "terminal_code": "04401901",
            "merchant_name": "Mercado Uno",
            "merchant_rubro_proxy": "5944",
            "country_code": "BO",
            "pos_entry_mode": "7",
            "has_pinblock": 0,
            "card_brand": "UNKNOWN",
            "transaction_datetime": "2026-04-14 15:30:57.643",
            "has_pinblock_source": "raw",
            "hour": 15,
            "day": 14,
            "weekday": 2,
            "is_weekend": 0,
            "is_night": 0,
            "is_international": 0,
            "card_presence_type": "TP",
        },
        {
            "transaction_id": "tx_66388fe04992501e",
            "amount": 88.0,
            "transaction_type": 1,
            "channel": "LPZ1POSID4",
            "location": "LA PAZ",
            "device_id": "04401901",
            "customer_hash": customer_hash,
            "merchant_hash": "merch_bo_2",
            "merchant_code": "000000430286",
            "terminal_code": "04401901",
            "merchant_name": "Mercado Dos",
            "merchant_rubro_proxy": "5944",
            "country_code": "BO",
            "pos_entry_mode": "7",
            "has_pinblock": 1,
            "card_brand": "UNKNOWN",
            "transaction_datetime": "2026-04-14 00:17:22.969",
            "has_pinblock_source": "raw",
            "hour": 0,
            "day": 14,
            "weekday": 2,
            "is_weekend": 0,
            "is_night": 1,
            "is_international": 0,
            "card_presence_type": "TP",
        },
        {
            "transaction_id": "tx_8a9f4d611a19303f",
            "amount": 59.5,
            "transaction_type": 1,
            "channel": "LPZ1POSID4",
            "location": "LA PAZ",
            "device_id": "04401901",
            "customer_hash": customer_hash,
            "merchant_hash": "merch_bo_3",
            "merchant_code": "000000430286",
            "terminal_code": "04401901",
            "merchant_name": "Mercado Tres",
            "merchant_rubro_proxy": "5944",
            "country_code": "BO",
            "pos_entry_mode": "5",
            "has_pinblock": 0,
            "card_brand": "UNKNOWN",
            "transaction_datetime": "2026-04-14 00:07:46.246",
            "has_pinblock_source": "raw",
            "hour": 0,
            "day": 14,
            "weekday": 2,
            "is_weekend": 0,
            "is_night": 1,
            "is_international": 0,
            "card_presence_type": "TP",
        },
    ]

    if include_aw_preprocessed:
        preprocessed_rows.insert(
            0,
            {
                "transaction_id": "tx_12b1c7eba9876095",
                "amount": 32.0,
                "transaction_type": 1,
                "channel": "LPZ1POSID4",
                "location": "COCHABAMBA",
                "device_id": "04401901",
                "customer_hash": customer_hash,
                "merchant_hash": "merch_aw_1",
                "merchant_code": "000000430286",
                "terminal_code": "04401901",
                "merchant_name": "Mercado AW",
                "merchant_rubro_proxy": "5944",
                "country_code": "AW",
                "pos_entry_mode": "7",
                "has_pinblock": 0,
                "card_brand": "UNKNOWN",
                "transaction_datetime": "2026-04-14 23:40:32.407",
                "has_pinblock_source": "raw",
                "hour": 23,
                "day": 14,
                "weekday": 2,
                "is_weekend": 0,
                "is_night": 1,
                "is_international": 0,
                "card_presence_type": "TP",
            },
        )

    pd.DataFrame(preprocessed_rows).to_csv(base_dir / "preprocessed_run_26.csv", index=False)
    return summary_alert_id


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
    assert "transaction_date" in summary_json["items"][0]
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


def test_rule_analyze_force_reuses_numeric_run_artifact_names(monkeypatch, test_client, tmp_path):
    processed_dir = tmp_path / "processed"
    _write_sample_rule_files(processed_dir)
    monkeypatch.setattr(rule_routes, "_processed_dir", lambda: processed_dir)

    response = test_client.post(
        "/api/rules/analyze",
        json={"preprocessed_run_id": "preprocessed_run_26", "force": True, "config": {}},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "COMPLETED"
    assert payload["alerts_file"] == "alerts_run_26.csv"
    assert payload["summary_file"] == "alerts_summary_run_26.csv"
    assert payload["report_file"] == "rules_report_run_26.md"
    assert (processed_dir / "alerts_run_26.csv").exists()
    assert (processed_dir / "alerts_summary_run_26.csv").exists()
    assert not (processed_dir / "alerts_run_preprocessed_run_26.csv").exists()
    assert not (processed_dir / "alerts_summary_run_preprocessed_run_26.csv").exists()


def test_customer_card_lookup_returns_masked_card(monkeypatch, test_client, tmp_path):
    uploads_dir = tmp_path / "uploads"
    customer_hash = _write_customer_card_lookup_files(uploads_dir)
    monkeypatch.setattr(rule_routes, "_uploads_dir", lambda: uploads_dir)

    response = test_client.get("/api/rules/customer-card-lookup", params={"customer_hash": customer_hash})
    assert response.status_code == 200
    payload = response.json()
    assert payload["customer_hash"] == customer_hash
    assert payload["available"] is True
    assert payload["masked_card"] == "469826******8047"
    assert payload["last4"] == "8047"
    assert "4698261234568047" not in response.text


def test_customer_card_lookup_returns_unavailable_for_missing_mapping(monkeypatch, test_client, tmp_path):
    uploads_dir = tmp_path / "uploads"
    _write_customer_card_lookup_files(uploads_dir)
    monkeypatch.setattr(rule_routes, "_uploads_dir", lambda: uploads_dir)

    response = test_client.get("/api/rules/customer-card-lookup", params={"customer_hash": "cust_missing"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["customer_hash"] == "cust_missing"
    assert payload["available"] is False
    assert payload["masked_card"] is None
    assert payload["last4"] is None


def test_summary_transactions_returns_grouped_transactions(monkeypatch, test_client, tmp_path):
    processed_dir = tmp_path / "processed"
    uploads_dir = tmp_path / "uploads"
    customer_hash = _write_customer_card_lookup_files(uploads_dir)
    summary_alert_id = _write_summary_transactions_rule_files(processed_dir, customer_hash)
    monkeypatch.setattr(rule_routes, "_processed_dir", lambda: processed_dir)
    monkeypatch.setattr(rule_routes, "_uploads_dir", lambda: uploads_dir)

    response = test_client.get(
        "/api/rules/summary-transactions",
        params={"run_id": "preprocessed_run_26", "alert_id": summary_alert_id},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["run_id"] == "preprocessed_run_26"
    assert payload["alert_id"] == summary_alert_id
    assert payload["total_transactions"] == 3
    assert [item["transaction_id"] for item in payload["items"]] == ["tx-001", "tx-002", "tx-003"]
    assert payload["items"][0]["transaction_datetime"] == "2026-04-14T00:07:46+00:00"
    assert payload["items"][0]["masked_card"] == "469826******8047"
    assert payload["items"][1]["masked_card"] == "469826******8047"
    assert payload["items"][0]["merchant_name"] == "Mercado Uno"
    assert payload["warning"] is None
    assert "4698261234568047" not in response.text
    assert "pan_card" not in response.text


def test_summary_transactions_reconstructs_double_country_transactions_from_preprocessed(monkeypatch, test_client, tmp_path):
    processed_dir = tmp_path / "processed"
    uploads_dir = tmp_path / "uploads"
    customer_hash = _write_customer_card_lookup_files(uploads_dir)
    summary_alert_id = _write_summary_transactions_double_country_aw_bo_files(processed_dir, customer_hash, include_aw_preprocessed=True)
    monkeypatch.setattr(rule_routes, "_processed_dir", lambda: processed_dir)
    monkeypatch.setattr(rule_routes, "_uploads_dir", lambda: uploads_dir)

    response = test_client.get(
        "/api/rules/summary-transactions",
        params={"run_id": "preprocessed_run_26", "alert_id": summary_alert_id},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["run_id"] == "preprocessed_run_26"
    assert payload["alert_id"] == summary_alert_id
    assert payload["warning"] is None
    assert payload["total_transactions"] == len(payload["items"])
    assert any(item["country_code"] == "AW" for item in payload["items"])
    assert any(item["country_code"] == "BO" for item in payload["items"])
    countries = []
    for item in payload["items"]:
        country = item["country_code"]
        if country and country not in countries:
            countries.append(country)
    assert set(countries) == {"AW", "BO"}
    assert "4698261234568047" not in response.text
    assert "confirmed_fraud" not in response.text


def test_summary_transactions_reconstructs_missing_country_with_day_scope(monkeypatch, test_client, tmp_path):
    processed_dir = tmp_path / "processed"
    uploads_dir = tmp_path / "uploads"
    customer_hash = _write_customer_card_lookup_files(uploads_dir)
    summary_alert_id = _write_summary_transactions_double_country_aw_bo_files(
        processed_dir,
        customer_hash,
        include_aw_preprocessed=True,
        narrowed_window=True,
    )
    monkeypatch.setattr(rule_routes, "_processed_dir", lambda: processed_dir)
    monkeypatch.setattr(rule_routes, "_uploads_dir", lambda: uploads_dir)

    response = test_client.get(
        "/api/rules/summary-transactions",
        params={"run_id": "preprocessed_run_26", "alert_id": summary_alert_id},
    )

    assert response.status_code == 200
    payload = response.json()
    countries = {
        item["country_code"]
        for item in payload["items"]
        if item.get("country_code")
    }
    assert countries == {"AW", "BO"}
    assert payload["warning"] is None


def test_summary_transactions_warns_when_double_country_resolves_to_one_country(monkeypatch, test_client, tmp_path):
    processed_dir = tmp_path / "processed"
    uploads_dir = tmp_path / "uploads"
    customer_hash = _write_customer_card_lookup_files(uploads_dir)
    summary_alert_id = _write_summary_transactions_double_country_aw_bo_files(processed_dir, customer_hash, include_aw_preprocessed=False)
    monkeypatch.setattr(rule_routes, "_processed_dir", lambda: processed_dir)
    monkeypatch.setattr(rule_routes, "_uploads_dir", lambda: uploads_dir)

    response = test_client.get(
        "/api/rules/summary-transactions",
        params={"run_id": "preprocessed_run_26", "alert_id": summary_alert_id},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["warning"] is not None
    assert payload["warning"] == "La alerta indica países AW|BO, pero no se encontró transacción hija para AW."


def test_summary_transactions_uses_child_transaction_ids_when_present(monkeypatch, test_client, tmp_path):
    processed_dir = tmp_path / "processed"
    uploads_dir = tmp_path / "uploads"
    customer_hash = _write_customer_card_lookup_files(uploads_dir)
    summary_alert_id = _write_summary_transactions_rule_files(processed_dir, customer_hash, include_child_ids=True)
    monkeypatch.setattr(rule_routes, "_processed_dir", lambda: processed_dir)
    monkeypatch.setattr(rule_routes, "_uploads_dir", lambda: uploads_dir)

    response = test_client.get(
        "/api/rules/summary-transactions",
        params={"run_id": "preprocessed_run_26", "alert_id": summary_alert_id},
    )

    assert response.status_code == 200
    payload = response.json()
    assert [item["transaction_id"] for item in payload["items"]] == ["tx-001", "tx-002", "tx-003"]
    assert payload["warning"] is None


def test_summary_transactions_enriches_merchant_name_from_preprocessed(monkeypatch, test_client, tmp_path):
    processed_dir = tmp_path / "processed"
    uploads_dir = tmp_path / "uploads"
    customer_hash = _write_customer_card_lookup_files(uploads_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    summary_alert_id = "26-S-MERCHANT-0001"

    pd.DataFrame(
        [
            {
                "summary_alert_id": summary_alert_id,
                "source_run": "26",
                "customer_hash": customer_hash,
                "rule_code": "RULE_VELOCITY_CARD_HOUR",
                "rule_name": "Velocity card hour",
                "risk_level": "MEDIUM",
                "max_risk_score": 60.0,
                "count_transactions": 1,
                "countries_detected": "BO",
                "child_transaction_ids": "tx-merchant-001",
                "window_start": "2026-04-06T12:38:00+00:00",
                "window_end": "2026-04-06T12:39:00+00:00",
                "representative_transaction_id": "tx-merchant-001",
                "status": "NEW",
                "created_at": "2026-04-06T12:40:00+00:00",
            }
        ]
    ).to_csv(processed_dir / "alerts_summary_run_26.csv", index=False)
    pd.DataFrame(
        [
            {
                "alert_id": "26-000501",
                "source_run": "26",
                "transaction_id": "tx-merchant-001",
                "customer_hash": customer_hash,
                "transaction_datetime": "2026-04-06T12:38:32+00:00",
                "amount": 199.0,
                "country_code": "BO",
                "pos_entry_mode": "1",
                "merchant_rubro_proxy": "4814",
                "risk_score": 60,
                "rule_code": "RULE_VELOCITY_CARD_HOUR",
                "rule_name": "Velocity card hour",
                "status": "NEW",
            }
        ]
    ).to_csv(processed_dir / "alerts_run_26.csv", index=False)
    pd.DataFrame(
        [
            {
                "transaction_id": "tx-merchant-001",
                "customer_hash": customer_hash,
                "transaction_datetime": "2026-04-06 12:38:32",
                "amount": 199.0,
                "country_code": "BO",
                "pos_entry_mode": "1",
                "merchant_rubro_proxy": "4814",
                "merchant_name": "TELEFONIA LA PAZ",
            }
        ]
    ).to_csv(processed_dir / "preprocessed_run_26.csv", index=False)
    monkeypatch.setattr(rule_routes, "_processed_dir", lambda: processed_dir)
    monkeypatch.setattr(rule_routes, "_uploads_dir", lambda: uploads_dir)

    response = test_client.get(
        "/api/rules/summary-transactions",
        params={"run_id": "preprocessed_run_26", "alert_id": summary_alert_id},
    )

    assert response.status_code == 200
    item = response.json()["items"][0]
    assert item["merchant_name"] == "TELEFONIA LA PAZ"
    assert item["risk_score"] == 60


def test_summary_transactions_respects_run_id_and_returns_empty_when_no_detail(monkeypatch, test_client, tmp_path):
    processed_dir = tmp_path / "processed"
    uploads_dir = tmp_path / "uploads"
    customer_hash = _write_customer_card_lookup_files(uploads_dir)
    summary_alert_id = _write_summary_transactions_rule_files(processed_dir, customer_hash, include_alert_rows=False)
    monkeypatch.setattr(rule_routes, "_processed_dir", lambda: processed_dir)
    monkeypatch.setattr(rule_routes, "_uploads_dir", lambda: uploads_dir)

    empty_response = test_client.get(
        "/api/rules/summary-transactions",
        params={"run_id": "preprocessed_run_26", "alert_id": summary_alert_id},
    )
    assert empty_response.status_code == 200
    assert empty_response.json()["total_transactions"] == 0
    assert empty_response.json()["items"] == []

    missing_run = test_client.get(
        "/api/rules/summary-transactions",
        params={"run_id": "preprocessed_run_999", "alert_id": summary_alert_id},
    )
    assert missing_run.status_code == 404


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

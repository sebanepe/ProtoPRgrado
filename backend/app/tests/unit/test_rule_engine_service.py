from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd

from backend.app.ml.validate_alerts import READY_VERDICT, NOT_READY_VERDICT, validate as validate_alerts
from backend.app.ml.validate_alert_summary import READY_VERDICT as SUMMARY_READY_VERDICT, validate as validate_alert_summary
from backend.app.rules.fraud_rules import (
    DOUBLE_COUNTRY_RULE,
    VELOCITY_HOUR_RULE,
    VELOCITY_DAY_RULE,
    CONTACTLESS_NO_PIN_HOUR_RULE,
    INTERNET_VELOCITY_DAY_RULE,
    MAGSTRIPE_VELOCITY_HOUR_RULE,
    ATM_VELOCITY_HOUR_RULE,
    ATM_VELOCITY_DAY_RULE,
    GAMBLING_MCC_RULE,
    JEWELRY_HIGH_AMOUNT_RULE,
    _normalize_mcc_code,
    evaluate_transaction_rules,
)
from backend.app.services.rule_engine_service import build_alert_summary_df, generate_alerts_from_preprocessed_csv


BASE_DT = datetime(2026, 5, 28, 10, 0, 0)


def make_tx(
    index: int,
    *,
    customer_hash: str = "cust-1",
    country_code: str = "BO",
    pos_entry_mode: str | int | float = "7",
    has_pinblock: int = 0,
    amount: float = 100.0,
    merchant_rubro_proxy: str | int = "0000",
    minutes: int = 0,
    days: int = 0,
) -> dict[str, object]:
    dt = BASE_DT + timedelta(days=days, minutes=minutes)
    return {
        "transaction_id": f"tx-{index}",
        "customer_hash": customer_hash,
        "transaction_datetime": dt.isoformat(),
        "amount": amount,
        "country_code": country_code,
        "pos_entry_mode": pos_entry_mode,
        "has_pinblock": has_pinblock,
        "merchant_rubro_proxy": merchant_rubro_proxy,
    }


def tx_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_mcc_three_digits_are_padded():
    assert _normalize_mcc_code("742") == "0742"
    assert _normalize_mcc_code("763") == "0763"
    assert _normalize_mcc_code("780") == "0780"


def test_double_country_same_day_excludes_unknown():
    df = tx_frame(
        [
            make_tx(1, country_code="BO", minutes=0),
            make_tx(2, country_code="AR", minutes=5),
            make_tx(3, country_code="UNKNOWN", minutes=10),
        ]
    )

    alerts_df, summary = evaluate_transaction_rules(df, config={"source_run": "26"})

    dc_alerts = alerts_df.loc[alerts_df["rule_code"].astype(str).str.startswith("RULE_DOUBLE_COUNTRY")]
    assert len(dc_alerts) == 2
    assert set(dc_alerts["country_code"].astype(str)) == {"BO", "AR"}
    total_dc = sum(v for k, v in (summary.get("alerts_by_rule_code") or {}).items() if str(k).startswith("RULE_DOUBLE_COUNTRY"))
    assert total_dc == 2


def test_double_country_summary_groups_by_customer_and_date():
    df = tx_frame(
        [
            make_tx(1, country_code="BO", minutes=0),
            make_tx(2, country_code="AR", minutes=5),
            make_tx(3, country_code="BO", minutes=10),
        ]
    )

    alerts_df, _ = evaluate_transaction_rules(df, config={"source_run": "26"})
    summary_df = build_alert_summary_df(alerts_df)

    assert len(alerts_df.loc[alerts_df["rule_code"].astype(str).str.startswith("RULE_DOUBLE_COUNTRY")]) == 3
    assert len(summary_df) == 1
    row = summary_df.iloc[0]
    assert row["summary_alert_id"].startswith("26-S-")
    assert row["customer_hash"] == "cust-1"
    assert str(row["rule_code"]).startswith("RULE_DOUBLE_COUNTRY")
    assert row["count_transactions"] == 3
    assert row["countries_detected"] == "AR|BO"
    assert row["representative_transaction_id"] == "tx-1"
    assert str(row["status"]).upper() == "NEW"


def test_summary_validator_accepts_grouped_summary(tmp_path):
    df = tx_frame(
        [
            make_tx(1, country_code="BO", minutes=0),
            make_tx(2, country_code="AR", minutes=5),
            make_tx(3, country_code="BO", minutes=10),
        ]
    )
    alerts_df, _ = evaluate_transaction_rules(df, config={"source_run": "26"})
    summary_df = build_alert_summary_df(alerts_df)
    path = tmp_path / "alerts_summary_run_26.csv"
    summary_df.to_csv(path, index=False)

    report = validate_alert_summary(path)

    assert report["verdict"] == SUMMARY_READY_VERDICT
    assert report["missing_required_columns"] == []
    assert report["forbidden_columns_present"] == []


def test_velocity_hour_triggers_above_threshold():
    df = tx_frame([make_tx(i, minutes=0) for i in range(1, 5)])

    alerts_df, _ = evaluate_transaction_rules(df, config={"source_run": "26"})

    hour_alerts = alerts_df.loc[alerts_df["rule_code"] == VELOCITY_HOUR_RULE]
    assert len(hour_alerts) == 4


def test_velocity_day_triggers_above_threshold():
    df = tx_frame([make_tx(i, minutes=i * 5) for i in range(1, 12)])

    alerts_df, _ = evaluate_transaction_rules(df, config={"source_run": "26"})

    day_alerts = alerts_df.loc[alerts_df["rule_code"] == VELOCITY_DAY_RULE]
    assert len(day_alerts) == 11


def test_pem_10_is_excluded_from_internet_and_general_velocity():
    df = tx_frame([make_tx(i, pos_entry_mode=10, minutes=0) for i in range(1, 12)])

    alerts_df, _ = evaluate_transaction_rules(df, config={"source_run": "26"})

    assert alerts_df.empty
    assert INTERNET_VELOCITY_DAY_RULE not in set(alerts_df.get("rule_code", pd.Series(dtype=str)).astype(str))


def test_internet_velocity_accepts_pem_0_1_and_81():
    rows = []
    pem_values = [0, 1, 81] * 4
    for index in range(1, 12):
        rows.append(make_tx(index, pos_entry_mode=pem_values[index - 1], minutes=index * 2))

    alerts_df, _ = evaluate_transaction_rules(tx_frame(rows), config={"source_run": "26"})

    internet_alerts = alerts_df.loc[alerts_df["rule_code"] == INTERNET_VELOCITY_DAY_RULE]
    assert len(internet_alerts) == 11
    assert set(internet_alerts["pos_entry_mode"].astype(str)) <= {"0", "1", "81"}


def test_contactless_no_pin_hour_triggers():
    df = tx_frame([make_tx(i, pos_entry_mode=7, has_pinblock=0, minutes=0) for i in range(1, 7)])

    alerts_df, _ = evaluate_transaction_rules(df, config={"source_run": "26"})

    contactless_alerts = alerts_df.loc[alerts_df["rule_code"] == CONTACTLESS_NO_PIN_HOUR_RULE]
    assert len(contactless_alerts) == 6
    assert set(contactless_alerts["has_pinblock"].astype(int)) == {0}


def test_magstripe_velocity_hour_triggers():
    df = tx_frame([make_tx(i, pos_entry_mode=90, minutes=0) for i in range(1, 5)])

    alerts_df, _ = evaluate_transaction_rules(df, config={"source_run": "26"})

    magstripe_alerts = alerts_df.loc[alerts_df["rule_code"] == MAGSTRIPE_VELOCITY_HOUR_RULE]
    assert len(magstripe_alerts) == 4


def test_atm_mcc_6010_6011_trigger_hour_and_day_rules():
    df = tx_frame(
        [
            make_tx(1, merchant_rubro_proxy="6010", minutes=0),
            make_tx(2, merchant_rubro_proxy="6011", minutes=0),
            make_tx(3, merchant_rubro_proxy="6010", minutes=0),
            make_tx(4, merchant_rubro_proxy="6011", minutes=0),
            make_tx(5, merchant_rubro_proxy="6010", minutes=0),
            make_tx(6, merchant_rubro_proxy="6011", minutes=0),
        ]
    )

    alerts_df, _ = evaluate_transaction_rules(df, config={"source_run": "26"})

    assert len(alerts_df.loc[alerts_df["rule_code"] == ATM_VELOCITY_HOUR_RULE]) == 6
    assert len(alerts_df.loc[alerts_df["rule_code"] == ATM_VELOCITY_DAY_RULE]) == 6


def test_gambling_mcc_triggers():
    df = tx_frame([make_tx(i, merchant_rubro_proxy="7995", amount=400, minutes=0) for i in range(1, 2)])

    alerts_df, _ = evaluate_transaction_rules(df, config={"source_run": "26"})

    gambling_alerts = alerts_df.loc[alerts_df["rule_code"] == GAMBLING_MCC_RULE]
    assert len(gambling_alerts) == 1


def test_jewelry_mcc_high_amount_triggers():
    df = tx_frame([make_tx(1, merchant_rubro_proxy="5944", amount=1000, minutes=0)])

    alerts_df, _ = evaluate_transaction_rules(df, config={"source_run": "26"})

    jewelry_alerts = alerts_df.loc[alerts_df["rule_code"] == JEWELRY_HIGH_AMOUNT_RULE]
    assert len(jewelry_alerts) == 1


def test_alerts_output_has_no_forbidden_training_columns_and_status_new():
    df = tx_frame(
        [
            make_tx(1, country_code="BO", merchant_rubro_proxy="5944", amount=1000, minutes=0),
            make_tx(2, country_code="AR", merchant_rubro_proxy="5944", amount=1000, minutes=1),
        ]
    )

    alerts_df, _ = evaluate_transaction_rules(df, config={"source_run": "26"})

    assert "is_fraud" not in alerts_df.columns
    assert "confirmed_fraud" not in alerts_df.columns
    assert "DESCRIPTION" not in alerts_df.columns
    assert alerts_df["status"].astype(str).eq("NEW").all()


def test_validate_alerts_accepts_generated_alerts(tmp_path):
    df = tx_frame([make_tx(i, merchant_rubro_proxy="5944", amount=1000, minutes=0) for i in range(1, 2)])
    alerts_df, _ = evaluate_transaction_rules(df, config={"source_run": "26"})
    path = tmp_path / "alerts_run_26.csv"
    alerts_df.to_csv(path, index=False)

    report = validate_alerts(path)

    assert report["verdict"] == READY_VERDICT
    assert report["missing_required_columns"] == []


def test_validate_alerts_rejects_duplicates_and_forbidden_columns(tmp_path):
    df = pd.DataFrame(
        [
            {
                "alert_id": "26-000001",
                "source_run": "26",
                "transaction_id": "tx-1",
                "customer_hash": "cust-1",
                "transaction_datetime": "2026-05-28T10:00:00+00:00",
                "amount": 1000.0,
                "country_code": "BO",
                "pos_entry_mode": "7",
                "has_pinblock": 0,
                "merchant_rubro_proxy": "5944",
                "rule_code": JEWELRY_HIGH_AMOUNT_RULE,
                "rule_name": "Jewelry MCC High Amount",
                "risk_level": "HIGH",
                "risk_score": 80,
                "alert_reason": "x",
                "triggered_rules": JEWELRY_HIGH_AMOUNT_RULE,
                "status": "NEW",
                "created_at": "2026-05-28T10:00:00+00:00",
                "is_fraud": False,
            },
            {
                "alert_id": "26-000001",
                "source_run": "26",
                "transaction_id": "tx-1",
                "customer_hash": "cust-1",
                "transaction_datetime": "2026-05-28T10:00:00+00:00",
                "amount": 1000.0,
                "country_code": "BO",
                "pos_entry_mode": "7",
                "has_pinblock": 0,
                "merchant_rubro_proxy": "5944",
                "rule_code": JEWELRY_HIGH_AMOUNT_RULE,
                "rule_name": "Jewelry MCC High Amount",
                "risk_level": "HIGH",
                "risk_score": 80,
                "alert_reason": "x",
                "triggered_rules": JEWELRY_HIGH_AMOUNT_RULE,
                "status": "NEW",
                "created_at": "2026-05-28T10:00:00+00:00",
                "is_fraud": False,
            },
        ]
    )
    path = tmp_path / "alerts_run_26.csv"
    df.to_csv(path, index=False)

    report = validate_alerts(path)

    assert report["verdict"] == NOT_READY_VERDICT
    assert report["duplicate_alert_rows"] > 0
    assert any("is_fraud" in note for note in report["notes"])


def test_generate_alerts_from_preprocessed_csv_writes_expected_files(tmp_path):
    input_path = tmp_path / "preprocessed_run_26.csv"
    df = tx_frame(
        [
            make_tx(1, country_code="BO", minutes=0),
            make_tx(2, country_code="AR", minutes=5),
            make_tx(3, country_code="BO", minutes=10),
        ]
    )
    df.to_csv(input_path, index=False)

    result = generate_alerts_from_preprocessed_csv(str(input_path), output_dir=str(tmp_path))

    assert Path(result["alerts_path"]).name == "alerts_run_26.csv"
    assert Path(result["summary_path"]).name == "alerts_summary_run_26.csv"
    assert Path(result["report_path"]).name == "rules_report_run_26.md"
    assert Path(result["alerts_path"]).exists()
    assert Path(result["summary_path"]).exists()
    assert Path(result["report_path"]).exists()
    detailed_df = pd.read_csv(result["alerts_path"])
    summary_df = pd.read_csv(result["summary_path"])
    assert len(summary_df) < len(detailed_df)
    assert len(summary_df) == 1
    assert "is_fraud" not in summary_df.columns
    assert "confirmed_fraud" not in summary_df.columns

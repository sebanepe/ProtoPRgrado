from __future__ import annotations

from collections import Counter
import hashlib
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from backend.app.ml.validate_alerts import validate as validate_alerts
from backend.app.rules.fraud_rules import (
    DOUBLE_COUNTRY_CARD_ABSENT_RULE,
    DOUBLE_COUNTRY_CARD_PRESENT_RULE,
    DOUBLE_COUNTRY_RULE,
    evaluate_transaction_rules,
)


PROJECT_PROCESSED_DIR = os.environ.get("PROJECT_PROCESSED_DIR") or os.path.join(
    os.getcwd(), "data", "processed"
)


def _extract_run_token(path: str | os.PathLike[str], fallback: str = "UNKNOWN") -> str:
    stem = Path(path).stem
    match = re.search(r"(?:preprocessed|alerts|rules_report)_run_(\d+)", stem)
    if match:
        return match.group(1)
    match = re.search(r"(\d+)$", stem)
    if match:
        return match.group(1)
    return fallback


def _build_rules_report_markdown(source_path: str, alerts_df: pd.DataFrame, summary: dict[str, Any], validation: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Rules Report")
    lines.append("")
    lines.append(f"- source_file: {source_path}")
    lines.append(f"- source_run: {summary.get('source_run', 'UNKNOWN')}")
    lines.append(f"- generated_at: {datetime.now(timezone.utc).isoformat()}")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- input_rows: {summary.get('input_rows', 0)}")
    lines.append(f"- normalized_rows: {summary.get('normalized_rows', 0)}")
    lines.append(f"- detailed_alerts: {summary.get('alerts_generated', 0)}")
    lines.append(f"- grouped_alerts: {summary.get('grouped_alerts_generated', 0)}")
    lines.append(f"- grouping_reduction: {summary.get('grouping_reduction_pct', 0.0)}%")
    lines.append(f"- detailed_alerts_removed_by_grouping: {summary.get('detailed_alerts_removed_by_grouping', 0)}")
    lines.append(f"- double_country_detailed_before_grouping: {summary.get('double_country_detailed_alerts_before_grouping', 0)}")
    lines.append(f"- double_country_grouped_summary_cases: {summary.get('double_country_grouped_summary_cases', 0)}")
    lines.append(f"- double_country_excluded_pem10: {summary.get('double_country_excluded_pem10', 0)}")
    lines.append(f"- double_country_tnp_discarded_without_additional_signal: {summary.get('double_country_tnp_discarded_without_additional_signal', 0)}")
    lines.append("")
    lines.append("## Detailed Alerts By Rule")
    for rule_code, count in sorted((summary.get("alerts_by_rule_code", {}) or {}).items()):
        lines.append(f"- {rule_code}: {count}")
    lines.append("")
    lines.append("## Double Country By Rule")
    lines.append(f"- {DOUBLE_COUNTRY_CARD_PRESENT_RULE}: {summary.get('double_country_card_present_alerts', 0)}")
    lines.append(f"- {DOUBLE_COUNTRY_CARD_ABSENT_RULE}: {summary.get('double_country_card_absent_alerts', 0)}")
    lines.append(f"- deprecated_{DOUBLE_COUNTRY_RULE}: {summary.get('double_country_deprecated_alerts', 0)}")
    lines.append("")
    lines.append("## Grouped Alerts By Rule")
    for rule_code, count in sorted((summary.get("grouped_alerts_by_rule_code", {}) or {}).items()):
        lines.append(f"- {rule_code}: {count}")
    lines.append("")
    lines.append("## Double Country Distributions")
    lines.append("### Country Distribution")
    for rule_code, distribution in (summary.get("double_country_country_distribution", {}) or {}).items():
        lines.append(f"- {rule_code}")
        for country_code, count in sorted((distribution or {}).items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"  - {country_code}: {count}")
    lines.append("### PEM Distribution")
    for rule_code, distribution in (summary.get("double_country_pem_distribution", {}) or {}).items():
        lines.append(f"- {rule_code}")
        for pem, count in sorted((distribution or {}).items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"  - {pem}: {count}")
    lines.append("")
    lines.append("## Alerts By Risk Level")
    for risk_level, count in sorted((summary.get("alerts_by_risk_level", {}) or {}).items()):
        lines.append(f"- {risk_level}: {count}")
    lines.append("")
    lines.append("## Notes")
    lines.append("- alerts_run_N.csv preserves transaction-level traceability.")
    lines.append("- alerts_summary_run_N.csv is intended for operational review.")
    lines.append("- PEM 10 was excluded from automatic double-country rules because it corresponds to recurring/subscription activity.")
    lines.append("")
    lines.append("## Validation")
    lines.append(f"- verdict: {validation.get('verdict')}")
    for note in validation.get("notes", []) or []:
        lines.append(f"- note: {note}")
    lines.append("")
    lines.append("## Output Columns")
    lines.append(f"- {', '.join(alerts_df.columns.tolist())}")
    return "\n".join(lines) + "\n"


SUMMARY_COLUMNS = [
    "summary_alert_id",
    "source_run",
    "customer_hash",
    "rule_code",
    "rule_name",
    "risk_level",
    "max_risk_score",
    "count_transactions",
    "countries_detected",
    "merchant_rubro_proxy",
    "merchant_rubro_values",
    "top_merchant_rubro_proxy",
    "window_start",
    "window_end",
    "representative_transaction_id",
    "child_alert_ids",
    "child_transaction_ids",
    "status",
    "created_at",
]


HOUR_GROUP_RULES = {
    "RULE_VELOCITY_CARD_HOUR",
    "RULE_ATM_OR_CASH_WITHDRAWAL_VELOCITY_HOUR",
    "RULE_CONTACTLESS_NO_PIN_HOUR",
    "RULE_MAGSTRIPE_VELOCITY_HOUR",
}


DAY_GROUP_RULES = {
    "RULE_VELOCITY_CARD_DAY",
    "RULE_ATM_OR_CASH_WITHDRAWAL_DAY",
    "RULE_CONTACTLESS_NO_PIN_DAY",
    "RULE_INTERNET_VELOCITY_DAY",
}


MCC_GROUP_RULES = {
    "RULE_CONTEXTUAL_HIGH_RISK_MCC_WITH_SIGNAL",
    "RULE_STRONG_HIGH_RISK_MCC_AMOUNT",
    "RULE_JEWELRY_MCC_HIGH_AMOUNT",
    "RULE_GAMBLING_MCC",
}


def _normalize_text(value: Any, default: str = "UNKNOWN") -> str:
    if value is None:
        return default
    normalized = str(value).strip()
    if not normalized or normalized.lower() in {"nan", "none", "null"}:
        return default
    return normalized


def _normalize_mcc_value(value: Any) -> str:
    normalized = _normalize_text(value, default="UNKNOWN")
    return normalized.upper()


def _country_codes_from_text(value: Any) -> list[str]:
    if value is None:
        return []

    text = str(value).upper()
    match = re.search(r"COUNTRIES\s*:\s*([^\.\n;]+)", text)
    if not match:
        return []

    countries: list[str] = []
    for token in re.split(r"[^A-Z0-9]+", match.group(1).upper()):
        token = token.strip()
        if len(token) == 2 and token.isalpha() and token not in countries:
            countries.append(token)
    return countries


def _join_unique_values(values: list[Any]) -> str:
    normalized_values: list[str] = []
    for value in values:
        normalized_value = _normalize_text(value)
        if normalized_value and normalized_value not in normalized_values:
            normalized_values.append(normalized_value)
    return "|".join(normalized_values)


def _summary_rule_category(rule_code: str) -> str:
    normalized_rule = _normalize_text(rule_code).upper()
    if normalized_rule.startswith("RULE_DOUBLE_COUNTRY"):
        return "double_country"
    if normalized_rule in HOUR_GROUP_RULES:
        return "hour"
    if normalized_rule in DAY_GROUP_RULES:
        return "day"
    if normalized_rule in MCC_GROUP_RULES:
        return "mcc"
    return "fallback"


def _format_bucket_for_key(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "NONE"
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


def _stable_summary_alert_id(source_run: str, group_key: tuple[Any, ...]) -> str:
    token_parts = [_normalize_text(source_run)]
    token_parts.extend(_format_bucket_for_key(item) for item in group_key)
    digest = hashlib.sha1("||".join(token_parts).encode("utf-8")).hexdigest()[:12]
    return f"{_normalize_text(source_run)}-S-{digest}"


def _empty_summary_df() -> pd.DataFrame:
    return pd.DataFrame(columns=SUMMARY_COLUMNS)


def build_alert_summary_df(alerts_df: pd.DataFrame) -> pd.DataFrame:
    if alerts_df is None or alerts_df.empty:
        return _empty_summary_df()

    df = alerts_df.copy()
    df["transaction_datetime"] = pd.to_datetime(df["transaction_datetime"], errors="coerce", utc=True)
    source_series = df["source_run"] if "source_run" in df.columns else pd.Series(["UNKNOWN"] * len(df), index=df.index)
    customer_series = df["customer_hash"] if "customer_hash" in df.columns else pd.Series(["UNKNOWN"] * len(df), index=df.index)
    rule_series = df["rule_code"] if "rule_code" in df.columns else pd.Series(["UNKNOWN"] * len(df), index=df.index)
    merchant_series = df["merchant_rubro_proxy"] if "merchant_rubro_proxy" in df.columns else pd.Series(["UNKNOWN"] * len(df), index=df.index)

    df["_source_run"] = source_series.map(lambda value: _normalize_text(value))
    df["_customer_hash"] = customer_series.map(lambda value: _normalize_text(value))
    df["_rule_code"] = rule_series.map(lambda value: _normalize_text(value).upper())
    df["_rule_category"] = df["_rule_code"].map(_summary_rule_category)
    df["_date"] = df["transaction_datetime"].dt.floor("D")
    df["_hour"] = df["transaction_datetime"].dt.floor("h")
    df["_merchant_rubro_proxy"] = merchant_series.map(_normalize_mcc_value)

    has_dt = df["transaction_datetime"].notna()
    default_bucket = df["transaction_id"].map(lambda value: f"tx:{_normalize_text(value)}")

    day_bucket = df["_date"].map(lambda value: value.isoformat() if isinstance(value, pd.Timestamp) else None)
    hour_bucket = df["_hour"].map(lambda value: value.isoformat() if isinstance(value, pd.Timestamp) else None)

    df["_bucket"] = default_bucket
    day_like_mask = df["_rule_category"].isin({"double_country", "day", "fallback", "mcc"})
    hour_like_mask = df["_rule_category"].eq("hour")
    df.loc[day_like_mask & has_dt, "_bucket"] = day_bucket[day_like_mask & has_dt]
    df.loc[hour_like_mask & has_dt, "_bucket"] = hour_bucket[hour_like_mask & has_dt]
    df["_bucket"] = df["_bucket"].fillna(default_bucket)

    df["_mcc_key"] = "*"
    mcc_mask = df["_rule_category"].eq("mcc")
    df.loc[mcc_mask, "_mcc_key"] = df.loc[mcc_mask, "_merchant_rubro_proxy"]

    grouped = df.groupby(["_source_run", "_customer_hash", "_rule_code", "_bucket", "_mcc_key"], dropna=False, sort=False)

    summary_rows: list[dict[str, Any]] = []
    created_at = datetime.now(timezone.utc).isoformat()
    for group_key, group in grouped:
        source_run, customer_hash, rule_code, _bucket, mcc_key = group_key
        sorted_group = group.sort_values(["transaction_datetime", "transaction_id"], na_position="last")
        first_row = sorted_group.iloc[0]

        risk_source = sorted_group["risk_score"] if "risk_score" in sorted_group.columns else pd.Series([None] * len(sorted_group), index=sorted_group.index)
        risk_scores = pd.to_numeric(risk_source, errors="coerce")
        max_risk_score = float(risk_scores.max()) if risk_scores.notna().any() else 0.0
        max_row = sorted_group.loc[risk_scores.idxmax()] if risk_scores.notna().any() else first_row

        country_source = sorted_group["country_code"] if "country_code" in sorted_group.columns else pd.Series([None] * len(sorted_group), index=sorted_group.index)
        country_values = [
            _normalize_text(value)
            for value in country_source.tolist()
            if _normalize_text(value) not in {"UNKNOWN"}
        ]
        if rule_code.startswith("RULE_DOUBLE_COUNTRY") and "alert_reason" in sorted_group.columns:
            reason_country_values: list[str] = []
            for reason_value in sorted_group["alert_reason"].tolist():
                for country_code in _country_codes_from_text(reason_value):
                    if country_code not in reason_country_values:
                        reason_country_values.append(country_code)
            country_values = reason_country_values + [country for country in country_values if country not in reason_country_values]
        countries_detected = _join_unique_values(country_values) if country_values else "UNKNOWN"

        child_alert_ids = _join_unique_values(sorted_group["alert_id"].tolist() if "alert_id" in sorted_group.columns else [])
        child_transaction_ids = _join_unique_values(sorted_group["transaction_id"].tolist() if "transaction_id" in sorted_group.columns else [])

        rubro_counter = Counter(
            value
            for value in sorted_group["_merchant_rubro_proxy"].astype(str).tolist()
            if value and value.upper() not in {"UNKNOWN", "NAN", "NONE", "NULL"}
        )
        rubro_values = [code for code, _count in rubro_counter.most_common()]
        top_rubro = rubro_values[0] if rubro_values else "UNKNOWN"

        if _normalize_text(mcc_key, default="*") != "*" and top_rubro == "UNKNOWN":
            top_rubro = _normalize_text(mcc_key).upper()
            rubro_values = [top_rubro]

        summary_rows.append(
            {
                "summary_alert_id": _stable_summary_alert_id(source_run, group_key),
                "source_run": source_run,
                "customer_hash": customer_hash,
                "rule_code": rule_code,
                "rule_name": _normalize_text(max_row.get("rule_name"), default=_normalize_text(first_row.get("rule_name"), default=rule_code)),
                "risk_level": _normalize_text(max_row.get("risk_level"), default=_normalize_text(first_row.get("risk_level"), default="MEDIUM")).upper(),
                "max_risk_score": max_risk_score,
                "count_transactions": int(len(sorted_group)),
                "countries_detected": countries_detected,
                "merchant_rubro_proxy": top_rubro,
                "merchant_rubro_values": "|".join(rubro_values) if rubro_values else None,
                "top_merchant_rubro_proxy": top_rubro,
                "window_start": sorted_group["transaction_datetime"].min().isoformat() if sorted_group["transaction_datetime"].notna().any() else None,
                "window_end": sorted_group["transaction_datetime"].max().isoformat() if sorted_group["transaction_datetime"].notna().any() else None,
                "representative_transaction_id": _normalize_text(first_row.get("transaction_id")),
                "child_alert_ids": child_alert_ids,
                "child_transaction_ids": child_transaction_ids,
                "status": "NEW",
                "created_at": created_at,
            }
        )

    if not summary_rows:
        return _empty_summary_df()

    summary_df = pd.DataFrame(summary_rows)
    for column in SUMMARY_COLUMNS:
        if column not in summary_df.columns:
            summary_df[column] = None
    return summary_df[SUMMARY_COLUMNS]


def generate_alerts_from_preprocessed_csv(
    csv_path: str,
    output_dir: str | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source_path = Path(csv_path)
    if not source_path.exists():
        raise FileNotFoundError(f"File not found: {source_path}")

    df = pd.read_csv(source_path)
    source_run_label = (config or {}).get("source_run")
    run_token = _extract_run_token(source_run_label or source_path, fallback=_extract_run_token(source_path))
    alerts_df, summary = evaluate_transaction_rules(df, config={**(config or {}), "source_run": run_token})
    summary_df = build_alert_summary_df(alerts_df)

    target_dir = Path(output_dir) if output_dir else source_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    alerts_path = target_dir / f"alerts_run_{run_token}.csv"
    summary_path = target_dir / f"alerts_summary_run_{run_token}.csv"
    report_path = target_dir / f"rules_report_run_{run_token}.md"

    alerts_df.to_csv(alerts_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    validation = validate_alerts(alerts_path)
    detailed_rows = int(len(alerts_df))
    double_country_mask = alerts_df["rule_code"].astype(str).str.startswith("RULE_DOUBLE_COUNTRY") if not alerts_df.empty else pd.Series(dtype=bool)
    double_country_detailed_rows = int(double_country_mask.sum()) if not alerts_df.empty else 0
    grouped_rows = int(len(summary_df))
    double_country_grouped_rows = int(summary_df["rule_code"].astype(str).str.startswith("RULE_DOUBLE_COUNTRY").sum()) if not summary_df.empty else 0
    reduction_rows = double_country_detailed_rows - double_country_grouped_rows
    reduction_pct = round((reduction_rows / double_country_detailed_rows) * 100, 2) if double_country_detailed_rows else 0.0

    double_country_card_present_alerts = int((alerts_df["rule_code"].astype(str) == DOUBLE_COUNTRY_CARD_PRESENT_RULE).sum()) if not alerts_df.empty else 0
    double_country_card_absent_alerts = int((alerts_df["rule_code"].astype(str) == DOUBLE_COUNTRY_CARD_ABSENT_RULE).sum()) if not alerts_df.empty else 0
    double_country_deprecated_alerts = int((alerts_df["rule_code"].astype(str) == DOUBLE_COUNTRY_RULE).sum()) if not alerts_df.empty else 0

    def _country_distribution(rule_code: str) -> dict[str, int]:
        if alerts_df.empty:
            return {}
        subset = alerts_df.loc[alerts_df["rule_code"].astype(str) == rule_code]
        return subset["country_code"].fillna("UNKNOWN").astype(str).value_counts().to_dict()

    def _pem_distribution(rule_code: str) -> dict[str, int]:
        if alerts_df.empty:
            return {}
        subset = alerts_df.loc[alerts_df["rule_code"].astype(str) == rule_code]
        return subset["pos_entry_mode"].fillna("UNKNOWN").astype(str).value_counts().to_dict()

    double_country_country_distribution = {
        DOUBLE_COUNTRY_CARD_PRESENT_RULE: _country_distribution(DOUBLE_COUNTRY_CARD_PRESENT_RULE),
        DOUBLE_COUNTRY_CARD_ABSENT_RULE: _country_distribution(DOUBLE_COUNTRY_CARD_ABSENT_RULE),
    }
    double_country_pem_distribution = {
        DOUBLE_COUNTRY_CARD_PRESENT_RULE: _pem_distribution(DOUBLE_COUNTRY_CARD_PRESENT_RULE),
        DOUBLE_COUNTRY_CARD_ABSENT_RULE: _pem_distribution(DOUBLE_COUNTRY_CARD_ABSENT_RULE),
    }

    double_country_excluded_pem10 = 0
    double_country_tnp_discarded_without_additional_signal = 0
    if not alerts_df.empty and "pos_entry_mode" in alerts_df.columns:
        double_country_excluded_pem10 = 0

    summary.update(
        {
            "grouped_alerts_generated": grouped_rows,
            "detailed_alerts_removed_by_grouping": reduction_rows,
            "grouping_reduction_pct": reduction_pct,
            "double_country_detailed_alerts": double_country_detailed_rows,
            "double_country_detailed_alerts_before_grouping": double_country_detailed_rows,
            "double_country_grouped_summary_cases": double_country_grouped_rows,
            "double_country_excluded_pem10": double_country_excluded_pem10,
            "double_country_tnp_discarded_without_additional_signal": double_country_tnp_discarded_without_additional_signal,
            "double_country_card_present_alerts": double_country_card_present_alerts,
            "double_country_card_absent_alerts": double_country_card_absent_alerts,
            "double_country_deprecated_alerts": double_country_deprecated_alerts,
            "double_country_country_distribution": double_country_country_distribution,
            "double_country_pem_distribution": double_country_pem_distribution,
            "grouped_alerts_by_rule_code": summary_df["rule_code"].value_counts().to_dict() if not summary_df.empty else {},
        }
    )
    report_path.write_text(_build_rules_report_markdown(str(source_path), alerts_df, summary, validation), encoding="utf-8")

    return {
        "alerts_path": str(alerts_path),
        "summary_path": str(summary_path),
        "report_path": str(report_path),
        "summary": summary,
        "validation": validation,
        "summary_df": summary_df,
    }

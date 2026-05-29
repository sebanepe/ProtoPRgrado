from __future__ import annotations

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


def build_alert_summary_df(alerts_df: pd.DataFrame) -> pd.DataFrame:
    if alerts_df is None or alerts_df.empty:
        return pd.DataFrame(
            columns=[
                "summary_alert_id",
                "source_run",
                "customer_hash",
                "rule_code",
                "rule_name",
                "risk_level",
                "max_risk_score",
                "count_transactions",
                "countries_detected",
                "window_start",
                "window_end",
                "representative_transaction_id",
                "status",
                "created_at",
            ]
        )

    df = alerts_df.copy()
    df["transaction_datetime"] = pd.to_datetime(df["transaction_datetime"], errors="coerce", utc=True)
    df["_date"] = df["transaction_datetime"].dt.date

    double_country_mask = df["rule_code"].astype(str).str.startswith("RULE_DOUBLE_COUNTRY")
    double_country = df.loc[double_country_mask].copy()
    others = df.loc[df["rule_code"].astype(str) != "RULE_DOUBLE_COUNTRY_SAME_DAY"].copy()

    summary_rows: list[dict[str, Any]] = []
    if not double_country.empty:
        grouped = double_country.groupby(["source_run", "customer_hash", "rule_code", "_date"], dropna=False, sort=False)
        for index, ((source_run, customer_hash, rule_code, _date), group) in enumerate(grouped, start=1):
            group = group.sort_values("transaction_datetime")
            first_row = group.iloc[0]
            summary_rows.append(
                {
                    "summary_alert_id": f"{source_run}-S-{index:06d}",
                    "source_run": source_run,
                    "customer_hash": customer_hash,
                    "rule_code": rule_code,
                    "rule_name": first_row["rule_name"],
                    "risk_level": first_row["risk_level"],
                    "max_risk_score": float(group["risk_score"].max()),
                    "count_transactions": int(len(group)),
                    "countries_detected": "|".join(sorted(set(group["country_code"].fillna("UNKNOWN").astype(str).tolist()))),
                    "window_start": group["transaction_datetime"].min().isoformat() if group["transaction_datetime"].notna().any() else None,
                    "window_end": group["transaction_datetime"].max().isoformat() if group["transaction_datetime"].notna().any() else None,
                    "representative_transaction_id": first_row["transaction_id"],
                    "status": "NEW",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
            )

    if not summary_rows and not others.empty:
        # keep a non-empty summary path only when there are no double-country alerts
        pass

    summary_df = pd.DataFrame(summary_rows)
    if summary_df.empty:
        summary_df = pd.DataFrame(
            columns=[
                "summary_alert_id",
                "source_run",
                "customer_hash",
                "rule_code",
                "rule_name",
                "risk_level",
                "max_risk_score",
                "count_transactions",
                "countries_detected",
                "window_start",
                "window_end",
                "representative_transaction_id",
                "status",
                "created_at",
            ]
        )
    return summary_df


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
    reduction_rows = double_country_detailed_rows - grouped_rows
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
            "double_country_grouped_summary_cases": grouped_rows,
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

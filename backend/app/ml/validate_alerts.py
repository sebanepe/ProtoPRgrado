from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import re

import pandas as pd


READY_VERDICT = "ALERTS_READY_FOR_REVIEW"
NOT_READY_VERDICT = "ALERTS_NOT_READY"

REQUIRED_COLUMNS = {
    "alert_id",
    "source_run",
    "transaction_id",
    "customer_hash",
    "transaction_datetime",
    "amount",
    "country_code",
    "pos_entry_mode",
    "has_pinblock",
    "merchant_rubro_proxy",
    "rule_code",
    "rule_name",
    "risk_level",
    "risk_score",
    "alert_reason",
    "triggered_rules",
    "status",
    "created_at",
}

FORBIDDEN_COLUMNS = {
    "is_fraud",
    "confirmed_fraud",
    "description",
    "description_1",
    "response_description",
    "merchant_rubro_description",
}

INTERNET_RULE_CODES = {"RULE_INTERNET_VELOCITY_DAY"}
# Accept both legacy and new split double-country rule codes (card-present / card-absent)
DOUBLE_COUNTRY_RULE_PREFIX = "RULE_DOUBLE_COUNTRY"
DOUBLE_COUNTRY_RULE_CODES = {
    "RULE_DOUBLE_COUNTRY_SAME_DAY",
    "RULE_DOUBLE_COUNTRY_CARD_PRESENT_SAME_DAY",
    "RULE_DOUBLE_COUNTRY_CARD_ABSENT_CONTEXTUAL",
}

DOUBLE_COUNTRY_ABSENT_RULE = "RULE_DOUBLE_COUNTRY_CARD_ABSENT_CONTEXTUAL"
DOUBLE_COUNTRY_CONTEXTUAL_SIGNALS = {
    "countries_gt_2",
    "txs_day_gt_10",
    "txs_hour_gt_3",
    "mcc_strong_risk",
    "mcc_contextual",
    "amount_ge_contextual_threshold",
    "bo_foreign_with_risk_or_amount",
}


@dataclass
class ValidationResult:
    verdict: str
    rows: int
    columns: list[str]
    missing_required_columns: list[str]
    forbidden_columns_present: list[str]
    duplicate_alert_rows: int
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalized_column_key(value: str) -> str:
    if not isinstance(value, str):
        value = str(value)
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
    return value


def _split_triggered_rules(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    parts = [part.strip() for part in str(value).split("|")]
    return [part for part in parts if part]


def validate(csv_path: str | Path) -> dict[str, Any]:
    path = Path(csv_path)
    if not path.exists():
        return ValidationResult(
            verdict=NOT_READY_VERDICT,
            rows=0,
            columns=[],
            missing_required_columns=sorted(REQUIRED_COLUMNS),
            forbidden_columns_present=[],
            duplicate_alert_rows=0,
            notes=[f"File not found: {path}"],
        ).to_dict()

    df = pd.read_csv(path)
    columns = list(df.columns)
    present = set(columns)
    missing_required = sorted(REQUIRED_COLUMNS - present)
    forbidden_present = sorted([c for c in columns if _normalized_column_key(c) in FORBIDDEN_COLUMNS])

    notes: list[str] = []
    duplicate_alert_rows = 0
    if not df.empty and {"transaction_id", "rule_code", "source_run"}.issubset(present):
        dup_mask = df.duplicated(subset=["transaction_id", "rule_code", "source_run"], keep=False)
        duplicate_alert_rows = int(dup_mask.sum())
        if duplicate_alert_rows > 0:
            notes.append("Duplicated alerts found for transaction_id + rule_code + source_run.")

    if "status" in present and not df.empty:
        status_values = set(df["status"].astype(str).str.strip().str.upper().tolist())
        if status_values != {"NEW"}:
            notes.append("status must be NEW for every generated alert.")

    if "is_fraud" in present:
        notes.append("is_fraud must not be present in alerts output.")
    if "confirmed_fraud" in present:
        notes.append("confirmed_fraud must not be present in alerts output.")

    if any(_normalized_column_key(c) in {"description", "description_1", "response_description", "merchant_rubro_description"} for c in columns):
        notes.append("Forbidden description columns must not be present in alerts output.")

    if "rule_code" in present and "pos_entry_mode" in present:
        internet_mask = df["rule_code"].astype(str).isin(INTERNET_RULE_CODES)
        if internet_mask.any():
            bad_internet = df.loc[internet_mask & (df["pos_entry_mode"].astype(str).str.strip() == "10")]
            if not bad_internet.empty:
                notes.append("PEM 10 must not appear in internet velocity alerts.")

    if "rule_code" in present and "country_code" in present:
        bad_country = df.loc[(df["rule_code"].astype(str).str.startswith(DOUBLE_COUNTRY_RULE_PREFIX)) & (df["country_code"].astype(str).str.strip().str.upper() == "UNKNOWN")]
        if not bad_country.empty:
            notes.append("country_code UNKNOWN must not appear in double-country alerts.")

    # Ensure PEM 10 is not used with any automatic double-country rules
    if "rule_code" in present and "pos_entry_mode" in present:
        bad_pem10 = df.loc[(df["rule_code"].astype(str).str.startswith(DOUBLE_COUNTRY_RULE_PREFIX)) & (df["pos_entry_mode"].astype(str).str.strip() == "10")]
        if not bad_pem10.empty:
            notes.append("PEM 10 must not appear in double-country alerts.")

    if "rule_code" in present and "triggered_rules" in present:
        triggered_parts = df["triggered_rules"].apply(_split_triggered_rules)
        prefix_mismatch = df.loc[
            df["triggered_rules"].notna()
            & (df["triggered_rules"].astype(str).str.strip() != "")
            & (
                df["rule_code"].astype(str)
                != triggered_parts.apply(lambda parts: parts[0] if parts else "")
            )
        ]
        if not prefix_mismatch.empty:
            notes.append("triggered_rules must start with the rule_code for every alert.")

        absent_mask = df["rule_code"].astype(str) == DOUBLE_COUNTRY_ABSENT_RULE
        if absent_mask.any():
            absent_parts = triggered_parts.loc[absent_mask]
            contextual_mask = absent_parts.apply(
                lambda parts: len(parts) > 1 and any(signal in DOUBLE_COUNTRY_CONTEXTUAL_SIGNALS for signal in parts[1:])
            )
            missing_context = absent_mask.copy()
            missing_context.loc[absent_mask] = ~contextual_mask.to_numpy()

            fallback_mask = missing_context & df["alert_reason"].astype(str).str.contains(r"Signals:", case=False, na=False)
            truly_missing = missing_context & ~fallback_mask
            if truly_missing.any():
                notes.append("Card-absent double-country alerts must include contextual signals in triggered_rules.")
            if fallback_mask.any():
                notes.append("Card-absent double-country alerts used alert_reason as a temporary signal fallback.")

    verdict = READY_VERDICT if not missing_required and not forbidden_present and duplicate_alert_rows == 0 and not notes else NOT_READY_VERDICT

    return ValidationResult(
        verdict=verdict,
        rows=int(len(df)),
        columns=columns,
        missing_required_columns=missing_required,
        forbidden_columns_present=forbidden_present,
        duplicate_alert_rows=duplicate_alert_rows,
        notes=notes,
    ).to_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate generated alerts CSV for Phase B")
    parser.add_argument("--data_path", required=True, help="Path to alerts_run_N.csv")
    args = parser.parse_args()

    report = validate(args.data_path)
    print(report["verdict"])
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()

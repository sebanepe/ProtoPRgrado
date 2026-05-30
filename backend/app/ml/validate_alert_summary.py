from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import pandas as pd


READY_VERDICT = "ALERT_SUMMARY_READY"
NOT_READY_VERDICT = "ALERT_SUMMARY_NOT_READY"

REQUIRED_COLUMNS = {
    "summary_alert_id",
    "source_run",
    "customer_hash",
    "rule_code",
    "rule_name",
    "risk_level",
    "max_risk_score",
    "count_transactions",
    "countries_detected",
    "child_alert_ids",
    "child_transaction_ids",
    "window_start",
    "window_end",
    "representative_transaction_id",
    "status",
    "created_at",
}

FORBIDDEN_COLUMNS = {"is_fraud", "confirmed_fraud"}


@dataclass
class ValidationResult:
    verdict: str
    rows: int
    columns: list[str]
    missing_required_columns: list[str]
    forbidden_columns_present: list[str]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate(csv_path: str | Path) -> dict[str, Any]:
    path = Path(csv_path)
    if not path.exists():
        return ValidationResult(
            verdict=NOT_READY_VERDICT,
            rows=0,
            columns=[],
            missing_required_columns=sorted(REQUIRED_COLUMNS),
            forbidden_columns_present=[],
            notes=[f"File not found: {path}"],
        ).to_dict()

    df = pd.read_csv(path)
    columns = list(df.columns)
    present = set(columns)
    missing_required = sorted(REQUIRED_COLUMNS - present)
    forbidden_present = sorted([c for c in columns if c in FORBIDDEN_COLUMNS])

    notes: list[str] = []
    if "status" in present and not df.empty:
        if set(df["status"].astype(str).str.strip().str.upper().tolist()) != {"NEW"}:
            notes.append("status must be NEW for every summary alert.")

    if {"window_start", "window_end"}.issubset(present) and not df.empty:
        starts = pd.to_datetime(df["window_start"], errors="coerce", utc=True)
        ends = pd.to_datetime(df["window_end"], errors="coerce", utc=True)
        if (starts > ends).any():
            notes.append("window_start must be less than or equal to window_end.")

    if "count_transactions" in present and not df.empty:
        if (pd.to_numeric(df["count_transactions"], errors="coerce") <= 0).any():
            notes.append("count_transactions must be greater than zero.")

    if "is_fraud" in present:
        notes.append("is_fraud must not be present in summary output.")
    if "confirmed_fraud" in present:
        notes.append("confirmed_fraud must not be present in summary output.")

    verdict = READY_VERDICT if not missing_required and not forbidden_present and not notes else NOT_READY_VERDICT
    return ValidationResult(
        verdict=verdict,
        rows=int(len(df)),
        columns=columns,
        missing_required_columns=missing_required,
        forbidden_columns_present=forbidden_present,
        notes=notes,
    ).to_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate generated grouped alerts CSV for Phase B")
    parser.add_argument("--data_path", required=True, help="Path to alerts_summary_run_N.csv")
    args = parser.parse_args()

    report = validate(args.data_path)
    print(report["verdict"])
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
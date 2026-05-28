from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import pandas as pd

READY_VERDICT = "CLEANED_DATASET_READY_FOR_RULES"
NOT_READY_VERDICT = "CLEANED_DATASET_NOT_READY"

REQUIRED_COLUMNS = {
    "transaction_id",
    "amount",
    "transaction_datetime",
}

PREFERRED_COLUMNS = {
    "customer_hash",
    "merchant_hash",
    "country_code",
    "pos_entry_mode",
    "has_pinblock",
    "card_presence_type",
    "card_brand",
    "merchant_rubro_proxy",
    "mcc",
    "channel",
    "currency_code",
    "transaction_type",
    "transaction_category",
}

FORBIDDEN_COLUMNS = {
    "is_fraud",
    "is_fraud_proxy",
    "confirmed_fraud",
    "analyst_label",
    "label_source",
    "fraud_label_reason",
    "risk_signal_reason",
    "behavioral_risk_score",
    "independent_rule_groups",
    "amount_scaled",
    "card_product_proxy",
    "response_high_risk",
    "normalized_response_code",
    "response_code_reason",
}

FORBIDDEN_PREFIXES = ("feature_",)


@dataclass
class ValidationResult:
    verdict: str
    rows: int
    columns: list[str]
    missing_required_columns: list[str]
    forbidden_columns_present: list[str]
    preferred_columns_present: list[str]
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
            preferred_columns_present=[],
            notes=[f"File not found: {path}"],
        ).to_dict()

    df = pd.read_csv(path)
    columns = list(df.columns)
    present = set(columns)
    missing_required = sorted(REQUIRED_COLUMNS - present)
    forbidden_present = sorted(
        [c for c in columns if c in FORBIDDEN_COLUMNS or c.startswith(FORBIDDEN_PREFIXES)]
    )
    preferred_present = sorted(PREFERRED_COLUMNS & present)

    notes: list[str] = []
    if missing_required:
        notes.append("Missing required base columns for rule evaluation.")
    if forbidden_present:
        notes.append("Training or label columns are still present.")
    if df.empty:
        notes.append("The cleaned dataset has no rows.")

    verdict = READY_VERDICT if not missing_required and not forbidden_present and not df.empty else NOT_READY_VERDICT

    if preferred_present:
        notes.append("Preferred rule columns are available for later phases.")

    return ValidationResult(
        verdict=verdict,
        rows=int(len(df)),
        columns=columns,
        missing_required_columns=missing_required,
        forbidden_columns_present=forbidden_present,
        preferred_columns_present=preferred_present,
        notes=notes,
    ).to_dict()

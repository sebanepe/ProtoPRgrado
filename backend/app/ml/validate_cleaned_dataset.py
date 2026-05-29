from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import re

import pandas as pd

from backend.app.ml.preprocessing import MERCANT_RUBRO_SOURCE_COLUMNS, _normalize_column_key

READY_VERDICT = "CLEANED_DATASET_READY_FOR_RULES"
NOT_READY_VERDICT = "CLEANED_DATASET_NOT_READY"

REQUIRED_COLUMNS = {
    "transaction_id",
    "amount",
    "transaction_datetime",
    "merchant_rubro_proxy",
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
    "merchant_rubro_description",
    "description",
    "description_1",
    "response_description",
    "mcc_code",
    "codigo_mcc",
    "mcc",
    "rubro",
    "codigo_rubro",
    "merchant_category_code",
    "categoria_comercio",
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


def _normalized_source_columns(df: pd.DataFrame) -> set[str]:
    return {_normalize_column_key(c) for c in df.columns}


def validate(csv_path: str | Path, source_path: str | Path | None = None) -> dict[str, Any]:
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

    rubro_unknown_count = 0
    rubro_valid_4digit_count = 0
    source_mcc_present = False
    source_mcc_columns: list[str] = []
    if "merchant_rubro_proxy" in present:
        rubro_series = df["merchant_rubro_proxy"].fillna("UNKNOWN").astype(str).str.strip().replace({"": "UNKNOWN"})
        rubro_unknown_count = int((rubro_series == "UNKNOWN").sum())
        rubro_valid_4digit_count = int(rubro_series.str.fullmatch(r"\d{4}").fillna(False).sum())

    if source_path is not None:
        source_file = Path(source_path)
        if source_file.exists():
            source_df = pd.read_csv(source_file)
            source_columns = _normalized_source_columns(source_df)
            source_mcc_present = bool(source_columns & MERCANT_RUBRO_SOURCE_COLUMNS)
            source_mcc_columns = sorted(list(source_columns & MERCANT_RUBRO_SOURCE_COLUMNS))

    notes: list[str] = []
    if missing_required:
        notes.append("Missing required base columns for rule evaluation.")
    if forbidden_present:
        notes.append("Training or label columns are still present.")
    if df.empty:
        notes.append("The cleaned dataset has no rows.")
    if "merchant_rubro_proxy" not in present:
        notes.append("merchant_rubro_proxy is missing from the cleaned dataset.")
    else:
        notes.append(f"merchant_rubro_proxy_unknown_count={rubro_unknown_count}")
        notes.append(f"merchant_rubro_proxy_valid_4digit_count={rubro_valid_4digit_count}")
        if source_mcc_present and rubro_unknown_count == len(df) and len(df) > 0:
            notes.append("El archivo de origen contiene MCC_CODE, pero no se preservó ningún código MCC válido en merchant_rubro_proxy.")

    verdict = READY_VERDICT if not missing_required and not forbidden_present and not df.empty else NOT_READY_VERDICT
    if source_mcc_present and "merchant_rubro_proxy" in present and rubro_unknown_count == len(df) and len(df) > 0:
        verdict = NOT_READY_VERDICT

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

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


READY = "HUMAN_SUPERVISED_DATASET_READY"
INSUFFICIENT_CLASSES = "HUMAN_SUPERVISED_DATASET_INSUFFICIENT_CLASSES"
EMPTY = "HUMAN_SUPERVISED_DATASET_EMPTY"
INVALID = "HUMAN_SUPERVISED_DATASET_INVALID"
FILE_NOT_FOUND = "FILE_NOT_FOUND"
SENSITIVE_COLUMNS = {"PAN_TARJETA", "TARJETA", "pan_card", "raw_card"}


def validate_human_supervised_dataset(data_path: str | Path) -> dict[str, Any]:
    path = Path(data_path)
    errors: list[str] = []
    warnings: list[str] = []
    if not path.exists():
        return {"verdict": FILE_NOT_FOUND, "status": "EMPTY", "errors": [f"File not found: {path}"], "warnings": []}

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        return {"verdict": INVALID, "status": "INVALID", "errors": [str(exc)], "warnings": []}

    if df.empty:
        return {"verdict": EMPTY, "status": "EMPTY", "row_count": 0, "errors": [], "warnings": []}

    columns = set(df.columns)
    if "target_human_label" not in columns:
        errors.append("Missing target_human_label")
        target_series = pd.Series(dtype="float")
    else:
        target_series = pd.to_numeric(df["target_human_label"], errors="coerce")
        targets = set(target_series.dropna().astype(int).unique().tolist())
        if not targets.issubset({0, 1}):
            errors.append("target_human_label must contain only 0 and 1")
        if target_series.isna().any():
            errors.append("target_human_label contains null values")

    for forbidden in ["is_fraud", "confirmed_fraud"]:
        if forbidden in columns:
            errors.append(f"Forbidden column present: {forbidden}")
    for sensitive in SENSITIVE_COLUMNS:
        if sensitive in columns:
            errors.append(f"Sensitive column present: {sensitive}")

    if "anomaly_flag" in columns and "target_human_label" not in columns:
        errors.append("anomaly_flag appears to be used as target")
    if "rule_code" not in columns:
        warnings.append("rule_code is not present as a feature")
    if "summary_alert_id" not in columns or df.get("summary_alert_id", pd.Series(dtype=object)).isna().any():
        errors.append("All rows must have summary_alert_id")
    if "source_run" not in columns or df.get("source_run", pd.Series(dtype=object)).isna().any():
        errors.append("All rows must have source_run")
    if "human_review_status" in columns:
        statuses = set(df["human_review_status"].dropna().astype(str).str.upper().unique().tolist())
        blocked = statuses.intersection({"NEW", "IN_REVIEW", "FALSE_POSITIVE"})
        if blocked:
            errors.append(f"Excluded human_review_status values present: {sorted(blocked)}")
    if "target_label_source" not in columns:
        errors.append("Missing target_label_source")
    elif set(df["target_label_source"].dropna().unique().tolist()) != {"HUMAN_REVIEW"}:
        errors.append("target_label_source must be HUMAN_REVIEW")
    if "target_label_meaning" not in columns:
        errors.append("Missing target_label_meaning")
    elif "target_human_label" in columns:
        check_df = df.assign(_target_numeric=target_series)
        invalid_meaning = df[
            ((check_df["_target_numeric"] == 1) & (check_df["target_label_meaning"] != "CONFIRMED_FRAUD"))
            | ((check_df["_target_numeric"] == 0) & (check_df["target_label_meaning"] != "DISMISSED"))
        ]
        if not invalid_meaning.empty:
            errors.append("target_label_meaning does not match target_human_label")

    if errors:
        verdict = INVALID
    else:
        classes = set(target_series.astype(int).unique().tolist())
        verdict = READY if classes == {0, 1} else INSUFFICIENT_CLASSES

    return {
        "verdict": verdict,
        "status": "VALID" if verdict == READY else "NOT_READY" if not errors else "INVALID",
        "row_count": len(df),
        "positive_count": int((target_series == 1).sum()) if "target_human_label" in columns else 0,
        "negative_count": int((target_series == 0).sum()) if "target_human_label" in columns else 0,
        "errors": errors,
        "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a human supervised alert dataset")
    parser.add_argument("--data-path", required=True)
    args = parser.parse_args()
    result = validate_human_supervised_dataset(args.data_path)
    print(result["verdict"])
    if result.get("errors"):
        for error in result["errors"]:
            print(f"ERROR: {error}")
    if result.get("warnings"):
        for warning in result["warnings"]:
            print(f"WARNING: {warning}")


if __name__ == "__main__":
    main()

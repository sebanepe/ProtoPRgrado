from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


READY = "HUMAN_SUPERVISED_TRAINING_READY"
INVALID = "HUMAN_SUPERVISED_TRAINING_INVALID"
SENSITIVE_COLUMNS = {"PAN_TARJETA", "TARJETA", "pan_card", "raw_card"}
EVALUATION_RESULTS = {"TRUE_POSITIVE", "TRUE_NEGATIVE", "FALSE_POSITIVE", "FALSE_NEGATIVE"}


def validate_human_supervised_training(metadata_file: str | Path, predictions_file: str | Path) -> dict[str, Any]:
    metadata_path = Path(metadata_file)
    predictions_path = Path(predictions_file)
    errors: list[str] = []

    if not metadata_path.exists():
        errors.append(f"Metadata file not found: {metadata_path}")
        metadata: dict[str, Any] = {}
    else:
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception as exc:
            metadata = {}
            errors.append(f"Invalid metadata JSON: {exc}")

    if not predictions_path.exists():
        errors.append(f"Predictions file not found: {predictions_path}")
        df = pd.DataFrame()
    else:
        try:
            df = pd.read_csv(predictions_path)
        except Exception as exc:
            df = pd.DataFrame()
            errors.append(f"Invalid predictions CSV: {exc}")

    metrics = metadata.get("metrics") if isinstance(metadata, dict) else None
    if not isinstance(metrics, dict) or not metrics:
        errors.append("Missing metrics")
    elif "confusion_matrix" not in metrics:
        errors.append("Missing confusion_matrix")

    columns = set(df.columns)
    for forbidden in {"is_fraud", "confirmed_fraud"}:
        if forbidden in columns:
            errors.append(f"Forbidden column present: {forbidden}")
    for sensitive in SENSITIVE_COLUMNS:
        if sensitive in columns:
            errors.append(f"Sensitive column present: {sensitive}")

    if df.empty:
        errors.append("Predictions file is empty")
    else:
        for column in ["y_true", "y_pred", "evaluation_result"]:
            if column not in columns:
                errors.append(f"Missing predictions column: {column}")
        if "y_true" in columns and set(pd.to_numeric(df["y_true"], errors="coerce").dropna().astype(int).unique()) != {0, 1}:
            errors.append("y_true must contain 0 and 1")
        if "y_pred" in columns and not set(pd.to_numeric(df["y_pred"], errors="coerce").dropna().astype(int).unique()).issubset({0, 1}):
            errors.append("y_pred must contain only 0 and 1")
        if "evaluation_result" in columns:
            values = set(df["evaluation_result"].dropna().astype(str).unique())
            if not values.issubset(EVALUATION_RESULTS):
                errors.append("evaluation_result contains invalid values")

    return {
        "verdict": READY if not errors else INVALID,
        "status": "VALID" if not errors else "INVALID",
        "metadata_file": str(metadata_path),
        "predictions_file": str(predictions_path),
        "errors": errors,
        "row_count": len(df),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate human supervised training artifacts")
    parser.add_argument("--metadata-file", required=True)
    parser.add_argument("--predictions-file", required=True)
    args = parser.parse_args()
    result = validate_human_supervised_training(args.metadata_file, args.predictions_file)
    print(result["verdict"])
    for error in result.get("errors", []):
        print(f"ERROR: {error}")


if __name__ == "__main__":
    main()

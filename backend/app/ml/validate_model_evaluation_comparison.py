from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

FORBIDDEN_COLUMNS = {"is_fraud", "confirmed_fraud", "PAN_TARJETA", "TARJETA", "pan_card", "raw_card"}
READY = "MODEL_EVALUATION_COMPARISON_READY"
INVALID = "MODEL_EVALUATION_COMPARISON_INVALID"
PARTIAL = "MODEL_EVALUATION_COMPARISON_PARTIAL_READY"


def validate_model_evaluation_comparison(alert_file: str | Path, transaction_file: str | Path, metadata_file: str | Path) -> dict:
    issues: list[str] = []
    warnings: list[str] = []
    files = {"alert_file": Path(alert_file), "transaction_file": Path(transaction_file), "metadata_file": Path(metadata_file)}
    for name, path in files.items():
        if not path.exists():
            issues.append(f"{name}_not_found")

    if issues:
        return {"verdict": INVALID, "issues": issues, "warnings": warnings}

    alert_df = pd.read_csv(files["alert_file"])
    tx_df = pd.read_csv(files["transaction_file"])
    metadata = json.loads(files["metadata_file"].read_text(encoding="utf-8"))

    for col in FORBIDDEN_COLUMNS:
        if col in alert_df.columns or col in tx_df.columns:
            issues.append(f"forbidden_column_present:{col}")
    for required in ("source_run", "summary_alert_id"):
        if required not in alert_df.columns:
            issues.append(f"missing_alert_column:{required}")
    if "transaction_id" not in tx_df.columns:
        issues.append("missing_transaction_column:transaction_id")
    for required in ("available_methods", "warnings"):
        if required not in metadata:
            issues.append(f"missing_metadata_field:{required}")

    if any("confirmed_fraud" in str(c).lower() for c in alert_df.columns):
        issues.append("predictions_presented_as_confirmed_fraud")

    missing_methods = set(metadata.get("missing_methods", []))
    partial_allowed = bool(missing_methods) and "autoencoder" in missing_methods
    if issues:
        verdict = PARTIAL if partial_allowed else INVALID
    else:
        verdict = PARTIAL if partial_allowed else READY
    return {"verdict": verdict, "issues": issues, "warnings": warnings}


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate model evaluation comparison outputs")
    parser.add_argument("--alert-file", required=True)
    parser.add_argument("--transaction-file", required=True)
    parser.add_argument("--metadata-file", required=True)
    args = parser.parse_args()
    result = validate_model_evaluation_comparison(args.alert_file, args.transaction_file, args.metadata_file)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["verdict"] in {READY, PARTIAL} else 1


if __name__ == "__main__":
    raise SystemExit(main())

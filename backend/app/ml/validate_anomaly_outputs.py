from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict

import pandas as pd


REQUIRED_SCORE_COLUMNS = [
    "anomaly_run_id",
    "source_run",
    "transaction_id",
    "customer_hash",
    "transaction_datetime",
    "amount",
    "country_code",
    "pos_entry_mode",
    "has_pinblock",
    "merchant_rubro_proxy",
    "anomaly_model_name",
    "anomaly_score",
    "anomaly_rank",
    "anomaly_flag",
    "anomaly_percentile",
    "created_at",
]

FORBIDDEN_LABEL_COLUMNS = {"is_fraud", "confirmed_fraud", "target_is_fraud", "analyst_label"}
FORBIDDEN_RULE_COLUMNS = {"rule_code", "rule_name", "alert_reason", "risk_score", "behavioral_risk_score"}
FORBIDDEN_SENSITIVE_COLUMNS = {"pan_tarjeta", "tarjeta", "pan_card", "masked_card", "authorization_code", "reference_number", "response_description"}


def _load_metadata(metadata_file: str | None) -> Dict[str, Any]:
    if not metadata_file or not os.path.exists(metadata_file):
        return {}
    with open(metadata_file, "r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_anomaly_outputs(
    score_file: str,
    feature_file: str | None = None,
    metadata_file: str | None = None,
    contamination: float | None = None,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {"score_file": score_file, "feature_file": feature_file, "metadata_file": metadata_file, "issues": [], "verdict": "ANOMALY_OUTPUTS_INVALID"}

    if not os.path.exists(score_file):
        report["issues"].append(f"missing_score_file:{score_file}")
        return report

    score_df = pd.read_csv(score_file)
    missing_columns = [column for column in REQUIRED_SCORE_COLUMNS if column not in score_df.columns]
    if missing_columns:
        report["issues"].append(f"missing_required_columns:{missing_columns}")

    if "anomaly_flag" in score_df.columns:
        invalid_flags = sorted(set(score_df["anomaly_flag"].dropna().astype(int).tolist()) - {0, 1})
        if invalid_flags:
            report["issues"].append(f"invalid_anomaly_flag_values:{invalid_flags}")

    for forbidden in sorted(FORBIDDEN_LABEL_COLUMNS | FORBIDDEN_RULE_COLUMNS | FORBIDDEN_SENSITIVE_COLUMNS):
        if forbidden in score_df.columns:
            report["issues"].append(f"forbidden_column_present:{forbidden}")

    if "transaction_id" in score_df.columns and score_df["transaction_id"].duplicated().any():
        report["issues"].append("duplicate_transaction_id_present")

    if "anomaly_score" in score_df.columns and score_df["anomaly_score"].isna().any():
        report["issues"].append("anomaly_score_contains_nan")

    if "anomaly_rank" in score_df.columns:
        ranks = score_df["anomaly_rank"].dropna().astype(int).tolist()
        expected_ranks = list(range(1, len(score_df) + 1))
        if sorted(ranks) != expected_ranks:
            report["issues"].append("anomaly_rank_not_consistent")

    anomaly_count = int(score_df["anomaly_flag"].fillna(0).astype(int).sum()) if "anomaly_flag" in score_df.columns else 0
    anomaly_rate = anomaly_count / len(score_df) if len(score_df) else 0.0
    report["anomaly_count"] = anomaly_count
    report["anomaly_rate"] = anomaly_rate
    if anomaly_count < 1:
        report["issues"].append("no_anomalies_detected")

    metadata = _load_metadata(metadata_file)
    if contamination is None:
        contamination = metadata.get("contamination")
    if contamination is not None:
        allowed_delta = max(0.01, float(contamination) * 0.5)
        if abs(float(anomaly_rate) - float(contamination)) > allowed_delta:
            report["issues"].append("anomaly_rate_not_close_to_contamination")

    if feature_file and os.path.exists(feature_file):
        feature_df = pd.read_csv(feature_file)
        if any(column in feature_df.columns for column in FORBIDDEN_LABEL_COLUMNS | FORBIDDEN_RULE_COLUMNS):
            report["issues"].append("forbidden_columns_present_in_feature_file")

    if metadata:
        model_input_columns = metadata.get("model_input_columns", [])
        if any(column in model_input_columns for column in ["transaction_id", "customer_hash", "merchant_hash", "is_fraud", "confirmed_fraud", "target_is_fraud"]):
            report["issues"].append("forbidden_columns_present_in_model_inputs")

    report["verdict"] = "ANOMALY_OUTPUTS_READY" if not report["issues"] else "ANOMALY_OUTPUTS_INVALID"
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate anomaly detection outputs")
    parser.add_argument("--score-file", required=True)
    parser.add_argument("--feature-file", default=None)
    parser.add_argument("--metadata-file", default=None)
    parser.add_argument("--contamination", type=float, default=None)
    args = parser.parse_args()

    result = validate_anomaly_outputs(args.score_file, args.feature_file, args.metadata_file, args.contamination)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

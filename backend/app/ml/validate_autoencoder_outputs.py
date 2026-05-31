from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Optional

import pandas as pd


REQUIRED_COLUMNS = [
    "source_run",
    "reconstruction_error",
    "autoencoder_anomaly_score",
    "autoencoder_anomaly_flag",
    "anomaly_rank",
]
FORBIDDEN_COLUMNS = {
    "is_fraud",
    "confirmed_fraud",
    "target_is_fraud",
    "pan_tarjeta",
    "tarjeta",
    "pan_card",
    "raw_card",
    "rule_code",
}


def _load_metadata(metadata_file: Optional[str]) -> Dict[str, Any]:
    if not metadata_file:
        return {}
    if not os.path.exists(metadata_file):
        return {}
    try:
        with open(metadata_file, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def validate_autoencoder_outputs(score_file: str, metadata_file: Optional[str] = None) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "score_file": score_file,
        "metadata_file": metadata_file,
        "issues": [],
        "verdict": "AUTOENCODER_OUTPUTS_INVALID",
        "anomaly_count": 0,
        "anomaly_rate": 0.0,
        "expected_contamination": None,
    }
    if not os.path.exists(score_file):
        report["verdict"] = "AUTOENCODER_OUTPUTS_FILE_NOT_FOUND"
        report["issues"].append(f"missing_score_file:{score_file}")
        return report

    score_df = pd.read_csv(score_file)
    if score_df.empty:
        report["verdict"] = "AUTOENCODER_OUTPUTS_EMPTY"
        report["issues"].append("empty_score_file")
        return report

    lower_columns = {column.lower(): column for column in score_df.columns}
    for column in REQUIRED_COLUMNS:
        if column not in score_df.columns:
            report["issues"].append(f"missing_required_column:{column}")

    for forbidden in sorted(FORBIDDEN_COLUMNS):
        if forbidden in lower_columns:
            report["issues"].append(f"forbidden_column_present:{forbidden}")

    if "autoencoder_anomaly_flag" in score_df.columns:
        normalized_flags = pd.to_numeric(score_df["autoencoder_anomaly_flag"], errors="coerce")
        invalid = sorted(set(normalized_flags.dropna().astype(int).tolist()) - {0, 1})
        if invalid or normalized_flags.isna().any():
            report["issues"].append("invalid_autoencoder_anomaly_flag_values")
        anomaly_count = int(normalized_flags.fillna(0).astype(int).sum())
        anomaly_rate = float(anomaly_count / len(score_df)) if len(score_df) else 0.0
        report["anomaly_count"] = anomaly_count
        report["anomaly_rate"] = anomaly_rate

    if "reconstruction_error" in score_df.columns and score_df["reconstruction_error"].isna().any():
        report["issues"].append("reconstruction_error_contains_nan")
    if "autoencoder_anomaly_score" in score_df.columns and score_df["autoencoder_anomaly_score"].isna().any():
        report["issues"].append("autoencoder_anomaly_score_contains_nan")
    if "anomaly_rank" in score_df.columns:
        ranks = pd.to_numeric(score_df["anomaly_rank"], errors="coerce").dropna().astype(int).tolist()
        if sorted(ranks) != list(range(1, len(score_df) + 1)):
            report["issues"].append("anomaly_rank_not_consistent")

    if "transaction_id" not in score_df.columns:
        report["issues"].append("transaction_id_missing_if_available")
    if "customer_hash" not in score_df.columns:
        report["issues"].append("customer_hash_missing_if_available")

    metadata = _load_metadata(metadata_file)
    contamination = metadata.get("contamination")
    report["expected_contamination"] = contamination
    if metadata_file and not os.path.exists(metadata_file):
        report["issues"].append(f"missing_metadata_file:{metadata_file}")
    if metadata:
        feature_columns = [str(column).lower() for column in metadata.get("feature_columns", [])]
        for forbidden in ["anomaly_flag", "rule_code", "is_fraud", "confirmed_fraud"]:
            if forbidden in feature_columns:
                report["issues"].append(f"forbidden_feature_column:{forbidden}")
    if contamination is not None:
        allowed_delta = max(0.02, float(contamination) * 0.75)
        if abs(float(report["anomaly_rate"]) - float(contamination)) > allowed_delta:
            report["issues"].append("anomaly_rate_not_close_to_contamination")

    report["verdict"] = "AUTOENCODER_OUTPUTS_READY" if not report["issues"] else "AUTOENCODER_OUTPUTS_INVALID"
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate PyTorch autoencoder anomaly outputs")
    parser.add_argument("--score-file", required=True)
    parser.add_argument("--metadata-file", default=None)
    args = parser.parse_args()
    print(json.dumps(validate_autoencoder_outputs(args.score_file, args.metadata_file), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

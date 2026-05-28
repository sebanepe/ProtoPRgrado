import argparse
import pandas as pd
import os

FORBIDDEN = set([
    "response_code", "normalized_response_code", "response_high_risk", "response_code_reason",
    "is_fraud_proxy", "behavioral_risk_score", "independent_rule_groups", "label_source", "fraud_label_reason", "risk_signal_reason",
    "transaction_id", "customer_hash", "merchant_hash", "device_id", "reference_number", "authorization_code", "merchant_code", "terminal_code",
    "pan_card", "masked_card", "PAN_TARJETA", "TARJETA",
    "has_pinblock_source",
])

DIST_COLUMNS = [
    "country_code",
    "pos_entry_mode",
    "has_pinblock",
    "card_presence_type",
    "card_brand",
]


def validate(path: str):
    if not os.path.exists(path):
        raise SystemExit(f"Path {path} does not exist")
    df = pd.read_csv(path)
    report = {"path": path}
    report["columns"] = list(df.columns)
    # existence of is_fraud
    report["has_is_fraud"] = "is_fraud" in df.columns
    if report["has_is_fraud"]:
        vc = df["is_fraud"].value_counts(dropna=False).to_dict()
        report["is_fraud_distribution"] = vc
        report["n_classes"] = len([k for k in vc.keys() if pd.notna(k)])
    else:
        report["is_fraud_distribution"] = None
        report["n_classes"] = 0
    # forbidden columns
    forbidden_present = [c for c in df.columns if c in FORBIDDEN]
    report["forbidden_present"] = forbidden_present
    # nulls per column
    nulls = df.isnull().sum().to_dict()
    report["nulls"] = nulls
    # feature frequencies
    freq_high = {}
    freq_full = {}
    for c in df.columns:
        try:
            vc = df[c].value_counts(normalize=True, dropna=False)
            if len(vc) == 0:
                continue
            top = float(vc.iloc[0])
            if top >= 1.0:
                freq_full[c] = top
            if top >= 0.3 and len(vc) > 1:
                freq_high[c] = top
        except Exception:
            continue
    report["freq_high"] = freq_high
    report["freq_full"] = freq_full

    # distributions for key columns
    dist = {}
    for c in DIST_COLUMNS:
        if c in df.columns:
            try:
                dist[c] = df[c].value_counts(dropna=False).to_dict()
            except Exception:
                dist[c] = {}
    report["distributions"] = dist

    # verdict
    if not report["has_is_fraud"] or report["n_classes"] <= 1:
        verdict = "NOT_READY"
    elif forbidden_present:
        verdict = "ONLY_UNSUPERVISED_RECOMMENDED"
    elif freq_full:
        verdict = "NOT_READY"
    else:
        verdict = "READY_FOR_SUPERVISED_TRAINING"
    report["verdict"] = verdict
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    args = parser.parse_args()
    r = validate(args.data_path)
    import json

    print(json.dumps(r, indent=2))

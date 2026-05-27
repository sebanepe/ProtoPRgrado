import os
import pandas as pd
from .proxy_labeling import generate_proxy_fraud_label, normalize_response_code

PROJECT_PROCESSED_DIR = r"C:\Users\seban\Documents\GitHub\Sistema-GIS-La-Paz-Microservicios\ProtoPRgrado\data\processed"


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # standard names
    # amount
    if "amount" not in df.columns:
        for c in ["AMOUNT", "monto", "MONTO"]:
            if c in df.columns:
                df["amount"] = pd.to_numeric(df[c], errors="coerce")
                break
    # transaction_datetime
    if "transaction_datetime" not in df.columns:
        # try common combos
        if "date" in df.columns and "time" in df.columns:
            df["transaction_datetime"] = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str), errors="coerce", utc=True)
        else:
            for c in ["transaction_datetime", "datetime", "fecha", "fecha_hora"]:
                if c in df.columns:
                    df["transaction_datetime"] = pd.to_datetime(df[c], errors="coerce", utc=True)
                    break
    # customer_hash, merchant_hash
    if "customer_hash" not in df.columns:
        for c in ["customer_id", "customer", "cust_hash"]:
            if c in df.columns:
                df["customer_hash"] = df[c].astype(str)
                break
    if "merchant_hash" not in df.columns:
        for c in ["merchant", "merchant_id", "merchant_hash"]:
            if c in df.columns:
                df["merchant_hash"] = df[c].astype(str)
                break
    return df


def build_training_dataset(input_csv: str, out_name: str | None = None):
    df_orig = pd.read_csv(input_csv)
    df = normalize_columns(df_orig)

    # Ensure labels/features exist by applying labeling if missing
    if "is_fraud" not in df.columns or "behavioral_risk_score" not in df.columns:
        try:
            df = generate_proxy_fraud_label(df)
        except Exception:
            # if labeling fails, continue with original df
            pass

    # Persist cleaned dataset (auditable) but drop raw PAN fields
    os.makedirs(PROJECT_PROCESSED_DIR, exist_ok=True)
    cleaned_path = os.path.join(PROJECT_PROCESSED_DIR, (out_name or "cleaned_dataset") + "_cleaned.csv")
    cleaned_df = df.copy()
    # drop raw PAN fields if present
    for raw in ["pan_card", "masked_card", "PAN_TARJETA", "TARJETA"]:
        if raw in cleaned_df.columns:
            cleaned_df = cleaned_df.drop(columns=[raw], errors="ignore")
    cleaned_df.to_csv(cleaned_path, index=False)

    # Define forbidden columns to remove from feature set
    forbidden = [
        "response_code", "normalized_response_code", "response_high_risk", "response_code_reason",
        "is_fraud_proxy", "behavioral_risk_score", "independent_rule_groups", "label_source", "fraud_label_reason", "risk_signal_reason",
        "transaction_id", "customer_hash", "merchant_hash", "device_id", "reference_number", "authorization_code", "merchant_code", "terminal_code",
        "pan_card", "masked_card", "PAN_TARJETA", "TARJETA",
    ]

    # Consider also removing merchant_name and transaction_datetime from features
    optional_drop = ["merchant_name", "transaction_datetime"]

    # build feature set: all columns except forbidden and optional drops
    drop_cols = [c for c in df.columns if c in forbidden or c in optional_drop]
    feature_set = df.drop(columns=drop_cols, errors="ignore")

    # Ensure target present in feature set for training (as integer)
    if "is_fraud" not in feature_set.columns and "is_fraud" in df.columns:
        feature_set["is_fraud"] = df["is_fraud"].astype(int)

    # Write feature set CSV
    feature_set_path = os.path.join(PROJECT_PROCESSED_DIR, (out_name or "feature_set") + "_training.csv")
    feature_set.to_csv(feature_set_path, index=False)

    # Build preprocessing report
    report_path = feature_set_path + ".md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Preprocessing Report\n\n")
        f.write(f"Source file: {input_csv}\n\n")
        f.write("## Rows\n")
        f.write(f"- original_rows: {len(df_orig)}\n")
        f.write(f"- cleaned_rows: {len(cleaned_df)}\n")
        f.write(f"- feature_set_rows: {len(feature_set)}\n\n")

        # rows removed and reason (simple diff)
        removed = len(df_orig) - len(cleaned_df)
        f.write("## Rows removed\n")
        f.write(f"- total_removed: {removed}\n")
        f.write("- note: detailed removal reasons are available in preprocessing logs if any.\n\n")

        f.write("## Distributions\n")
        if "is_fraud" in df.columns:
            f.write("### is_fraud distribution\n")
            vc = df["is_fraud"].value_counts(dropna=False).to_dict()
            for k, v in vc.items():
                f.write(f"- {k}: {v}\n")
        else:
            f.write("- is_fraud: not present\n")

        if "behavioral_risk_score" in df.columns:
            f.write("\n### behavioral_risk_score summary\n")
            desc = df["behavioral_risk_score"].describe()
            for k in ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]:
                if k in desc.index:
                    f.write(f"- {k}: {desc[k]}\n")
        else:
            f.write("- behavioral_risk_score: not present\n")

        if "independent_rule_groups" in df.columns:
            f.write("\n### independent_rule_groups distribution\n")
            vc2 = df["independent_rule_groups"].value_counts().to_dict()
            for k, v in vc2.items():
                f.write(f"- {k}: {v}\n")
        else:
            f.write("- independent_rule_groups: not present\n")

        f.write("\n## Columns removed for sensitivity and leakage\n")
        for c in forbidden:
            f.write(f"- {c}\n")
        f.write("\n## Optional columns removed to avoid overfitting\n")
        for c in optional_drop:
            f.write(f"- {c}\n")

        # warnings
        f.write("\n## Warnings and checks\n")
        # warning if single class
        if "is_fraud" in df.columns and df["is_fraud"].nunique() <= 1:
            f.write("- WARNING: is_fraud contains a single class. Model training will be affected.\n")
        # confirm response_code not used
        if "label_source" in df.columns:
            vals = set(df["label_source"].dropna().unique())
            if "response_code_proxy" in vals:
                f.write("- ALERT: response_code_proxy was used for labeling in this dataset.\n")
            else:
                f.write("- CONFIRMATION: response_code was NOT used to generate is_fraud (label_source contains no response_code_proxy).\n")
        else:
            f.write("- CONFIRMATION: label_source column not present; labeling function should be behavioral-only.\n")

    return feature_set_path, report_path

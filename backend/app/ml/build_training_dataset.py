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
    df = pd.read_csv(input_csv)
    df = normalize_columns(df)
    # generate proxy labels and behavioral features
    df_labeled = generate_proxy_fraud_label(df)

    # remove leakage/sensitive columns from training
    forbidden = [
        "response_code", "codigo_respuesta", "CODIGO_RESPUESTA", "RESPUESTA", "cod_respuesta",
        "normalized_response_code", "response_high_risk", "response_code_reason",
        "is_fraud", "is_fraud_proxy", "label_source", "fraud_label_reason",
        "behavioral_risk_score", "independent_rule_groups",
    ]
    # choose features as all columns except forbidden and identifiers
    drop_cols = [c for c in forbidden if c in df_labeled.columns]
    drop_cols += [c for c in ["transaction_id", "transaction_datetime", "device_id", "customer_hash"] if c in df_labeled.columns]
    training = df_labeled.drop(columns=drop_cols, errors="ignore")

    os.makedirs(PROJECT_PROCESSED_DIR, exist_ok=True)
    out_file = os.path.join(PROJECT_PROCESSED_DIR, out_name or "training_dataset.csv")
    training.to_csv(out_file, index=False)

    # save report
    report_path = out_file + ".md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Training dataset generation report\n\n")
        f.write(f"Source: {input_csv}\n\n")
        f.write("## Label generation\n")
        f.write("`is_fraud` generated as `is_fraud_proxy` via proxy_labeling.generate_proxy_fraud_label`.\n\n")
        f.write("## Columns removed for leakage\n")
        for c in forbidden:
            f.write(f"- {c}\n")
        f.write("\n## Summary\n")
        f.write(f"rows: {len(training)}\n")
        f.write(f"columns: {len(training.columns)}\n")

    return out_file, report_path

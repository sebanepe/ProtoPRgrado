import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from typing import Tuple, Dict
from backend.app.models.models import Transaction


def fetch_transactions_df(db) -> pd.DataFrame:
    rows = db.query(Transaction).all()
    if not rows:
        return pd.DataFrame()
    data = []
    for r in rows:
        data.append(
            {
                "transaction_id": r.transaction_id,
                "amount": float(r.amount),
                "transaction_type": r.transaction_type,
                "channel": r.channel,
                "location": r.location,
                "device_id": r.device_id,
                "customer_hash": r.customer_hash,
                "transaction_datetime": r.transaction_datetime,
                "is_fraud": bool(r.is_fraud),
            }
        )
    df = pd.DataFrame(data)
    return df


def preprocess_dataframe(
    df: pd.DataFrame,
    apply_smote: bool = True,
    smote_random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict]:
    summary = {}
    before = len(df)
    summary["before"] = before

    if df.empty:
        summary.update({"after": 0, "columns_transformed": [], "fraud_ratio": {}})
        return pd.DataFrame(), summary

    # Drop duplicates based on transaction_id
    df = df.drop_duplicates(subset=["transaction_id"])

    # Handle missing values
    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
        df["amount"].fillna(df["amount"].median() if not df["amount"].isna().all() else 0.0, inplace=True)

    for col in ["transaction_type", "channel", "location"]:
        if col in df.columns:
            df[col] = df[col].fillna("unknown").astype(str)

    if "is_fraud" in df.columns:
        df["is_fraud"] = df["is_fraud"].astype(int)
    else:
        df["is_fraud"] = 0

    # Drop rows without transaction_datetime
    if "transaction_datetime" in df.columns:
        df["transaction_datetime"] = pd.to_datetime(df["transaction_datetime"], errors="coerce")
        df = df.dropna(subset=["transaction_datetime"])

    after_clean = len(df)
    summary["after_clean"] = after_clean

    # Encode categorical variables
    cat_cols = [c for c in ["transaction_type", "channel", "location"] if c in df.columns]
    df_encoded = pd.get_dummies(df, columns=cat_cols, dummy_na=False)

    # Normalize numerical features
    num_cols = ["amount"]
    existing_num = [c for c in num_cols if c in df_encoded.columns]
    scaler = None
    if existing_num:
        scaler = StandardScaler()
        df_encoded[["amount_scaled"]] = scaler.fit_transform(df_encoded[["amount"]])

    # Prepare X and y
    y = df_encoded["is_fraud"].astype(int)
    drop_cols = ["transaction_id", "transaction_datetime", "is_fraud", "device_id", "customer_hash", "amount"]
    X = df_encoded.drop(columns=[c for c in drop_cols if c in df_encoded.columns])

    summary["columns_transformed"] = list(X.columns)

    # Apply SMOTE if imbalance present
    fraud_counts = y.value_counts().to_dict()
    summary["fraud_counts_before"] = fraud_counts
    X_res, y_res = X, y
    if apply_smote and len(fraud_counts) == 2:
        maj = max(fraud_counts.values())
        minc = min(fraud_counts.values())
        if minc > 1 and (minc / maj) < 0.5:
            k = min(5, minc - 1)
            if k < 1:
                summary["smote_applied"] = False
            else:
                sm = SMOTE(random_state=smote_random_state, k_neighbors=int(k))
                X_res, y_res = sm.fit_resample(X, y)
                summary["smote_applied"] = True
        else:
            summary["smote_applied"] = False
    else:
        summary["smote_applied"] = False

    # Final proportions
    final_counts = pd.Series(y_res).value_counts().to_dict()
    summary["fraud_counts_after"] = final_counts
    summary["fraud_ratio"] = {
        "before": {k: v for k, v in fraud_counts.items()},
        "after": {k: v for k, v in final_counts.items()},
    }

    # Construct final processed dataframe for saving
    processed = X_res.copy()
    processed["is_fraud"] = y_res

    return processed, summary


def save_processed(df: pd.DataFrame, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

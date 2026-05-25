import os
import unicodedata
import hashlib
import pandas as pd
import numpy as np
from typing import Tuple, Dict
from backend.app.models.models import Transaction
from backend.app.ml import proxy_labeling


def fetch_transactions_df(db, dataset_id: int | None = None) -> pd.DataFrame:
    if dataset_id:
        rows = db.query(Transaction).filter(Transaction.dataset_id == dataset_id).all()
    else:
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


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    def clean_name(s: str) -> str:
        if not isinstance(s, str):
            s = str(s)
        s = s.strip()
        s = s.replace("\u00A0", " ")
        # remove accents
        s = unicodedata.normalize("NFKD", s)
        s = "".join([c for c in s if not unicodedata.combining(c)])
        s = s.strip().lower()
        # replace spaces and repeated spaces with underscore
        s = "_".join(s.split())
        return s

    mapping_candidates = {
        "tipo_transaccion": "transaction_type",
        "categoria transaccion": "transaction_category",
        "codigo de proceso": "process_code",
        "tiene_pinblock": "has_pinblock",
        "numero_referencia": "reference_number",
        "codigo_autorizacion": "authorization_code",
        "codigo_establecimiento": "merchant_code",
        "moneda": "currency_code",
        "codigo_terminal": "terminal_code",
        "canal": "channel",
        "hora": "transaction_datetime",
        "codigo_respuesta": "response_code",
        "tarjeta": "masked_card",
        "establecimiento": "merchant_name",
        "monto": "amount",
        "pos_entry_mode": "pos_entry_mode",
        "sucursal": "branch",
        "pan_tarjeta": "pan_card",
        "pais": "country_code",
    }

    new_cols = {}
    for c in df.columns:
        cn = clean_name(c)
        # direct mapping if present
        if cn in mapping_candidates:
            new_cols[c] = mapping_candidates[cn]
        else:
            new_cols[c] = cn
    return df.rename(columns=new_cols)


def validate_minimum_columns(df: pd.DataFrame):
    missing = []
    if "amount" not in df.columns:
        missing.append("amount")
    if "transaction_datetime" not in df.columns:
        missing.append("transaction_datetime")
    # customer_hash is optional; will be generated later if missing
    # merchant fields are optional; downstream code will synthesize merchant_hash if missing
    if missing:
        raise ValueError("Missing minimum columns: " + ", ".join(missing))


def convert_data_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # amount: handle comma decimal
    if "amount" in df.columns:
        def conv_amount(x):
            if pd.isna(x):
                return np.nan
            s = str(x).strip()
            s = s.replace(' ', '')
            # if contains comma and not dot, assume comma decimal
            if s.count(',') > 0 and s.count('.') == 0:
                s = s.replace(',', '.')
            # remove any non-numeric except dot and minus
            s = ''.join([ch for ch in s if ch.isdigit() or ch in ['.', '-']])
            try:
                return float(s)
            except Exception:
                return np.nan

        df["amount"] = df["amount"].apply(conv_amount)

    # transaction_datetime
    if "transaction_datetime" in df.columns:
            # transaction_datetime parsing: handle mixed timezones by parsing with utc=True
            # then convert to naive UTC timestamps to keep downstream code consistent.
            try:
                ts = pd.to_datetime(df["transaction_datetime"], errors="coerce", utc=True)
                # make naive UTC datetimes
                if hasattr(ts.dt, "tz"):
                    df["transaction_datetime"] = ts.dt.tz_convert("UTC").dt.tz_localize(None)
                else:
                    df["transaction_datetime"] = ts
            except Exception:
                # fallback: try parsing with utc then coerce; if still failing, strip tz suffixes
                try:
                    ts2 = pd.to_datetime(df["transaction_datetime"], errors="coerce", utc=True)
                    if hasattr(ts2.dt, "tz"):
                        df["transaction_datetime"] = ts2.dt.tz_convert("UTC").dt.tz_localize(None)
                    else:
                        df["transaction_datetime"] = ts2
                except Exception:
                    s = df["transaction_datetime"].astype(str).fillna("")
                    s = s.str.replace(r"(\+|-)\d{2}:?\d{2}$|Z$", "", regex=True)
                    df["transaction_datetime"] = pd.to_datetime(s, errors="coerce")

    # pos_entry_mode to numeric
    if "pos_entry_mode" in df.columns:
        df["pos_entry_mode"] = pd.to_numeric(df["pos_entry_mode"], errors="coerce")

    # has_pinblock to 0/1
    if "has_pinblock" in df.columns:
        df["has_pinblock"] = df["has_pinblock"].apply(lambda x: 1 if str(x).strip().lower() in ["1","true","yes","y"] else 0 if pd.notna(x) else np.nan)

    # normalize country and currency codes to string
    for c in ["country_code", "currency_code"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().replace({'nan': None})

    return df


def handle_missing_values(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    df = df.copy()
    report = {"removed_records": 0, "reasons": []}
    initial = len(df)

    # drop rows with invalid amount
    if "amount" in df.columns:
        mask_bad_amount = df["amount"].isna()
        if mask_bad_amount.any():
            # impute missing amounts to 0.0 instead of dropping to preserve records
            report["reasons"].append(f"imputed_amount_missing:{mask_bad_amount.sum()}")
            df.loc[mask_bad_amount, "amount"] = 0.0

    # drop rows with invalid datetime
    if "transaction_datetime" in df.columns:
        mask_bad_dt = df["transaction_datetime"].isna()
        if mask_bad_dt.any():
            report["reasons"].append(f"dropped_datetime_invalid:{mask_bad_dt.sum()}")
            df = df.loc[~mask_bad_dt]

    # categorical imputations
    for col in ["transaction_type", "channel", "location", "merchant_name", "transaction_category"]:
        if col in df.columns:
            df[col] = df[col].fillna("UNKNOWN")

    # merchant_hash
    if "merchant_hash" not in df.columns:
        if "merchant_code" in df.columns:
            df["merchant_hash"] = df["merchant_code"].astype(str).fillna("UNKNOWN_MERCHANT")
        else:
            df["merchant_hash"] = "UNKNOWN_MERCHANT"

    # customer_hash: if missing and masked_card/pan_card available, will be generated later; otherwise drop
    if "customer_hash" not in df.columns or df["customer_hash"].isna().all():
        if not any(c in df.columns for c in ["masked_card", "pan_card"]):
            # cannot derive customer_hash -> drop rows
            report["reasons"].append("dropped_missing_customer_hash")
            df = df.loc[~df.index.isin(df.index)] if df.empty else df

    # has_pinblock impute
    if "has_pinblock" in df.columns:
        df["has_pinblock"] = df["has_pinblock"].fillna(0).astype(int)
    else:
        df["has_pinblock"] = 0

    report["removed_records"] = initial - len(df)
    return df, report


def remove_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    df = df.copy()
    initial = len(df)
    # remove exact duplicates
    df = df.drop_duplicates()
    after_exact = len(df)
    removed_exact = initial - after_exact
    # remove duplicates by transaction_id if exists
    removed_by_tid = 0
    if "transaction_id" in df.columns:
        before_tid = len(df)
        df = df.drop_duplicates(subset=["transaction_id"])
        removed_by_tid = before_tid - len(df)

    report = {"duplicates_removed": removed_exact + removed_by_tid, "removed_exact": removed_exact, "removed_by_transaction_id": removed_by_tid}
    return df, report


def remove_sensitive_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    sensitive = ["pan_tarjeta", "tarjeta", "pan_card", "masked_card", "tarjeta", "numero_cuenta", "documento_identidad"]
    to_drop = [c for c in df.columns if c.lower() in sensitive]
    # also drop common raw names
    for raw in ["pan_card", "masked_card", "pan_tarjeta", "tarjeta"]:
        if raw in df.columns:
            to_drop.append(raw)
    to_drop = list(set(to_drop))
    df = df.drop(columns=[c for c in to_drop if c in df.columns], errors="ignore")
    return df


def generate_anonymized_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    def sha_prefix(val, prefix):
        if pd.isna(val):
            return None
        h = hashlib.sha256(str(val).encode('utf-8')).hexdigest()[:16]
        return f"{prefix}_{h}"

    # transaction_id
    if "transaction_id" not in df.columns:
        df["transaction_id"] = [sha_prefix(str(i) + str(row.get('transaction_datetime','')), 'tx') for i, row in df.iterrows()]

    # customer_hash
    if "customer_hash" not in df.columns:
        if "masked_card" in df.columns:
            df["customer_hash"] = df["masked_card"].apply(lambda x: sha_prefix(x, 'cust') if pd.notna(x) else None)
        elif "pan_card" in df.columns:
            df["customer_hash"] = df["pan_card"].apply(lambda x: sha_prefix(x, 'cust') if pd.notna(x) else None)

    # merchant_hash
    if "merchant_hash" not in df.columns:
        if "merchant_code" in df.columns:
            df["merchant_hash"] = df["merchant_code"].apply(lambda x: sha_prefix(x, 'merch') if pd.notna(x) else 'UNKNOWN_MERCHANT')
        elif "merchant_name" in df.columns:
            df["merchant_hash"] = df["merchant_name"].apply(lambda x: sha_prefix(x, 'merch') if pd.notna(x) else 'UNKNOWN_MERCHANT')
        else:
            df["merchant_hash"] = 'UNKNOWN_MERCHANT'

    # device_id from terminal_code
    if "device_id" not in df.columns and "terminal_code" in df.columns:
        df["device_id"] = df["terminal_code"].apply(lambda x: sha_prefix(x, 'dev') if pd.notna(x) else None)

    return df


def generate_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "transaction_datetime" in df.columns:
        df["hour"] = df["transaction_datetime"].dt.hour
        df["day"] = df["transaction_datetime"].dt.day
        df["weekday"] = df["transaction_datetime"].dt.weekday
        df["is_weekend"] = df["weekday"].isin([5,6]).astype(int)
        df["is_night"] = df["hour"].between(0,5).astype(int)
    else:
        df["hour"] = df["day"] = df["weekday"] = 0
        df["is_weekend"] = df["is_night"] = 0
    return df


def generate_location_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "country_code" in df.columns:
        df["country_code"] = df["country_code"].astype(str).str.strip().str.upper().replace({"NAN": None})
    else:
        df["country_code"] = None

    def is_international(c):
        if c is None or c == "" or str(c).upper() in ["NONE", "UNKNOWN"]:
            return 0
        if str(c).upper() in ["BO","BOL","BOLIVIA"]:
            return 0
        return 1

    df["feature_international_transaction"] = df["country_code"].apply(is_international).astype(int)
    return df


def infer_card_presence(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    tp_set = {2,5,7,90,91}
    tnp_set = {10,81}
    df["pos_entry_mode"] = pd.to_numeric(df.get("pos_entry_mode", pd.NA), errors="coerce")
    if "has_pinblock" in df.columns:
        df["has_pinblock"] = pd.to_numeric(df["has_pinblock"], errors="coerce").fillna(0).astype(int)
    else:
        df["has_pinblock"] = 0

    def infer_presence(row):
        pem = row.get("pos_entry_mode")
        has_pin = row.get("has_pinblock")
        try:
            if (not pd.isna(pem) and int(pem) in tp_set) or int(has_pin) == 1:
                return "TP"
            if (not pd.isna(pem) and int(pem) in tnp_set) or int(has_pin) == 0:
                return "TNP"
        except Exception:
            return "UNKNOWN"
        return "UNKNOWN"

    df["card_presence_type"] = df.apply(infer_presence, axis=1)
    return df


def preprocess_dataframe(df: pd.DataFrame, apply_smote: bool = False) -> Tuple[pd.DataFrame, Dict]:
    """FASE A: cleaning and feature generation only. No SMOTE, no heavy encoding.

    Returns cleaned dataframe and a summary report.
    """
    summary: Dict = {}
    before = len(df)
    summary["before"] = before

    if df is None or df.empty:
        summary.update({"after_clean": 0, "report": {}})
        return pd.DataFrame(), summary

    df = df.copy()
    # Normalize column names
    df = normalize_column_names(df)

    # Validate minimal structure
    try:
        validate_minimum_columns(df)
    except Exception as e:
        raise

    # Convert data types
    df = convert_data_types(df)

    # Handle missing values
    df, missing_report = handle_missing_values(df)
    summary["missing_report"] = missing_report

    # Remove duplicates
    df, dup_report = remove_duplicates(df)
    summary["duplicates_report"] = dup_report

    # Generate anonymized keys
    df = generate_anonymized_keys(df)

    # Time features
    df = generate_time_features(df)

    # Location features
    df = generate_location_features(df)

    # Infer card presence
    df = infer_card_presence(df)

    # Scale amount into a normalized column for modeling diagnostics
    try:
        if "amount" in df.columns:
            vals = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
            mn = vals.min()
            mx = vals.max()
            if pd.isna(mn) or pd.isna(mx) or mx - mn == 0:
                df["amount_scaled"] = 0.0
            else:
                df["amount_scaled"] = ((vals - mn) / (mx - mn)).astype(float)
        else:
            df["amount_scaled"] = 0.0
    except Exception:
        df["amount_scaled"] = 0.0

    # Generate behavioral features (these will add binary rule columns)
    try:
        df = proxy_labeling.generate_behavioral_risk_features(df)
    except Exception:
        # if behavioral features fail, ensure flow continues
        pass

    # Calculate behavioral score and independent groups for diagnostics
    try:
        df["behavioral_risk_score"] = proxy_labeling.calculate_behavioral_risk_score(df)
        df["independent_rule_groups"] = proxy_labeling.calculate_independent_rule_groups(df)
    except Exception:
        df["behavioral_risk_score"] = 0.0
        df["independent_rule_groups"] = 0

    # Generate proxy weak label but do NOT use response codes
    try:
        df = proxy_labeling.generate_proxy_fraud_label(df)
    except Exception:
        pass

    after_clean = len(df)
    summary["after_clean"] = after_clean
    summary["columns"] = list(df.columns)
    # report which columns were transformed/added during preprocessing
    summary["columns_transformed"] = ["amount_scaled"]

    return df, summary


def save_processed(df: pd.DataFrame, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

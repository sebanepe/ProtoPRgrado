import os
import unicodedata
import hashlib
import re
import pandas as pd
import numpy as np
from typing import Tuple, Dict
from backend.app.models.models import Transaction
from backend.app.ml import proxy_labeling


COUNTRY_CODE_CANONICAL_MAP = {
    # USA variants
    "USA": "US",
    "UNITED STATES": "US",
    "ESTADOS UNIDOS": "US",
    # ISO-3 to ISO-2 mappings (alphabetical)
    "ARG": "AR",
    "AUS": "AU",
    "AUT": "AT",
    "BEL": "BE",
    "BGR": "BG",
    "BOL": "BO",
    "BOLIVIA": "BO",
    "BRA": "BR",
    "CAN": "CA",
    "CHE": "CH",
    "CHL": "CL",
    "CHN": "CN",
    "COL": "CO",
    "CRI": "CR",
    "CYP": "CY",
    "DEU": "DE",
    "DNK": "DK",
    "DOM": "DO",
    "ECU": "EC",
    "EST": "EE",
    "ESP": "ES",
    "FIN": "FI",
    "FRA": "FR",
    "GBR": "GB",
    "GRC": "GR",
    "GTM": "GT",
    "HND": "HN",
    "IND": "IN",
    "IRL": "IE",
    "ITA": "IT",
    "JAM": "JM",
    "JPN": "JP",
    "KOR": "KR",
    "LTU": "LT",
    "LUX": "LU",
    "MYS": "MY",
    "MEX": "MX",
    "NIC": "NI",
    "NLD": "NL",
    "NOR": "NO",
    "NZL": "NZ",
    "PAN": "PA",
    "PER": "PE",
    "PHL": "PH",
    "POL": "PL",
    "PRT": "PT",
    "PRY": "PY",
    "SAU": "SA",
    "SRB": "RS",
    "SGP": "SG",
    "SLV": "SV",
    "SVK": "SK",
    "SWE": "SE",
    "THA": "TH",
    "TUR": "TR",
    "URY": "UY",
    "VEN": "VE",
    "ARE": "AE",
}

MERCANT_RUBRO_CANDIDATES = [
    "merchant_rubro_proxy",
    "rubro",
    "mcc",
    "mcc_code",
    "merchant_category_code",
    "codigo_rubro",
    "codigo_mcc",
    "categoria_comercio",
]

MERCANT_RUBRO_SOURCE_COLUMNS = {
    "rubro",
    "mcc",
    "mcc_code",
    "merchant_category_code",
    "codigo_rubro",
    "codigo_mcc",
    "categoria_comercio",
}

MERCANT_RUBRO_RAW_DROP_COLUMNS = MERCANT_RUBRO_SOURCE_COLUMNS | {
    "merchant_rubro_description",
    "description",
    "description_1",
    "response_description",
}


def _normalize_text_token(value) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        text = unicodedata.normalize("NFKD", str(value))
        text = "".join(ch for ch in text if not unicodedata.combining(ch))
        text = text.replace("\u00A0", " ").strip().upper()
        text = " ".join(text.split())
        return text or None
    except Exception:
        text = str(value).replace("\u00A0", " ").strip().upper()
        text = " ".join(text.split())
        return text or None


def _normalize_column_key(value) -> str:
    if not isinstance(value, str):
        value = str(value)
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
    return value


def normalize_country_code_value(value) -> str:
    """Normalize country_code: ISO-3 to ISO-2, clean Bolivia variants, UNKNOWN fallback."""
    normalized = _normalize_text_token(value)
    if normalized is None or normalized in {"", "NAN", "NONE", "NULL", "UNKNOWN"}:
        return "UNKNOWN"
    
    # Check if already in map (ISO-3 or known variants)
    if normalized in COUNTRY_CODE_CANONICAL_MAP:
        return COUNTRY_CODE_CANONICAL_MAP[normalized]
    
    # Handle dirty Bolivia variants (ends with BO, e.g., 0BO, ZBO, CBO, etc.)
    if len(normalized) <= 3 and normalized.endswith("BO") and normalized != "BO":
        # Pattern: ?BO where ? is any character
        return "BO"
    
    # Return as-is if unrecognized but looks valid (2 chars, typically ISO-2)
    return normalized


def normalize_merchant_rubro_value(value) -> str:
    normalized = _normalize_text_token(value)
    if normalized is None or normalized in {"", "NAN", "NONE", "NULL", "UNKNOWN"}:
        return "UNKNOWN"
    try:
        numeric = pd.to_numeric(str(value).strip().replace(",", "."), errors="coerce")
        if not pd.isna(numeric):
            if float(numeric).is_integer():
                return str(int(numeric))
            return str(float(numeric)).rstrip("0").rstrip(".")
    except Exception:
        pass

    if normalized.endswith(".0") and normalized[:-2].isdigit():
        return normalized[:-2]

    return normalized


def ensure_merchant_rubro_proxy(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def pick_rubro(row):
        for col in MERCANT_RUBRO_CANDIDATES:
            if col in row and not pd.isna(row[col]):
                value = normalize_merchant_rubro_value(row[col])
                if value != "UNKNOWN":
                    return value
        return "UNKNOWN"

    if any(col in df.columns for col in MERCANT_RUBRO_CANDIDATES):
        df["merchant_rubro_proxy"] = df.apply(pick_rubro, axis=1)
    else:
        df["merchant_rubro_proxy"] = "UNKNOWN"
    return df


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
                "merchant_hash": r.merchant_hash,
                "merchant_code": r.merchant_code,
                "terminal_code": r.terminal_code,
                "merchant_name": r.merchant_name,
                "merchant_rubro_proxy": r.merchant_rubro_proxy,
                "country_code": r.country_code,
                "pos_entry_mode": r.pos_entry_mode,
                "has_pinblock": r.has_pinblock,
                "card_brand": r.card_brand,
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
        "categoria_transaccion": "transaction_category",
        "codigo_de_proceso": "process_code",
        "tiene_pinblock": "has_pinblock",
        "numero_referencia": "reference_number",
        "codigo_autorizacion": "authorization_code",
        "codigo_establecimiento": "merchant_code",
        "moneda": "currency_code",
        "codigo_terminal": "terminal_code",
        "canal": "channel",
        "hora": "transaction_datetime",
        "fecha": "transaction_datetime",
        "fecha_transaccion": "transaction_datetime",
        "fecha_operacion": "transaction_datetime",
        "fecha_hora": "transaction_datetime",
        "datetime": "transaction_datetime",
        "timestamp": "transaction_datetime",
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
        def _to_pin(x):
            if pd.isna(x):
                return np.nan
            s = str(x).strip().lower()
            if s in ["1", "true", "yes", "y"]:
                return 1
            if s in ["0", "false", "no", "n"]:
                return 0
            try:
                return int(float(s))
            except Exception:
                return np.nan
        df["has_pinblock"] = df["has_pinblock"].apply(_to_pin)

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

    # merchant_hash hint (actual hashing happens later)
    if "merchant_hash" not in df.columns:
        if "merchant_code" in df.columns:
            df["merchant_hash"] = df["merchant_code"].astype(str)
        elif "terminal_code" in df.columns:
            df["merchant_hash"] = df["terminal_code"].astype(str)
        elif "merchant_name" in df.columns:
            df["merchant_hash"] = df["merchant_name"].astype(str)
        else:
            df["merchant_hash"] = "UNKNOWN_MERCHANT"

    # customer_hash: if missing and masked_card/pan_card available, will be generated later; otherwise drop
    if "customer_hash" not in df.columns or df["customer_hash"].isna().all():
        if not any(c in df.columns for c in ["masked_card", "pan_card"]):
            # cannot derive customer_hash -> drop rows
            report["reasons"].append("dropped_missing_customer_hash")
            df = df.loc[~df.index.isin(df.index)] if df.empty else df

    # has_pinblock impute (only when source is imputed)
    if "has_pinblock" in df.columns:
        if "has_pinblock_source" in df.columns:
            mask_imputed = df["has_pinblock_source"] == "imputed"
            if mask_imputed.any():
                df.loc[mask_imputed, "has_pinblock"] = df.loc[mask_imputed, "has_pinblock"].fillna(0)
        else:
            df["has_pinblock"] = df["has_pinblock"].fillna(0)
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
        # deterministically generate transaction_id from stable fields to ensure reproducible deduplication
        def make_tid(row):
            components = [
                str(row.get('transaction_datetime', '')),
                str(row.get('amount', '')),
                str(row.get('customer_hash', '')),
                str(row.get('merchant_hash', '')),
                str(row.get('reference_number', '')),
                str(row.get('authorization_code', '')),
            ]
            key = "|".join([c if c is not None else '' for c in components])
            return sha_prefix(key, 'tx')

        df["transaction_id"] = [make_tid(row) for _, row in df.iterrows()]

    # customer_hash
    # customer_hash: if missing, derive from masked_card/pan_card; if present but looks like raw PAN, re-hash
    if "customer_hash" not in df.columns:
        if "masked_card" in df.columns:
            df["customer_hash"] = df["masked_card"].apply(lambda x: sha_prefix(x, 'cust') if pd.notna(x) else None)
        elif "pan_card" in df.columns:
            df["customer_hash"] = df["pan_card"].apply(lambda x: sha_prefix(x, 'cust') if pd.notna(x) else None)
    else:
        # sanitize existing customer_hash values that look like raw PAN (all digits, length>=12)
        def sanitize_customer(val, pan_val=None):
            try:
                if pd.isna(val) or val == '':
                    return None
                s = str(val).strip()
                if s.startswith('cust_'):
                    return s
                # if looks like PAN (digits >=12) or contains spaces/dashes and mostly digits
                digits = ''.join(ch for ch in s if ch.isdigit())
                if len(digits) >= 12 and (digits == s or len(digits) / len(s) > 0.7):
                    # prefer to hash PAN if provided as raw; use pan_val if available else s
                    raw = pan_val if pan_val is not None and not pd.isna(pan_val) else s
                    return sha_prefix(raw, 'cust')
                return s
            except Exception:
                return val

        if 'pan_card' in df.columns:
            df['customer_hash'] = df.apply(lambda r: sanitize_customer(r.get('customer_hash'), r.get('pan_card')), axis=1)
        else:
            df['customer_hash'] = df['customer_hash'].apply(lambda v: sanitize_customer(v, None))

    # merchant_hash
    if "merchant_hash" not in df.columns:
        if "merchant_code" in df.columns:
            df["merchant_hash"] = df["merchant_code"].apply(lambda x: sha_prefix(x, 'merch') if pd.notna(x) else 'UNKNOWN_MERCHANT')
        elif "terminal_code" in df.columns:
            df["merchant_hash"] = df["terminal_code"].apply(lambda x: sha_prefix(x, 'merch') if pd.notna(x) else 'UNKNOWN_MERCHANT')
        elif "merchant_name" in df.columns:
            df["merchant_hash"] = df["merchant_name"].apply(lambda x: sha_prefix(x, 'merch') if pd.notna(x) else 'UNKNOWN_MERCHANT')
        else:
            df["merchant_hash"] = 'UNKNOWN_MERCHANT'
    else:
        # sanitize merchant_hash if raw code provided
        def sanitize_merchant(val, code_val=None, name_val=None):
            try:
                if pd.isna(val) or val == '':
                    if code_val is not None and not pd.isna(code_val):
                        return sha_prefix(code_val, 'merch')
                    if name_val is not None and not pd.isna(name_val):
                        return sha_prefix(name_val, 'merch')
                    return 'UNKNOWN_MERCHANT'
                s = str(val).strip()
                if s.upper() == "UNKNOWN_MERCHANT":
                    return "UNKNOWN_MERCHANT"
                if s.startswith('merch_'):
                    return s
                # if looks like a code (mostly digits/letters short), hash it
                return sha_prefix(s, 'merch')
            except Exception:
                return val

        df['merchant_hash'] = df.apply(lambda r: sanitize_merchant(r.get('merchant_hash'), r.get('merchant_code'), r.get('merchant_name')), axis=1)

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


def generate_location_features_with_stats(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Generate location features and return normalization statistics."""
    df = df.copy()
    stats = {
        "iso3_normalized_count": 0,
        "bolivia_dirty_normalized_count": 0,
        "country_code_distribution": {},
        "bo_is_international_check": None,
        "unknown_is_international_check": None,
    }
    
    if "country_code" in df.columns:
        # Track before normalization for comparison
        before_values = df["country_code"].copy()
        
        # Apply normalization
        df["country_code"] = df["country_code"].apply(lambda x: normalize_country_code_value(x))
        
        # Count ISO-3 normalizations
        iso3_patterns = {"CHL", "MEX", "PRY", "DEU", "PAN", "COL", "ITA", "FRA", "PER", "CAN", "SGP", "CHE", "LUX", "CYP", "CHN", "ECU", "URY", "TUR", "JPN", "AUS", "NZL", "KOR", "PRT", "THA", "ARE", "GTM", "LTU", "DOM", "POL", "IND", "HND", "CRI", "VEN", "MYS", "GRC", "EST", "JAM", "NIC", "AUT", "NOR", "FIN", "BEL", "DNK", "SAU", "SLV", "BGR", "SVK", "SRB", "PHL", "BRA", "ESP", "SWE", "ARG", "NLD", "BOL", "BOLIVIA"}
        for val in before_values:
            if pd.notna(val):
                normalized = _normalize_text_token(val)
                if normalized and normalized in iso3_patterns:
                    stats["iso3_normalized_count"] += 1
                elif normalized and len(normalized) <= 3 and normalized.endswith("BO") and normalized != "BO":
                    stats["bolivia_dirty_normalized_count"] += 1
    else:
        df["country_code"] = "UNKNOWN"
    
    # Country code distribution
    try:
        stats["country_code_distribution"] = df["country_code"].fillna("UNKNOWN").astype(str).value_counts().head(20).to_dict()
    except Exception:
        stats["country_code_distribution"] = {}
    
    # Define is_international logic
    def is_international(c):
        if c in {"UNKNOWN", "BO"}:
            return 0
        return 1
    
    df["feature_international_transaction"] = df["country_code"].apply(is_international).astype(int)
    df["is_international"] = df["feature_international_transaction"].astype(int)
    
    # Verify BO and UNKNOWN are not international
    if "BO" in df["country_code"].values:
        bo_rows = df[df["country_code"] == "BO"]
        stats["bo_is_international_check"] = (bo_rows["is_international"] == 0).all()
    
    if "UNKNOWN" in df["country_code"].values:
        unknown_rows = df[df["country_code"] == "UNKNOWN"]
        stats["unknown_is_international_check"] = (unknown_rows["is_international"] == 0).all()
    
    return df, stats


def generate_location_features(df: pd.DataFrame) -> pd.DataFrame:
    df, _ = generate_location_features_with_stats(df)
    return df


def infer_card_presence(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    tp_set = {2,5,7,90,91}
    tnp_set = {10,81}
    df["pos_entry_mode"] = pd.to_numeric(df.get("pos_entry_mode", pd.NA), errors="coerce")
    if "has_pinblock" in df.columns:
        df["has_pinblock"] = pd.to_numeric(df["has_pinblock"], errors="coerce")
    else:
        df["has_pinblock"] = 0
    if "has_pinblock_source" not in df.columns:
        df["has_pinblock_source"] = "imputed"

    def infer_presence(row):
        pem = row.get("pos_entry_mode")
        has_pin = row.get("has_pinblock")
        src = row.get("has_pinblock_source")
        try:
            if (not pd.isna(pem) and int(pem) in tp_set) or int(has_pin) == 1:
                return "TP"
            if (not pd.isna(pem) and int(pem) in tnp_set):
                return "TNP"
            if (not pd.isna(pem) and int(has_pin) == 0 and src == "raw"):
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
    original_columns = list(df.columns)
    merchant_rubro_source_columns = [c for c in original_columns if _normalize_column_key(c) in MERCANT_RUBRO_SOURCE_COLUMNS]
    summary["merchant_rubro_source_present"] = bool(merchant_rubro_source_columns)
    summary["merchant_rubro_source_columns"] = merchant_rubro_source_columns
    # Normalize column names
    df = normalize_column_names(df)

    # Track has_pinblock provenance before any imputation
    if "has_pinblock" in df.columns:
        df["has_pinblock_source"] = "raw"
    else:
        df["has_pinblock_source"] = "imputed"

    # Validate minimal structure
    try:
        validate_minimum_columns(df)
    except Exception as e:
        raise

    # Convert data types
    df = convert_data_types(df)

    # Normalize future rubro/MCC aliases into a single column
    df = ensure_merchant_rubro_proxy(df)

    # Drop raw rubro/MCC source aliases and description variants so the Phase A
    # output only keeps the canonical merchant_rubro_proxy column.
    raw_drop_cols = [c for c in df.columns if _normalize_column_key(c) in MERCANT_RUBRO_RAW_DROP_COLUMNS and c != "merchant_rubro_proxy"]
    if raw_drop_cols:
        df = df.drop(columns=raw_drop_cols, errors="ignore")

    # Handle missing values
    df, missing_report = handle_missing_values(df)
    summary["missing_report"] = missing_report

    # Remove duplicates
    df, dup_report = remove_duplicates(df)
    summary["duplicates_report"] = dup_report

    # Generate anonymized keys
    df = generate_anonymized_keys(df)

    # Infer card_brand from PAN/masked before removing sensitive columns
    if "card_brand" not in df.columns or df["card_brand"].isna().all():
        def _infer_brand(val):
            try:
                if pd.isna(val):
                    return None
                s = str(val).strip()
                if not s:
                    return None
                for ch in s:
                    if ch.isdigit():
                        d = ch
                        break
                else:
                    return None
                if d == "4":
                    return "VISA"
                if d == "5":
                    return "MASTERCARD"
                if d == "6":
                    return "DISCOVER_OR_OTHER"
                return "UNKNOWN"
            except Exception:
                return None

        if "pan_card" in df.columns:
            df["card_brand"] = df["pan_card"].apply(_infer_brand)
        elif "masked_card" in df.columns:
            df["card_brand"] = df["masked_card"].apply(_infer_brand)
        else:
            df["card_brand"] = "UNKNOWN"

    # Time features
    df = generate_time_features(df)

    # Location features
    df, location_stats = generate_location_features_with_stats(df)
    summary["country_code_normalization"] = location_stats

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

    # Track helper-only proxy columns for reporting but not output persistence
    try:
        if "_card_product_unknown" in df.columns:
            summary["card_product_unknown_count"] = int(df["_card_product_unknown"].fillna(0).astype(int).sum())
        else:
            summary["card_product_unknown_count"] = 0
    except Exception:
        summary["card_product_unknown_count"] = 0

    # Ensure is_international column aligns with the computed feature
    try:
        if "feature_international_transaction" in df.columns:
            df["is_international"] = df["feature_international_transaction"].astype(int)
    except Exception:
        pass

    # Merchant rubro diagnostics for reporting and validation
    try:
        if "merchant_rubro_proxy" in df.columns:
            rubro_series = df["merchant_rubro_proxy"].fillna("UNKNOWN").astype(str).str.strip().replace({"": "UNKNOWN"})
            summary["merchant_rubro_unknown_count"] = int((rubro_series == "UNKNOWN").sum())
            summary["merchant_rubro_valid_4digit_count"] = int(rubro_series.str.fullmatch(r"\d{4}").fillna(False).sum())
            summary["merchant_rubro_distribution"] = rubro_series.value_counts(dropna=False).head(20).to_dict()
        else:
            summary["merchant_rubro_unknown_count"] = 0
            summary["merchant_rubro_valid_4digit_count"] = 0
            summary["merchant_rubro_distribution"] = {}
    except Exception:
        summary["merchant_rubro_unknown_count"] = 0
        summary["merchant_rubro_valid_4digit_count"] = 0
        summary["merchant_rubro_distribution"] = {}

    # Calculate behavioral score and independent groups for diagnostics
    try:
        df["behavioral_risk_score"] = proxy_labeling.calculate_behavioral_risk_score(df)
        df["independent_rule_groups"] = proxy_labeling.calculate_independent_rule_groups(df)
    except Exception:
        df["behavioral_risk_score"] = 0.0
        df["independent_rule_groups"] = 0

    # compute feature frequencies for reporting: count and percent for each feature_
    try:
        total_rows = len(df)
        feature_freq = {}
        for c in df.columns:
            if c.startswith("feature_"):
                ones = int(df[c].fillna(0).astype(int).sum())
                pct = (ones / total_rows) * 100 if total_rows > 0 else 0.0
                feature_freq[c] = {"count": ones, "percent": round(pct, 4)}
        summary["feature_frequencies"] = feature_freq
        # mark features that exceed 30%
        summary["features_above_30pct"] = [f for f, v in feature_freq.items() if v["percent"] > 30.0]
    except Exception:
        summary["feature_frequencies"] = {}
        summary["features_above_30pct"] = []

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
    # do not persist diagnostic/scaled columns
    df_to_save = df.copy()
    if "amount_scaled" in df_to_save.columns:
        df_to_save = df_to_save.drop(columns=["amount_scaled"], errors="ignore")
    df_to_save.to_csv(output_path, index=False)


def get_training_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Return X, y for training: y is `is_fraud`, X excludes sensitive/label/fuga columns."""
    if df is None or df.empty:
        return pd.DataFrame(), pd.Series(dtype=int)
    df = df.copy()
    # ensure consistent lowercase column names
    cols = [c for c in df.columns]

    exclude = {
        "is_fraud",
        "is_fraud_proxy",
        "label_source",
        "fraud_label_reason",
        "risk_signal_reason",
        "behavioral_risk_score",
        "independent_rule_groups",
        # response code variants
        "codigo_respuesta",
        "codigo_respuesta",
        "codigo_respuesta",
        "codigo_respuesta",
        "codigo_respuesta",
        "codigo_respuesta",
        "codigo_respuesta",
        "codigo_respuesta",
        "codigo_respuesta",
        "codigo_respuesta",
    }
    # additional variants
    for v in ["codigo_respuesta", "response_code", "cod_respuesta", "respuesta", "codigo_respuesta", "codigo_respuesta"]:
        exclude.add(v)

    sensitive = {
        "pan_tarjeta",
        "tarjeta",
        "pan_card",
        "masked_card",
        "customer_hash",
        "merchant_hash",
        "device_id",
        "transaction_id",
        "authorization_code",
        "codigo_autorizacion",
        "reference_number",
        "numero_referencia",
    }

    # Ensure we drop explicit leakage / label columns per requirement
    extra_exclude = {
        'response_code', 'normalized_response_code', 'response_high_risk', 'response_code_reason',
        'is_fraud_proxy', 'behavioral_risk_score', 'independent_rule_groups', 'label_source',
        'fraud_label_reason', 'risk_signal_reason', 'transaction_id', 'customer_hash', 'merchant_hash',
        'device_id', 'reference_number', 'authorization_code', 'merchant_code', 'terminal_code',
        'pan_card', 'masked_card', 'PAN_TARJETA', 'TARJETA', 'merchant_name', 'transaction_datetime',
        'has_pinblock_source'
    }
    exclude = exclude.union(extra_exclude)

    exclude = exclude.union(sensitive)

    # drop any of the excluded columns that exist
    X = df.drop(columns=[c for c in df.columns if c in exclude], errors="ignore")

    # target
    y = None
    if "is_fraud" in df.columns:
        y = df["is_fraud"].astype(int)
    else:
        y = pd.Series([0] * len(df), index=df.index, dtype=int)

    return X, y


def split_train_test(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    """Split into train/test using stratify when possible."""
    from sklearn.model_selection import train_test_split

    if X is None or X.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.Series(dtype=int), pd.Series(dtype=int)

    stratify = None
    try:
        if y is not None and len(y.unique()) > 1:
            stratify = y
    except Exception:
        stratify = None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    return X_train, X_test, y_train, y_test


def detect_feature_types(X: pd.DataFrame):
    """Detect numeric, categorical and boolean columns in X."""
    if X is None or X.empty:
        return {"numeric": [], "categorical": [], "boolean": []}

    numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    # boolean columns: dtype bool or numeric with only 0/1
    boolean = [c for c in X.columns if X[c].dtype == bool]
    for c in X.columns:
        if c not in boolean and c not in numeric:
            # check if numeric-like but stored as object with 0/1
            vals = X[c].dropna().unique()
            if len(vals) > 0 and set([str(v) for v in vals]).issubset({"0", "1", "0.0", "1.0", "True", "False", "true", "false"}):
                boolean.append(c)

    # categorical: object, category, but exclude boolean
    categorical = [c for c in X.select_dtypes(include=[object, "category"]).columns.tolist() if c not in boolean]

    # ensure numeric does not contain boolean columns
    numeric = [c for c in numeric if c not in boolean]

    return {"numeric": numeric, "categorical": categorical, "boolean": boolean}


def build_preprocessing_pipeline(X: pd.DataFrame):
    """Build ColumnTransformer for numeric scaling and categorical one-hot encoding.

    Returns the ColumnTransformer and a sklearn Pipeline that applies it.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer

    types = detect_feature_types(X)

    numeric_cols = types["numeric"]
    cat_cols = types["categorical"]
    bool_cols = types["boolean"]

    transformers = []
    if numeric_cols:
        transformers.append(("num", StandardScaler(), numeric_cols))
    # Do NOT apply OneHotEncoding at preprocessing stage to avoid exploding columns.
    # OneHotEncoding will be applied inside the training pipeline after train/test split.
    # Therefore we skip categorical transformers here and only keep numeric/boolean handling.
    if bool_cols:
        # convert boolean-like to int
        transformers.append(("bool", Pipeline([("to_int", FunctionTransformer(lambda x: x.astype(int), validate=False))]), bool_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    pipeline = Pipeline([("preprocessor", preprocessor)])
    return preprocessor, pipeline


def apply_smote_if_needed(X_train: pd.DataFrame, y_train: pd.Series):
    """Apply SMOTE to X_train/y_train when appropriate.

    Returns X_res, y_res, report dict
    """
    report = {"smote_applied": False, "before_distribution": {}, "after_distribution": {}, "reason": None}

    try:
        from collections import Counter
        from imblearn.over_sampling import SMOTE
    except Exception as e:
        report["reason"] = f"imblearn_missing:{e}"
        return X_train, y_train, report

    if y_train is None or len(y_train.unique()) <= 1:
        report["reason"] = "single_class"
        report["before_distribution"] = dict(Counter(y_train))
        return X_train, y_train, report

    counts = dict(y_train.value_counts().to_dict())
    report["before_distribution"] = counts
    minority_count = min(counts.values())

    # determine k_neighbors safe
    k_neighbors = min(5, max(1, minority_count - 1))
    if minority_count < 2:
        report["reason"] = "too_few_minority"
        return X_train, y_train, report

    try:
        sm = SMOTE(random_state=42, k_neighbors=k_neighbors)
        X_res, y_res = sm.fit_resample(X_train, y_train)
        report["smote_applied"] = True
        report["after_distribution"] = dict(pd.Series(y_res).value_counts().to_dict())
        return X_res, y_res, report
    except Exception as e:
        report["reason"] = f"smote_failed:{e}"
        return X_train, y_train, report


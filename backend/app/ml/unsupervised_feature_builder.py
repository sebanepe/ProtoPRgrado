from __future__ import annotations

import os
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from backend.app.ml import preprocessing


PROHIBITED_FEATURE_COLUMNS = {
    "is_fraud",
    "confirmed_fraud",
    "target_is_fraud",
    "analyst_label",
    "rule_code",
    "rule_name",
    "alert_reason",
    "risk_score",
    "behavioral_risk_score",
    "pan_tarjeta",
    "tarjeta",
    "pan_card",
    "masked_card",
    "authorization_code",
    "reference_number",
    "response_description",
    "merchant_rubro_description",
    "description",
}

CONTEXT_COLUMNS = [
    "source_run",
    "transaction_id",
    "customer_hash",
    "transaction_datetime",
    "amount",
    "country_code",
    "pos_entry_mode",
    "has_pinblock",
    "merchant_rubro_proxy",
    "card_presence_type",
]

NUMERIC_FEATURE_COLUMNS = [
    "amount",
    "amount_log",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "is_international",
    "has_pinblock",
    "is_internet_transaction",
    "is_contactless",
    "is_contactless_no_pin",
    "is_magstripe",
    "is_cash_mcc",
    "is_atm_mcc",
    "is_gambling_mcc",
    "is_jewelry_mcc",
    "is_high_risk_mcc",
    "tx_count_customer_1h",
    "tx_count_customer_24h",
    "distinct_countries_customer_day",
    "distinct_mcc_customer_day",
    "avg_amount_customer_day",
    "max_amount_customer_day",
    "amount_vs_customer_day_avg",
]

CATEGORICAL_FEATURE_COLUMNS = ["country_code", "pos_entry_mode", "merchant_rubro_proxy", "card_presence_type"]
MODEL_INPUT_COLUMNS = NUMERIC_FEATURE_COLUMNS + CATEGORICAL_FEATURE_COLUMNS

POS_CONTACTLESS_CODES = {2, 5, 7, 90, 91}
POS_INTERNET_CODES = {10, 81}
POS_MAGSTRIPE_CODES = {80, 90, 91}

MCC_CASH_CODES = {6010, 6011, 6012, 6050, 6051, 4829}
MCC_ATM_CODES = {6010, 6011, 6051}
MCC_GAMBLING_CODES = {7995}
MCC_JEWELRY_CODES = {5094, 5944}
MCC_HIGH_RISK_CODES = MCC_CASH_CODES | MCC_ATM_CODES | MCC_GAMBLING_CODES | MCC_JEWELRY_CODES | {6211, 6536}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_processed_dir() -> Path:
    return Path(os.environ.get("PROJECT_PROCESSED_DIR") or (_repo_root() / "data" / "processed"))


def default_models_dir() -> Path:
    return Path(os.environ.get("PROJECT_MODELS_DIR") or (_repo_root() / "data" / "models"))


def normalize_run_token(source_run: Any) -> str:
    value = str(source_run).strip()
    match = re.search(r"(\d+)(?!.*\d)", value)
    if match:
        return match.group(1)
    value = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")
    return value or "run"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_datetime(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    try:
        return parsed.dt.tz_localize(None)
    except Exception:
        return pd.to_datetime(parsed.astype(str), errors="coerce")


def _ensure_transaction_datetime(df: pd.DataFrame) -> pd.DataFrame:
    if "transaction_datetime" in df.columns:
        df["transaction_datetime"] = _ensure_datetime(df["transaction_datetime"])
    elif "date" in df.columns and "time" in df.columns:
        df["transaction_datetime"] = _ensure_datetime(df["date"].astype(str) + " " + df["time"].astype(str))
    else:
        for candidate in ["datetime", "fecha", "fecha_hora", "timestamp"]:
            if candidate in df.columns:
                df["transaction_datetime"] = _ensure_datetime(df[candidate])
                break
    return df


def _normalize_rubro_token(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "UNKNOWN"
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace("\u00A0", " ").strip().upper()
    text = " ".join(text.split())
    if text in {"", "NAN", "NONE", "NULL", "UNKNOWN"}:
        return "UNKNOWN"
    numeric = pd.to_numeric(str(value).strip().replace(",", "."), errors="coerce")
    if not pd.isna(numeric):
        if float(numeric).is_integer():
            return str(int(numeric))
        return str(float(numeric)).rstrip("0").rstrip(".")
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def _normalize_pos_value(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "UNKNOWN"
    text = str(value).strip()
    if not text:
        return "UNKNOWN"
    numeric = pd.to_numeric(text, errors="coerce")
    if not pd.isna(numeric):
        if float(numeric).is_integer():
            return str(int(numeric))
        return str(float(numeric))
    return text.upper()


def _contains_any(series: pd.Series, needles: tuple[str, ...]) -> pd.Series:
    text = series.fillna("").astype(str).str.upper()
    result = pd.Series(False, index=series.index)
    for needle in needles:
        result = result | text.str.contains(needle, regex=False, na=False)
    return result


def _code_match(series: pd.Series, codes: set[int], needles: tuple[str, ...] = ()) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").astype("Int64")
    result = numeric.isin(codes)
    if needles:
        result = result | _contains_any(series, needles)
    return result.fillna(False)


def _safe_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default).astype(float)


def _prepare_input_frame(input_path: str) -> pd.DataFrame:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    df = preprocessing.normalize_column_names(df)
    df = preprocessing.convert_data_types(df)
    df = _ensure_transaction_datetime(df)

    if "transaction_id" not in df.columns or df["transaction_id"].isna().all():
        df = preprocessing.generate_anonymized_keys(df)
    if "customer_hash" not in df.columns:
        df = preprocessing.generate_anonymized_keys(df)
    if "merchant_hash" not in df.columns:
        df = preprocessing.generate_anonymized_keys(df)

    if "merchant_rubro_proxy" not in df.columns:
        df = preprocessing.ensure_merchant_rubro_proxy(df)
    else:
        df["merchant_rubro_proxy"] = df["merchant_rubro_proxy"].apply(_normalize_rubro_token)

    if "card_presence_type" not in df.columns:
        df = preprocessing.infer_card_presence(df)
    else:
        df["card_presence_type"] = df["card_presence_type"].fillna("UNKNOWN").astype(str).str.strip().replace({"": "UNKNOWN"})

    if "country_code" not in df.columns:
        df["country_code"] = "UNKNOWN"
    else:
        df["country_code"] = df["country_code"].apply(preprocessing.normalize_country_code_value)

    if "pos_entry_mode" not in df.columns:
        df["pos_entry_mode"] = np.nan

    if "has_pinblock" not in df.columns:
        df["has_pinblock"] = 0

    df["amount"] = _safe_numeric(df.get("amount", pd.Series([0.0] * len(df))))
    df["amount_log"] = np.log1p(df["amount"].clip(lower=0.0))
    df["hour_of_day"] = df["transaction_datetime"].dt.hour.fillna(0).astype(int)
    df["day_of_week"] = df["transaction_datetime"].dt.dayofweek.fillna(0).astype(int)
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_international"] = (~df["country_code"].fillna("UNKNOWN").astype(str).isin(["UNKNOWN", "BO"])).astype(int)

    pos_series = df["pos_entry_mode"].apply(_normalize_pos_value)
    rubro_series = df["merchant_rubro_proxy"].fillna("UNKNOWN").astype(str)
    presence_series = df["card_presence_type"].fillna("UNKNOWN").astype(str).str.upper()
    pin_series = pd.to_numeric(df["has_pinblock"], errors="coerce").fillna(0).astype(int)

    df["is_internet_transaction"] = _code_match(pos_series, POS_INTERNET_CODES, ("INTERNET", "ECOM", "E-COM", "CARD NOT PRESENT", "CNP")).astype(int)
    df["is_contactless"] = (_code_match(pos_series, POS_CONTACTLESS_CODES, ("CONTACTLESS", "NFC")) | presence_series.eq("TP")).astype(int)
    df["is_contactless_no_pin"] = ((df["is_contactless"] == 1) & (pin_series == 0)).astype(int)
    df["is_magstripe"] = _code_match(pos_series, POS_MAGSTRIPE_CODES, ("MAG", "SWIPE", "MAGSTRIPE")).astype(int)

    rubro_numeric = pd.to_numeric(rubro_series.apply(_normalize_rubro_token), errors="coerce")
    df["is_cash_mcc"] = (rubro_numeric.isin(MCC_CASH_CODES) | rubro_series.str.contains("CASH", na=False)).astype(int)
    df["is_atm_mcc"] = (rubro_numeric.isin(MCC_ATM_CODES) | rubro_series.str.contains("ATM", na=False)).astype(int)
    df["is_gambling_mcc"] = (rubro_numeric.isin(MCC_GAMBLING_CODES) | rubro_series.str.contains("GAMBL", na=False)).astype(int)
    df["is_jewelry_mcc"] = (rubro_numeric.isin(MCC_JEWELRY_CODES) | rubro_series.str.contains("JEWEL", na=False)).astype(int)
    df["is_high_risk_mcc"] = (
        df["is_cash_mcc"]
        | df["is_atm_mcc"]
        | df["is_gambling_mcc"]
        | df["is_jewelry_mcc"]
        | rubro_numeric.isin(MCC_HIGH_RISK_CODES)
    ).astype(int)

    df = df.reset_index(drop=False).rename(columns={"index": "_original_order"})
    df["customer_key"] = df.get("customer_hash", pd.Series(["UNKNOWN_CUSTOMER"] * len(df))).fillna("UNKNOWN_CUSTOMER").astype(str)
    df["mcc_key"] = rubro_series.apply(_normalize_rubro_token).astype(str)
    df["transaction_day"] = df["transaction_datetime"].dt.floor("D")

    sorted_df = df.sort_values(["customer_key", "transaction_datetime", "_original_order"]).copy()
    rolling_1h = (
        sorted_df.groupby("customer_key", sort=False)
        .rolling("1h", on="transaction_datetime")["amount"]
        .count()
        .reset_index(level=0, drop=True)
        .fillna(0)
        .astype(float)
    )
    rolling_24h = (
        sorted_df.groupby("customer_key", sort=False)
        .rolling("24h", on="transaction_datetime")["amount"]
        .count()
        .reset_index(level=0, drop=True)
        .fillna(0)
        .astype(float)
    )
    sorted_df["tx_count_customer_1h"] = rolling_1h.to_numpy()
    sorted_df["tx_count_customer_24h"] = rolling_24h.to_numpy()

    day_groups = sorted_df.groupby(["customer_key", "transaction_day"], sort=False)
    sorted_df["distinct_countries_customer_day"] = day_groups["country_code"].transform(lambda s: s.fillna("UNKNOWN").astype(str).nunique()).astype(float)
    sorted_df["distinct_mcc_customer_day"] = day_groups["mcc_key"].transform(lambda s: s.fillna("UNKNOWN").astype(str).nunique()).astype(float)
    sorted_df["avg_amount_customer_day"] = day_groups["amount"].transform("mean").astype(float)
    sorted_df["max_amount_customer_day"] = day_groups["amount"].transform("max").astype(float)
    sorted_df["amount_vs_customer_day_avg"] = np.where(
        sorted_df["avg_amount_customer_day"].fillna(0.0) > 0,
        sorted_df["amount"] / sorted_df["avg_amount_customer_day"].replace(0, np.nan),
        0.0,
    )
    sorted_df["amount_vs_customer_day_avg"] = sorted_df["amount_vs_customer_day_avg"].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)

    sorted_df = sorted_df.sort_values("_original_order").reset_index(drop=True)
    source_run = sorted_df.get("source_run", pd.Series([None] * len(sorted_df)))

    output_frame = pd.DataFrame(index=sorted_df.index)
    for column in CONTEXT_COLUMNS:
        if column == "source_run":
            continue
        if column in sorted_df.columns:
            output_frame[column] = sorted_df[column]
    output_frame.insert(0, "source_run", source_run)

    for column in MODEL_INPUT_COLUMNS:
        if column in sorted_df.columns:
            output_frame[column] = sorted_df[column]

    output_frame["source_run"] = output_frame["source_run"].fillna("UNKNOWN")
    output_frame["country_code"] = output_frame["country_code"].fillna("UNKNOWN").astype(str)
    output_frame["merchant_rubro_proxy"] = output_frame["merchant_rubro_proxy"].fillna("UNKNOWN").astype(str)
    output_frame["card_presence_type"] = output_frame["card_presence_type"].fillna("UNKNOWN").astype(str)

    for forbidden in PROHIBITED_FEATURE_COLUMNS:
        if forbidden in output_frame.columns and forbidden not in {"transaction_id", "customer_hash"}:
            output_frame = output_frame.drop(columns=[forbidden], errors="ignore")

    output_frame = output_frame.drop_duplicates(subset=["transaction_id"], keep="first") if "transaction_id" in output_frame.columns else output_frame

    run_token = normalize_run_token(source_run.iloc[0] if isinstance(source_run, pd.Series) and len(source_run) else "run")
    output_frame["source_run"] = output_frame.get("source_run", run_token).fillna(run_token)

    metadata = {
        "source_run": str(source_run.iloc[0] if isinstance(source_run, pd.Series) and len(source_run) else run_token),
        "source_run_token": int(run_token) if run_token.isdigit() else run_token,
        "created_at": _utc_now(),
        "total_rows": int(len(output_frame)),
        "context_columns": [column for column in CONTEXT_COLUMNS if column in output_frame.columns],
        "numeric_features": [column for column in NUMERIC_FEATURE_COLUMNS if column in output_frame.columns],
        "categorical_features": [column for column in CATEGORICAL_FEATURE_COLUMNS if column in output_frame.columns],
        "model_input_columns": [column for column in MODEL_INPUT_COLUMNS if column in output_frame.columns],
        "excluded_columns": sorted(PROHIBITED_FEATURE_COLUMNS),
        "transaction_id_unique": bool(output_frame["transaction_id"].is_unique) if "transaction_id" in output_frame.columns else True,
    }
    return output_frame, metadata


def build_unsupervised_features(input_path: str, source_run: Any, output_dir: str | os.PathLike[str] | None = None) -> Tuple[str, Dict[str, Any]]:
    output_frame, metadata = _prepare_input_frame(input_path)
    run_token = normalize_run_token(source_run)
    output_directory = Path(output_dir) if output_dir else default_processed_dir()
    output_directory.mkdir(parents=True, exist_ok=True)
    feature_path = output_directory / f"unsupervised_feature_set_run_{run_token}.csv"
    output_frame.to_csv(feature_path, index=False)
    metadata.update(
        {
            "input_path": str(input_path),
            "feature_set_file": str(feature_path),
            "feature_set_name": feature_path.name,
            "row_count": int(len(output_frame)),
            "feature_file_columns": list(output_frame.columns),
        }
    )
    return str(feature_path), metadata

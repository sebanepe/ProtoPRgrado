import os
import warnings
import pandas as pd
import numpy as np
from typing import List


def normalize_response_code(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    try:
        s = str(value).strip()
        # remove trailing .0 from floats represented as strings
        if s.endswith('.0'):
            s = s[:-2]
        # if numeric and single digit, return two-digit with leading zero
        if s.isdigit():
            v = int(s)
            if v < 10:
                return f"0{v}"
            return str(v)
        return s
    except Exception:
        return str(value)


_RESPONSE_HIGH = {"07": "HIGH_RISK_RESPONSE_CODE_07", "7": "HIGH_RISK_RESPONSE_CODE_07", "41": "LOST_CARD_CODE_41", "43": "STOLEN_CARD_CODE_43", "59": "SUSPECTED_FRAUD_CODE_59"}


def _find_response_column(df: pd.DataFrame) -> str | None:
    candidates = ["CODIGO_RESPUESTA", "codigo_respuesta", "response_code", "RESPUESTA", "cod_respuesta"]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def generate_response_code_signal(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    col = _find_response_column(df)
    df["normalized_response_code"] = ""
    if col:
        df["normalized_response_code"] = df[col].apply(normalize_response_code)
    df["response_high_risk"] = df["normalized_response_code"].apply(lambda x: 1 if x in _RESPONSE_HIGH else 0)

    def reason_for(x):
        if x in _RESPONSE_HIGH:
            return _RESPONSE_HIGH[x]
        return "NO_HIGH_RISK_RESPONSE_CODE"

    df["response_code_reason"] = df["normalized_response_code"].apply(reason_for)
    return df


BEHAVIORAL_RISK_THRESHOLD = float(os.getenv("BEHAVIORAL_RISK_THRESHOLD", "0.60"))
MIN_INDEPENDENT_RULE_GROUPS = int(os.getenv("MIN_INDEPENDENT_RULE_GROUPS", "3"))


def _infer_card_brand_from_masked(masked_card: str) -> str:
    try:
        if not masked_card or pd.isna(masked_card):
            return None
        s = str(masked_card).strip()
        # look for leading digit
        for ch in s:
            if ch.isdigit():
                d = ch
                break
        else:
            return None
        if d == '4':
            return 'VISA'
        if d == '5':
            return 'MASTERCARD'
        if d == '6':
            return 'DISCOVER'
        return 'UNKNOWN'
    except Exception:
        return None


def generate_behavioral_risk_features(df: pd.DataFrame, amount_threshold: float = 1000.0) -> pd.DataFrame:
    df = df.copy()
    # Ensure datetime
    if "transaction_datetime" in df.columns:
        try:
            ts = pd.to_datetime(df["transaction_datetime"], errors="coerce", utc=True)
            if hasattr(ts.dt, "tz"):
                df["transaction_datetime"] = ts.dt.tz_convert("UTC").dt.tz_localize(None)
            else:
                df["transaction_datetime"] = ts
        except Exception:
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
    else:
        df["transaction_datetime"] = pd.NaT

    # basic amount
    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    else:
        df["amount"] = pd.Series(0.0, index=df.index)
    df["feature_high_amount"] = (df["amount"] >= amount_threshold).astype(int)

    # aggregate sums per customer per hour/day
    if "customer_hash" not in df.columns:
        df["customer_hash"] = None

    # floor to hour and day (lowercase 'h' for pandas frequency)
    df["_hour_bucket"] = df["transaction_datetime"].dt.floor("h")
    df["_day_bucket"] = df["transaction_datetime"].dt.floor("D")

    sum_hour = df.groupby(["customer_hash", "_hour_bucket"]) ["amount"].transform("sum")
    sum_day = df.groupby(["customer_hash", "_day_bucket"]) ["amount"].transform("sum")
    df["feature_high_amount_1h"] = (sum_hour >= 3000).astype(int)
    df["feature_high_amount_1d"] = (sum_day >= 10000).astype(int)

    # time features
    df["_hour"] = df["transaction_datetime"].dt.hour
    df["feature_night_transaction"] = df["_hour"].between(0, 5).astype(int)
    df["_weekday"] = df["transaction_datetime"].dt.weekday
    df["feature_weekend_transaction"] = df["_weekday"].isin([5, 6]).astype(int)

    # location / international
    df["country_code"] = df.get("country_code", df.get("country", None))
    df["is_international"] = df.get("is_international", 0)
    df["feature_international_transaction"] = ((df["country_code"].fillna("") .str.upper().isin(["BO", "BOL"])==False) | (df["is_international"] == 1)).astype(int)

    # frequency
    txs_day = df.groupby(["customer_hash", "_day_bucket"]).size()
    df["_txs_day"] = df.groupby(["customer_hash", "_day_bucket"]) ["transaction_datetime"].transform("count")
    df["_txs_hour"] = df.groupby(["customer_hash", "_hour_bucket"]) ["transaction_datetime"].transform("count")
    df["feature_many_customer_transactions_day"] = (df["_txs_day"] >= 10).astype(int)
    df["feature_many_customer_transactions_hour"] = (df["_txs_hour"] >= 5).astype(int)

    # merchants distinct customers
    if "merchant_hash" not in df.columns:
        df["merchant_hash"] = None
    df["_distinct_customers_merchant_day"] = df.groupby(["merchant_hash", "_day_bucket"])["customer_hash"].transform(lambda x: x.nunique())
    df["_distinct_customers_merchant_hour"] = df.groupby(["merchant_hash", "_hour_bucket"])["customer_hash"].transform(lambda x: x.nunique())
    df["feature_many_merchants_customer_day"] = (df["_distinct_customers_merchant_day"] >= 5).astype(int)
    df["feature_many_merchants_customer_hour"] = (df["_distinct_customers_merchant_hour"] >= 3).astype(int)

    # card presence type
    df["pos_entry_mode"] = pd.to_numeric(df.get("pos_entry_mode", pd.NA), errors="coerce")
    if "has_pinblock" in df.columns:
        df["has_pinblock"] = pd.to_numeric(df["has_pinblock"], errors="coerce").fillna(0).astype(int)
    else:
        df["has_pinblock"] = 0
    tp_set = {2,5,7,90,91}
    tnp_set = {10,81}
    def infer_presence(row):
        pem = row["pos_entry_mode"]
        has_pin = row["has_pinblock"]
        try:
            if (not pd.isna(pem) and int(pem) in tp_set) or has_pin == 1:
                return "TP"
            if (not pd.isna(pem) and int(pem) in tnp_set) or has_pin == 0:
                return "TNP"
        except Exception:
            return "UNKNOWN"
        return "UNKNOWN"

    df["card_presence_type"] = df.apply(infer_presence, axis=1)
    df["feature_tp_pem_07"] = ((df["card_presence_type"] == "TP") & (df["pos_entry_mode"] == 7)).astype(int)
    df["feature_tnp_transaction"] = (df["card_presence_type"] == "TNP").astype(int)

    # merchant / brand / product heuristics
    df["card_brand"] = df.get("card_brand", None)
    # infer brand from masked_card when missing
    if ("card_brand" not in df.columns) or df["card_brand"].isna().all():
        if "masked_card" in df.columns:
            df["card_brand"] = df["masked_card"].apply(_infer_card_brand_from_masked)
    df["card_product_proxy"] = df.get("card_product_proxy", None)
    if ("card_product_proxy" not in df.columns) or df["card_product_proxy"].isna().all():
        df["card_product_proxy"] = pd.Series(["UNKNOWN"] * len(df), index=df.index)
    # mark unknown product proxy rows to allow upstream reporting
    df["_card_product_unknown"] = (df["card_product_proxy"] == "UNKNOWN").astype(int)
    df["_brand_customers_merchant_day"] = df.groupby(["merchant_hash", "card_brand", "_day_bucket"]) ["customer_hash"].transform(lambda x: x.nunique())
    df["feature_same_merchant_15_cards_by_brand"] = (df["_brand_customers_merchant_day"] >= 15).astype(int)

    df["_product_customers_merchant_day"] = df.groupby(["merchant_hash", "card_product_proxy", "card_presence_type", "_day_bucket"]) ["customer_hash"].transform(lambda x: x.nunique())
    df["feature_same_merchant_20_cards_by_product_presence"] = (df["_product_customers_merchant_day"] >= 20).astype(int)
    # If product proxy is UNKNOWN we cannot reliably evaluate product-based rules;
    # force those features to 0 for UNKNOWN rows.
    mask_unknown = df["card_product_proxy"] == "UNKNOWN"
    if mask_unknown.any():
        df.loc[mask_unknown, "feature_same_merchant_20_cards_by_product_presence"] = 0

    # TNP 50 approved by product — approximate: count TNP transactions per product per day
    df["_tnp_product_day"] = 0
    mask_tnp = df["card_presence_type"] == "TNP"
    if mask_tnp.any():
        df.loc[mask_tnp, "_tnp_product_day"] = df.loc[mask_tnp].groupby(["card_product_proxy", "_day_bucket"]) ["transaction_datetime"].transform("count")
    df["feature_tnp_50_approved_by_product"] = (df["_tnp_product_day"] >= 50).astype(int)
    if mask_unknown.any():
        df.loc[mask_unknown, "feature_tnp_50_approved_by_product"] = 0
        # emit a warning so upstream reporting can capture prevalence
        try:
            import warnings
            warnings.warn(f"card_product_proxy UNKNOWN for {int(mask_unknown.sum())} records; product-based rules set to 0 for those rows")
        except Exception:
            pass

    # cleanup helper columns
    to_fill_zero = ["feature_high_amount", "feature_high_amount_1h", "feature_high_amount_1d", "feature_night_transaction", "feature_weekend_transaction", "feature_international_transaction", "feature_many_customer_transactions_day", "feature_many_customer_transactions_hour", "feature_many_merchants_customer_day", "feature_many_merchants_customer_hour", "feature_tp_pem_07", "feature_tnp_transaction", "feature_same_merchant_15_cards_by_brand", "feature_same_merchant_20_cards_by_product_presence", "feature_tnp_50_approved_by_product"]
    for c in to_fill_zero:
        if c not in df.columns:
            df[c] = 0
    # drop helper buckets
    df = df.drop(columns=[c for c in ["_hour_bucket", "_day_bucket", "_hour", "_weekday", "_txs_day", "_txs_hour", "_distinct_customers_merchant_day", "_distinct_customers_merchant_hour", "_brand_customers_merchant_day", "_product_customers_merchant_day", "_tnp_product_day"] if c in df.columns], errors="ignore")
    return df


def calculate_behavioral_risk_score(df: pd.DataFrame, weights: dict | None = None) -> pd.Series:
    if weights is None:
        weights = {
            "feature_high_amount": 1,
            "feature_high_amount_1h": 3,
            "feature_high_amount_1d": 2,
            "feature_night_transaction": 1,
            "feature_weekend_transaction": 0.5,
            "feature_international_transaction": 2,
            "feature_many_customer_transactions_day": 2,
            "feature_many_customer_transactions_hour": 3,
            "feature_many_merchants_customer_day": 2,
            "feature_many_merchants_customer_hour": 3,
            "feature_tp_pem_07": 2,
            "feature_tnp_transaction": 1,
            "feature_same_merchant_15_cards_by_brand": 2,
            "feature_same_merchant_20_cards_by_product_presence": 2,
            "feature_tnp_50_approved_by_product": 2,
        }
    total = sum(weights.values())
    # ensure all features exist
    s = pd.Series(0.0, index=df.index)
    for feat, w in weights.items():
        if feat in df.columns:
            s += df[feat].fillna(0).astype(float) * float(w)
    # normalize
    return (s / total).clip(0, 1)


def calculate_independent_rule_groups(df: pd.DataFrame) -> pd.Series:
    groups = {
        "amount": ["feature_high_amount", "feature_high_amount_1h", "feature_high_amount_1d"],
        "time": ["feature_night_transaction", "feature_weekend_transaction"],
        "location": ["feature_international_transaction"],
        "customer_frequency": ["feature_many_customer_transactions_day", "feature_many_customer_transactions_hour"],
        "merchant_behavior": ["feature_many_merchants_customer_day", "feature_many_merchants_customer_hour", "feature_same_merchant_15_cards_by_brand", "feature_same_merchant_20_cards_by_product_presence"],
        "card_presence": ["feature_tp_pem_07", "feature_tnp_transaction", "feature_tnp_50_approved_by_product"],
    }
    counts = pd.Series(0, index=df.index)
    for g, feats in groups.items():
        present = False
        for f in feats:
            if f in df.columns:
                present = True
                break
        if not present:
            continue
        # group is active if any feature in group is 1
        group_active = pd.Series(0, index=df.index)
        for f in feats:
            if f in df.columns:
                group_active = group_active | (df[f].fillna(0).astype(int) == 1)
        counts += group_active.astype(int)
    return counts


def generate_proxy_fraud_label(df: pd.DataFrame, amount_threshold: float = 1000.0, behavioral_threshold: float | None = None, min_independent_groups: int | None = None) -> pd.DataFrame:
    df = df.copy()
    # generate behavioral features (this will add required binary rule columns)
    df = generate_behavioral_risk_features(df, amount_threshold=amount_threshold)

    # behavioral score and independent groups
    df["behavioral_risk_score"] = calculate_behavioral_risk_score(df)
    df["independent_rule_groups"] = calculate_independent_rule_groups(df)

    # create list of binary rule column names we consider for reasons
    binary_rules = [
        "feature_high_amount", "feature_high_amount_1h", "feature_high_amount_1d", "feature_night_transaction", "feature_weekend_transaction",
        "feature_international_transaction", "feature_many_customer_transactions_day", "feature_many_customer_transactions_hour", "feature_many_merchants_customer_day",
        "feature_many_merchants_customer_hour", "feature_tp_pem_07", "feature_tnp_transaction", "feature_same_merchant_15_cards_by_brand",
        "feature_same_merchant_20_cards_by_product_presence", "feature_tnp_50_approved_by_product",
    ]

    # Ensure binary rule columns exist
    for f in binary_rules:
        if f not in df.columns:
            df[f] = 0

    # risk signals (all activated signals for diagnostics)
    def make_risk_signal_reason(row):
        reasons = []
        for f in binary_rules:
            try:
                if int(row.get(f, 0)) == 1:
                    reasons.append(f.upper())
            except Exception:
                continue
        return "|".join(reasons) if reasons else "NONE"

    df["risk_signal_reason"] = df.apply(make_risk_signal_reason, axis=1)

    # use configured thresholds if not provided
    if behavioral_threshold is None:
        behavioral_threshold = BEHAVIORAL_RISK_THRESHOLD
    if min_independent_groups is None:
        min_independent_groups = MIN_INDEPENDENT_RULE_GROUPS

    # decide proxy label purely from behavioral signals
    def decide(row):
        try:
            br = float(row.get("behavioral_risk_score", 0))
        except Exception:
            br = 0.0
        irg = int(row.get("independent_rule_groups", 0)) if row.get("independent_rule_groups", None) is not None else 0
        if br >= float(behavioral_threshold) and irg >= int(min_independent_groups):
            return 1
        return 0

    df["is_fraud_proxy"] = df.apply(decide, axis=1).astype(int)
    df["is_fraud"] = df["is_fraud_proxy"].astype(int)

    # fraud_label_reason only when positive
    def make_fraud_label_reason(row):
        if int(row.get("is_fraud", 0)) == 1:
            return row.get("risk_signal_reason", "NONE")
        return "NO_PROXY_FRAUD_LABEL"

    df["fraud_label_reason"] = df.apply(make_fraud_label_reason, axis=1)

    # label_source: behavioral or not detected
    df["label_source"] = df["is_fraud"].apply(lambda x: "behavioral_weak_label" if int(x) == 1 else "no_proxy_risk_detected")

    # warn if no positives found for this labeling configuration
    try:
        positives = int(df["is_fraud"].sum())
        if positives == 0:
            warnings.warn("generate_proxy_fraud_label: no positives found under current thresholds; check BEHAVIORAL_RISK_THRESHOLD and dataset size.")
    except Exception:
        pass

    return df

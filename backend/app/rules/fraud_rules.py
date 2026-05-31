from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
import re

import pandas as pd


DEFAULT_RULE_CONFIG: dict[str, Any] = {
    "max_tx_hour_default": 3,
    "max_tx_day_default": 10,
    "contactless_no_pin_hour": 5,
    "contactless_no_pin_day": 10,
    "internet_hour": 5,
    "internet_day": 10,
    "cash_hour": 3,
    "cash_day": 5,
    "magstripe_hour": 3,
    "contactless_amount_limit_bob": 200,
    "high_risk_amount_threshold": 1000,
    "contextual_high_risk_amount_threshold": 500,
    "jewelry_amount_threshold": 1000,
    "gambling_amount_threshold": 300,
    "include_recurring_in_general_velocity": False,
}

INTERNET_POS_ENTRY_MODES = {"0", "1", "81"}
CARD_PRESENT_POS_ENTRY_MODES = {"5", "7", "90"}
RECURRING_POS_ENTRY_MODE = "10"

DOUBLE_COUNTRY_RULE = "RULE_DOUBLE_COUNTRY_SAME_DAY"
DOUBLE_COUNTRY_CARD_PRESENT_RULE = "RULE_DOUBLE_COUNTRY_CARD_PRESENT_SAME_DAY"
DOUBLE_COUNTRY_CARD_ABSENT_RULE = "RULE_DOUBLE_COUNTRY_CARD_ABSENT_CONTEXTUAL"
VELOCITY_HOUR_RULE = "RULE_VELOCITY_CARD_HOUR"
VELOCITY_DAY_RULE = "RULE_VELOCITY_CARD_DAY"
CONTACTLESS_NO_PIN_HOUR_RULE = "RULE_CONTACTLESS_NO_PIN_HOUR"
CONTACTLESS_NO_PIN_DAY_RULE = "RULE_CONTACTLESS_NO_PIN_DAY"
INTERNET_VELOCITY_DAY_RULE = "RULE_INTERNET_VELOCITY_DAY"
MAGSTRIPE_VELOCITY_HOUR_RULE = "RULE_MAGSTRIPE_VELOCITY_HOUR"
ATM_VELOCITY_HOUR_RULE = "RULE_ATM_OR_CASH_WITHDRAWAL_VELOCITY_HOUR"
ATM_VELOCITY_DAY_RULE = "RULE_ATM_OR_CASH_WITHDRAWAL_DAY"
STRONG_HIGH_RISK_MCC_RULE = "RULE_STRONG_HIGH_RISK_MCC_AMOUNT"
CONTEXTUAL_HIGH_RISK_MCC_RULE = "RULE_CONTEXTUAL_HIGH_RISK_MCC_WITH_SIGNAL"
GAMBLING_MCC_RULE = "RULE_GAMBLING_MCC"
JEWELRY_HIGH_AMOUNT_RULE = "RULE_JEWELRY_MCC_HIGH_AMOUNT"

RULE_DEFINITIONS: dict[str, dict[str, Any]] = {
    DOUBLE_COUNTRY_CARD_PRESENT_RULE: {"rule_name": "Double Country Card Present Same Day", "risk_level": "HIGH", "risk_score": 85},
    DOUBLE_COUNTRY_CARD_ABSENT_RULE: {"rule_name": "Double Country Card Absent Contextual", "risk_level": "MEDIUM", "risk_score": 65},
    VELOCITY_HOUR_RULE: {"rule_name": "Card Velocity Hour", "risk_level": "MEDIUM", "risk_score": 60},
    VELOCITY_DAY_RULE: {"rule_name": "Card Velocity Day", "risk_level": "MEDIUM", "risk_score": 65},
    CONTACTLESS_NO_PIN_HOUR_RULE: {"rule_name": "Contactless No PIN Hour", "risk_level": "HIGH", "risk_score": 80},
    CONTACTLESS_NO_PIN_DAY_RULE: {"rule_name": "Contactless No PIN Day", "risk_level": "HIGH", "risk_score": 80},
    INTERNET_VELOCITY_DAY_RULE: {"rule_name": "Internet Velocity Day", "risk_level": "HIGH", "risk_score": 75},
    MAGSTRIPE_VELOCITY_HOUR_RULE: {"rule_name": "Magstripe Velocity Hour", "risk_level": "HIGH", "risk_score": 70},
    ATM_VELOCITY_HOUR_RULE: {"rule_name": "ATM or Cash Withdrawal Velocity Hour", "risk_level": "HIGH", "risk_score": 85},
    ATM_VELOCITY_DAY_RULE: {"rule_name": "ATM or Cash Withdrawal Velocity Day", "risk_level": "HIGH", "risk_score": 80},
    STRONG_HIGH_RISK_MCC_RULE: {"rule_name": "Strong High Risk MCC Amount", "risk_level": "HIGH", "risk_score": 75},
    CONTEXTUAL_HIGH_RISK_MCC_RULE: {"rule_name": "Contextual High Risk MCC With Signal", "risk_level": "MEDIUM", "risk_score": 65},
    GAMBLING_MCC_RULE: {"rule_name": "Gambling MCC", "risk_level": "HIGH", "risk_score": 85},
    JEWELRY_HIGH_AMOUNT_RULE: {"rule_name": "Jewelry MCC High Amount", "risk_level": "HIGH", "risk_score": 80},
}

ATM_MCC_CODES = {"6010", "6011"}
STRONG_HIGH_RISK_MCC_CODES = {"6010", "6011", "6012", "6051", "6211", "7995"}
CONTEXTUAL_HIGH_RISK_MCC_CODES = {"5122", "5912", "5966", "5967", "5968", "5993", "7273", "5816", "5944", "5094"}
GAMBLING_MCC_CODES = {"7995", "7800", "7801", "7802"}
JEWELRY_MCC_CODES = {"5944", "5094"}


def _merge_config(config: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(DEFAULT_RULE_CONFIG)
    if config:
        merged.update(config)
    return merged


def _normalize_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "UNKNOWN"
    text = str(value).strip()
    if not text:
        return "UNKNOWN"
    text = text.replace("\u00A0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.upper()


def _normalize_amount(value: Any) -> float:
    try:
        amount = pd.to_numeric(value, errors="coerce")
        if pd.isna(amount):
            return 0.0
        return float(amount)
    except Exception:
        return 0.0


def _normalize_pos_entry_mode(value: Any) -> str:
    text = _normalize_text(value)
    if text == "UNKNOWN":
        return "UNKNOWN"
    numeric = pd.to_numeric(text, errors="coerce")
    if not pd.isna(numeric):
        if float(numeric).is_integer():
            return str(int(numeric))
        return str(int(float(numeric)))
    if text.endswith(".0"):
        base = text[:-2].strip()
        if base.isdigit():
            return str(int(base))
    return text


def _normalize_pinblock(value: Any) -> int:
    try:
        numeric = pd.to_numeric(value, errors="coerce")
        if pd.isna(numeric):
            return 0
        return 1 if int(float(numeric)) != 0 else 0
    except Exception:
        text = _normalize_text(value)
        return 1 if text in {"1", "TRUE", "YES", "Y"} else 0


def _normalize_country_code(value: Any) -> str:
    text = _normalize_text(value)
    if text in {"", "NAN", "NONE", "NULL", "UNKNOWN"}:
        return "UNKNOWN"
    return text


def _normalize_mcc_code(value: Any) -> str:
    text = _normalize_text(value)
    if text in {"", "NAN", "NONE", "NULL", "UNKNOWN"}:
        return "UNKNOWN"
    if text.endswith(".0"):
        text = text[:-2]
    compact = re.sub(r"\s+", "", text)
    if compact.isdigit():
        if len(compact) == 3:
            return compact.zfill(4)
        return compact
    return text


def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    for column in ["transaction_id", "customer_hash", "transaction_datetime", "country_code", "pos_entry_mode", "merchant_rubro_proxy"]:
        if column not in prepared.columns:
            prepared[column] = pd.NA

    prepared["transaction_id"] = prepared["transaction_id"].fillna("UNKNOWN").astype(str).str.strip()
    prepared["customer_hash"] = prepared["customer_hash"].fillna("UNKNOWN").astype(str).str.strip()
    prepared["country_code"] = prepared["country_code"].apply(_normalize_country_code)
    prepared["pos_entry_mode"] = prepared["pos_entry_mode"].apply(_normalize_pos_entry_mode)
    prepared["has_pinblock"] = prepared["has_pinblock"].apply(_normalize_pinblock) if "has_pinblock" in prepared.columns else 0
    prepared["amount"] = prepared["amount"].apply(_normalize_amount) if "amount" in prepared.columns else 0.0
    prepared["merchant_rubro_proxy"] = prepared["merchant_rubro_proxy"].apply(_normalize_mcc_code)
    prepared["transaction_datetime"] = pd.to_datetime(prepared["transaction_datetime"], errors="coerce", utc=True)
    return prepared


def _build_alert_row(row: pd.Series, rule_code: str, alert_reason: str, config: dict[str, Any]) -> dict[str, Any]:
    definition = RULE_DEFINITIONS[rule_code]
    created_at = datetime.now(timezone.utc).isoformat()
    source_run = str(config.get("source_run", "UNKNOWN"))
    return {
        "alert_id": None,
        "source_run": source_run,
        "transaction_id": row.get("transaction_id", "UNKNOWN"),
        "customer_hash": row.get("customer_hash", "UNKNOWN"),
        "transaction_datetime": row.get("transaction_datetime").isoformat() if pd.notna(row.get("transaction_datetime")) else None,
        "amount": float(row.get("amount") or 0.0),
        "country_code": row.get("country_code", "UNKNOWN"),
        "pos_entry_mode": str(row.get("pos_entry_mode", "UNKNOWN")),
        "has_pinblock": int(row.get("has_pinblock") or 0),
        "merchant_rubro_proxy": row.get("merchant_rubro_proxy", "UNKNOWN"),
        "rule_code": rule_code,
        "rule_name": definition["rule_name"],
        "risk_level": definition["risk_level"],
        "risk_score": definition["risk_score"],
        "alert_reason": alert_reason,
        "triggered_rules": rule_code,
        "status": "NEW",
        "created_at": created_at,
    }


def _risk_from_context_signal_count(signal_count: int) -> tuple[str, int]:
    if signal_count <= 1:
        return "MEDIUM", 65
    if signal_count == 2:
        return "HIGH", 75
    if signal_count == 3:
        return "HIGH", 75
    return "HIGH", 80


def _is_double_country_rule(rule_code: Any) -> bool:
    return str(rule_code).startswith("RULE_DOUBLE_COUNTRY")


def _double_country_alert_reason(countries: list[str], rule_label: str) -> str:
    countries_text = ", ".join(countries)
    if rule_label == DOUBLE_COUNTRY_CARD_PRESENT_RULE:
        return f"Cliente anonimizado registra operaciones presenciales en más de un país durante el mismo día. Countries: {countries_text}"
    return f"Cliente anonimizado registra operaciones no presenciales en más de un país durante el mismo día junto con señales adicionales de riesgo. Countries: {countries_text}"


def _double_country_signal_details(group: pd.DataFrame, config: dict[str, Any]) -> list[str]:
    threshold = float(config.get("contextual_high_risk_amount_threshold", DEFAULT_RULE_CONFIG["contextual_high_risk_amount_threshold"]))
    unique_countries = sorted(set(group["country_code"].astype(str).tolist()) - {"UNKNOWN"})
    has_strong_mcc = group["_merchant_mcc"].isin(STRONG_HIGH_RISK_MCC_CODES).any()
    has_contextual_mcc = group["_merchant_mcc"].isin(CONTEXTUAL_HIGH_RISK_MCC_CODES).any()
    has_high_amount = group["amount"].ge(threshold).any()
    group_day_count = int(len(group))
    group_hour_max = int(group.groupby("_hour", dropna=False).size().max()) if not group.empty else 0
    bo_foreign = "BO" in unique_countries and any(country != "BO" for country in unique_countries)

    signals: list[str] = []
    if len(unique_countries) > 2:
        signals.append("countries_gt_2")
    if group_day_count > 10:
        signals.append("txs_day_gt_10")
    if group_hour_max > 3:
        signals.append("txs_hour_gt_3")
    if has_strong_mcc:
        signals.append("mcc_strong_risk")
    if has_contextual_mcc:
        signals.append("mcc_contextual")
    if has_high_amount:
        signals.append("amount_ge_contextual_threshold")
    if bo_foreign and (has_strong_mcc or has_contextual_mcc or has_high_amount):
        signals.append("bo_foreign_with_risk_or_amount")
    return signals


def _build_double_country_alerts(group: pd.DataFrame, config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    alerts: list[dict[str, Any]] = []
    present_mask = group["_pem_norm"].isin(CARD_PRESENT_POS_ENTRY_MODES) & group["amount"].gt(0)
    absent_mask = group["_pem_norm"].isin(INTERNET_POS_ENTRY_MODES) & group["amount"].gt(0)
    excluded_pem10 = int((group["_pem_norm"] == RECURRING_POS_ENTRY_MODE).sum())
    tnp_discarded_without_signal = 0

    present_group = group.loc[present_mask]
    present_countries = sorted(set(present_group["country_code"].astype(str).tolist()) - {"UNKNOWN"})
    if len(present_countries) >= 2:
        reason = _double_country_alert_reason(present_countries, DOUBLE_COUNTRY_CARD_PRESENT_RULE)
        for _, row in group.loc[present_mask].iterrows():
            alert = _build_alert_row(row, DOUBLE_COUNTRY_CARD_PRESENT_RULE, reason, config)
            alert["risk_level"] = "HIGH"
            alert["risk_score"] = 85
            alerts.append(alert)

    absent_group = group.loc[absent_mask]
    absent_countries = sorted(set(absent_group["country_code"].astype(str).tolist()) - {"UNKNOWN"})
    if len(absent_countries) >= 2:
        signals = _double_country_signal_details(absent_group, config)
        if signals:
            risk_level, risk_score = _risk_from_context_signal_count(len(signals))
            reason = _double_country_alert_reason(absent_countries, DOUBLE_COUNTRY_CARD_ABSENT_RULE) + f" Signals: {', '.join(signals)}"
            triggered_rules = f"{DOUBLE_COUNTRY_CARD_ABSENT_RULE}|" + "|".join(signals)
            for _, row in group.loc[absent_mask].iterrows():
                alert = _build_alert_row(row, DOUBLE_COUNTRY_CARD_ABSENT_RULE, reason, config)
                alert["risk_level"] = risk_level
                alert["risk_score"] = risk_score
                alert["triggered_rules"] = triggered_rules
                alerts.append(alert)
        else:
            tnp_discarded_without_signal = int(absent_mask.sum())

    return alerts, {
        "double_country_excluded_pem10": excluded_pem10,
        "double_country_tnp_discarded_without_additional_signal": tnp_discarded_without_signal,
    }


def evaluate_transaction_rules(df: pd.DataFrame, config: dict[str, Any] | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    config = _merge_config(config)
    source_run = str(config.get("source_run", "UNKNOWN"))

    empty_columns = [
        "alert_id",
        "source_run",
        "transaction_id",
        "customer_hash",
        "transaction_datetime",
        "amount",
        "country_code",
        "pos_entry_mode",
        "has_pinblock",
        "merchant_rubro_proxy",
        "rule_code",
        "rule_name",
        "risk_level",
        "risk_score",
        "alert_reason",
        "triggered_rules",
        "status",
        "created_at",
    ]

    if df is None or df.empty:
        return pd.DataFrame(columns=empty_columns), {
            "source_run": source_run,
            "input_rows": 0,
            "normalized_rows": 0,
            "alerts_generated": 0,
            "deduplicated_alerts": 0,
            "alerts_by_rule_code": {},
            "alerts_by_risk_level": {},
            "config": config,
        }

    normalized = _prepare_dataframe(df)
    normalized["_date"] = normalized["transaction_datetime"].dt.date
    normalized["_hour"] = normalized["transaction_datetime"].dt.floor("h")
    normalized["_pem_norm"] = normalized["pos_entry_mode"].apply(_normalize_pos_entry_mode)
    normalized["_merchant_mcc"] = normalized["merchant_rubro_proxy"].apply(_normalize_mcc_code)
    normalized["_is_valid_country"] = normalized["country_code"].ne("UNKNOWN")

    alerts: list[dict[str, Any]] = []
    double_country_excluded_pem10 = 0
    double_country_tnp_discarded_without_additional_signal = 0

    country_groups = normalized[normalized["_is_valid_country"]].groupby(["customer_hash", "_date"], dropna=False)
    for _, group in country_groups:
        double_country_alerts, double_country_stats = _build_double_country_alerts(group, config)
        alerts.extend(double_country_alerts)
        double_country_excluded_pem10 += int(double_country_stats.get("double_country_excluded_pem10", 0))
        double_country_tnp_discarded_without_additional_signal += int(double_country_stats.get("double_country_tnp_discarded_without_additional_signal", 0))

    include_recurring = bool(config.get("include_recurring_in_general_velocity", False))
    general_velocity_mask = normalized["amount"].gt(0)
    if not include_recurring:
        general_velocity_mask &= normalized["_pem_norm"].ne(RECURRING_POS_ENTRY_MODE)
    for _, group in normalized.loc[general_velocity_mask].groupby(["customer_hash", "_hour"], dropna=False):
        if len(group) > int(config.get("max_tx_hour_default", DEFAULT_RULE_CONFIG["max_tx_hour_default"])):
            reason = f"More than {config.get('max_tx_hour_default', DEFAULT_RULE_CONFIG['max_tx_hour_default'])} transactions in one hour"
            for _, row in group.iterrows():
                alerts.append(_build_alert_row(row, VELOCITY_HOUR_RULE, reason, config))
    for _, group in normalized.loc[general_velocity_mask].groupby(["customer_hash", "_date"], dropna=False):
        if len(group) > int(config.get("max_tx_day_default", DEFAULT_RULE_CONFIG["max_tx_day_default"])):
            reason = f"More than {config.get('max_tx_day_default', DEFAULT_RULE_CONFIG['max_tx_day_default'])} transactions in one day"
            for _, row in group.iterrows():
                alerts.append(_build_alert_row(row, VELOCITY_DAY_RULE, reason, config))

    contactless_mask = normalized["_pem_norm"].eq("7") & normalized["has_pinblock"].eq(0) & normalized["amount"].gt(0)
    for _, group in normalized.loc[contactless_mask].groupby(["customer_hash", "_hour"], dropna=False):
        if len(group) > int(config.get("contactless_no_pin_hour", DEFAULT_RULE_CONFIG["contactless_no_pin_hour"])):
            reason = "Contactless without PIN: more than allowed operations per hour"
            for _, row in group.iterrows():
                alerts.append(_build_alert_row(row, CONTACTLESS_NO_PIN_HOUR_RULE, reason, config))
    for _, group in normalized.loc[contactless_mask].groupby(["customer_hash", "_date"], dropna=False):
        if len(group) > int(config.get("contactless_no_pin_day", DEFAULT_RULE_CONFIG["contactless_no_pin_day"])):
            reason = "Contactless without PIN: more than allowed operations per day"
            for _, row in group.iterrows():
                alerts.append(_build_alert_row(row, CONTACTLESS_NO_PIN_DAY_RULE, reason, config))

    internet_mask = normalized["_pem_norm"].isin(INTERNET_POS_ENTRY_MODES) & normalized["amount"].gt(0)
    for _, group in normalized.loc[internet_mask].groupby(["customer_hash", "_date"], dropna=False):
        if len(group) > int(config.get("internet_day", DEFAULT_RULE_CONFIG["internet_day"])):
            reason = "Internet / card-not-present velocity above daily threshold"
            for _, row in group.iterrows():
                alerts.append(_build_alert_row(row, INTERNET_VELOCITY_DAY_RULE, reason, config))

    magstripe_mask = normalized["_pem_norm"].eq("90") & normalized["amount"].gt(0)
    for _, group in normalized.loc[magstripe_mask].groupby(["customer_hash", "_hour"], dropna=False):
        if len(group) > int(config.get("magstripe_hour", DEFAULT_RULE_CONFIG["magstripe_hour"])):
            reason = "Magstripe velocity above hourly threshold"
            for _, row in group.iterrows():
                alerts.append(_build_alert_row(row, MAGSTRIPE_VELOCITY_HOUR_RULE, reason, config))

    atm_mask = normalized["_merchant_mcc"].isin(ATM_MCC_CODES) & normalized["amount"].gt(0)
    for _, group in normalized.loc[atm_mask].groupby(["customer_hash", "_hour"], dropna=False):
        if len(group) > int(config.get("cash_hour", DEFAULT_RULE_CONFIG["cash_hour"])):
            reason = "ATM/cash withdrawal velocity above hourly threshold"
            for _, row in group.iterrows():
                alerts.append(_build_alert_row(row, ATM_VELOCITY_HOUR_RULE, reason, config))
    for _, group in normalized.loc[atm_mask].groupby(["customer_hash", "_date"], dropna=False):
        if len(group) > int(config.get("cash_day", DEFAULT_RULE_CONFIG["cash_day"])):
            reason = "ATM/cash withdrawal velocity above daily threshold"
            for _, row in group.iterrows():
                alerts.append(_build_alert_row(row, ATM_VELOCITY_DAY_RULE, reason, config))

    strong_mask = normalized["_merchant_mcc"].isin(STRONG_HIGH_RISK_MCC_CODES) & normalized["amount"].ge(float(config.get("high_risk_amount_threshold", DEFAULT_RULE_CONFIG["high_risk_amount_threshold"])))
    for _, row in normalized.loc[strong_mask].iterrows():
        reason = f"High risk MCC with amount >= {config.get('high_risk_amount_threshold', DEFAULT_RULE_CONFIG['high_risk_amount_threshold'])}"
        alerts.append(_build_alert_row(row, STRONG_HIGH_RISK_MCC_RULE, reason, config))

    contextual_rows = normalized.loc[normalized["_merchant_mcc"].isin(CONTEXTUAL_HIGH_RISK_MCC_CODES)].copy()
    if not contextual_rows.empty:
        hour_counts = normalized.groupby(["customer_hash", "_hour"], dropna=False).size().rename("hour_count")
        day_counts = normalized.groupby(["customer_hash", "_date"], dropna=False).size().rename("day_count")
        contextual_rows = contextual_rows.join(hour_counts, on=["customer_hash", "_hour"])
        contextual_rows = contextual_rows.join(day_counts, on=["customer_hash", "_date"])
        contextual_rows["hour_count"] = contextual_rows["hour_count"].fillna(0).astype(int)
        contextual_rows["day_count"] = contextual_rows["day_count"].fillna(0).astype(int)
        for _, row in contextual_rows.iterrows():
            signals: list[str] = []
            if row["_pem_norm"] in INTERNET_POS_ENTRY_MODES:
                signals.append("pem_internet")
            if row["country_code"] != "BO":
                signals.append("country_non_bo")
            if row["amount"] >= float(config.get("contextual_high_risk_amount_threshold", DEFAULT_RULE_CONFIG["contextual_high_risk_amount_threshold"])):
                signals.append("amount_ge_contextual_threshold")
            if row["hour_count"] > int(config.get("max_tx_hour_default", DEFAULT_RULE_CONFIG["max_tx_hour_default"])):
                signals.append("hour_velocity")
            if row["day_count"] > int(config.get("max_tx_day_default", DEFAULT_RULE_CONFIG["max_tx_day_default"])):
                signals.append("day_velocity")
            if signals:
                risk_level, risk_score = _risk_from_context_signal_count(len(signals))
                definition = RULE_DEFINITIONS[CONTEXTUAL_HIGH_RISK_MCC_RULE]
                alerts.append(
                    {
                        "alert_id": None,
                        "source_run": source_run,
                        "transaction_id": row.get("transaction_id", "UNKNOWN"),
                        "customer_hash": row.get("customer_hash", "UNKNOWN"),
                        "transaction_datetime": row.get("transaction_datetime").isoformat() if pd.notna(row.get("transaction_datetime")) else None,
                        "amount": float(row.get("amount") or 0.0),
                        "country_code": row.get("country_code", "UNKNOWN"),
                        "pos_entry_mode": str(row.get("pos_entry_mode", "UNKNOWN")),
                        "has_pinblock": int(row.get("has_pinblock") or 0),
                        "merchant_rubro_proxy": row.get("merchant_rubro_proxy", "UNKNOWN"),
                        "rule_code": CONTEXTUAL_HIGH_RISK_MCC_RULE,
                        "rule_name": definition["rule_name"],
                        "risk_level": risk_level,
                        "risk_score": risk_score,
                        "alert_reason": f"Contextual high risk MCC with signals: {', '.join(signals)}",
                        "triggered_rules": CONTEXTUAL_HIGH_RISK_MCC_RULE,
                        "status": "NEW",
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    }
                )

    gambling_mask = normalized["_merchant_mcc"].isin(GAMBLING_MCC_CODES) & normalized["amount"].gt(0)
    for _, group in normalized.loc[gambling_mask].groupby(["customer_hash", "_hour"], dropna=False):
        hourly_hit = len(group) > int(config.get("max_tx_hour_default", DEFAULT_RULE_CONFIG["max_tx_hour_default"]))
        for _, row in group.iterrows():
            if row["amount"] >= float(config.get("gambling_amount_threshold", DEFAULT_RULE_CONFIG["gambling_amount_threshold"])) or hourly_hit:
                reason = "Gambling MCC with high amount or hourly velocity"
                alerts.append(_build_alert_row(row, GAMBLING_MCC_RULE, reason, config))

    jewelry_mask = normalized["_merchant_mcc"].isin(JEWELRY_MCC_CODES) & normalized["amount"].ge(float(config.get("jewelry_amount_threshold", DEFAULT_RULE_CONFIG["jewelry_amount_threshold"])))
    for _, row in normalized.loc[jewelry_mask].iterrows():
        reason = f"Jewelry MCC with amount >= {config.get('jewelry_amount_threshold', DEFAULT_RULE_CONFIG['jewelry_amount_threshold'])}"
        alerts.append(_build_alert_row(row, JEWELRY_HIGH_AMOUNT_RULE, reason, config))

    alerts_df = pd.DataFrame(alerts)
    if alerts_df.empty:
        alerts_df = pd.DataFrame(columns=empty_columns)

    dedupe_subset = ["transaction_id", "rule_code", "source_run"]
    before_dedupe = len(alerts_df)
    alerts_df = alerts_df.drop_duplicates(subset=dedupe_subset, keep="first").reset_index(drop=True)
    deduplicated_alerts = before_dedupe - len(alerts_df)

    if not alerts_df.empty:
        # Preserve any existing contextual signal details present in the `triggered_rules`
        # column when grouping multiple alerts for the same transaction. We build the
        # union of tokens found in the existing `triggered_rules` strings so that
        # contextual signals (e.g. "mcc_contextual", "amount_ge_contextual_threshold")
        # are not lost by grouping only on `rule_code`.
        def _aggregate_triggered_tokens(series: pd.Series) -> list[str]:
            tokens: list[str] = []
            for v in series.astype(str).fillna(""):
                parts = [p.strip() for p in v.split("|") if p.strip()]
                tokens.extend(parts)
            return list(dict.fromkeys(tokens))

        triggered_map = alerts_df.groupby("transaction_id", dropna=False)["triggered_rules"].apply(_aggregate_triggered_tokens).to_dict()

        def _format_triggered_rules(row: pd.Series) -> str:
            tokens = triggered_map.get(row["transaction_id"], [])
            primary_rule = str(row["rule_code"])
            remainder = [token for token in tokens if token != primary_rule]
            return "|".join([primary_rule, *remainder]) if remainder else primary_rule

        alerts_df["triggered_rules"] = alerts_df.apply(_format_triggered_rules, axis=1)
        alerts_df["alert_id"] = [f"{source_run}-{index + 1:06d}" for index in range(len(alerts_df))]
        alerts_df = alerts_df[empty_columns]

    summary = {
        "source_run": source_run,
        "input_rows": int(len(df)),
        "normalized_rows": int(len(normalized)),
        "alerts_generated": int(len(alerts_df)),
        "deduplicated_alerts": int(deduplicated_alerts),
        "alerts_by_rule_code": alerts_df["rule_code"].value_counts().to_dict() if not alerts_df.empty else {},
        "alerts_by_risk_level": alerts_df["risk_level"].value_counts().to_dict() if not alerts_df.empty else {},
        "double_country_excluded_pem10": int(double_country_excluded_pem10),
        "double_country_tnp_discarded_without_additional_signal": int(double_country_tnp_discarded_without_additional_signal),
        "config": config,
    }

    return alerts_df, summary

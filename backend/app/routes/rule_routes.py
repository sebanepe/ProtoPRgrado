from __future__ import annotations

import json
import logging
import math
import os
import re
import time
from datetime import datetime, timezone
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd
import numpy as np
from fastapi import APIRouter, HTTPException, Query, status, Depends
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.models.models import RuleAlertReview
from backend.app.schemas.rules import (
    AlertItem,
    AlertSummaryItem,
    AlertStatusUpdateRequest,
    AlertStatusUpdateResponse,
    AlertReviewHistoryResponse,
    PaginatedAlertSummaryResponse,
    PaginatedAlertsResponse,
    PaginatedReviewsResponse,
    RuleAnalyzeRequest,
    RuleAnalyzeResponse,
    RuleMetricsResponse,
    RunListItem,
)
from backend.app.services import rule_engine_service, rule_alert_review_service


router = APIRouter(prefix="/api/rules", tags=["rules", "alert_rules", "fraud_rules"])
logger = logging.getLogger(__name__)


def _processed_dir() -> Path:
    return Path(os.environ.get("PROJECT_PROCESSED_DIR") or rule_engine_service.PROJECT_PROCESSED_DIR)


def _to_posix_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except Exception:
        return path.as_posix()


def _run_token_from_run_id(run_id: str) -> str:
    match = re.search(r"(\d+)$", str(run_id))
    if match:
        return match.group(1)
    match = re.search(r"run_(\d+)", str(run_id))
    if match:
        return match.group(1)
    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="run_id must contain a numeric suffix")


def _processed_file(run_id: str) -> Path:
    return _processed_dir() / f"{run_id}.csv"


def _alerts_file(run_id: str) -> Path:
    return _processed_dir() / f"alerts_run_{_run_token_from_run_id(run_id)}.csv"


def _summary_file(run_id: str) -> Path:
    return _processed_dir() / f"alerts_summary_run_{_run_token_from_run_id(run_id)}.csv"


def _report_file(run_id: str) -> Path:
    return _processed_dir() / f"rules_report_run_{_run_token_from_run_id(run_id)}.md"


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"File not found: {path.name}")
    return pd.read_csv(path)


def _read_json_file(path: Path) -> Optional[dict[str, Any]]:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _write_json_file(path: Path, payload: dict[str, Any]) -> None:
    try:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        return


def _normalize_filter_value(value: Optional[str], *, uppercase: bool = False) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized or normalized.lower() in {"nan", "none", "null"}:
        return None
    return normalized.upper() if uppercase else normalized


def _sanitize_for_json(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, np.generic):
        return _sanitize_for_json(value.item())

    if isinstance(value, bool):
        return value

    if isinstance(value, int):
        return int(value)

    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None

    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.isoformat()

    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.isoformat()
        return value.astimezone(timezone.utc).isoformat()

    if isinstance(value, dict):
        return {str(key): _sanitize_for_json(item) for key, item in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_sanitize_for_json(item) for item in value]

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    return value


def _sanitize_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: _sanitize_for_json(value) for key, value in record.items()} for record in records]


def _clean_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []
    cleaned = df.where(pd.notnull(df), None)
    return cleaned.to_dict(orient="records")


def _latest_review_status_map(db: Session, run_id: str, *, is_summary: bool) -> Dict[str, str]:
    query = db.query(RuleAlertReview).filter(RuleAlertReview.source_run == run_id)
    if is_summary:
        query = query.filter(RuleAlertReview.summary_alert_id.isnot(None))
    else:
        query = query.filter(RuleAlertReview.alert_id.isnot(None))

    reviews = query.order_by(RuleAlertReview.reviewed_at.desc(), RuleAlertReview.id.desc()).all()
    status_map: Dict[str, str] = {}
    for review in reviews:
        key = review.summary_alert_id if is_summary else review.alert_id
        if not key:
            continue
        key_str = str(key).strip()
        if key_str and key_str not in status_map:
            status_map[key_str] = str(review.new_status).strip().upper()
    return status_map


def _review_signature(db: Session, run_id: str) -> str:
    count, latest_reviewed_at = (
        db.query(func.count(RuleAlertReview.id), func.max(RuleAlertReview.reviewed_at))
        .filter(RuleAlertReview.source_run == run_id)
        .one()
    )
    latest_token = latest_reviewed_at.isoformat() if latest_reviewed_at else "none"
    return f"{int(count or 0)}:{latest_token}"


def _merge_statuses(df: pd.DataFrame, db: Session, run_id: str, *, is_summary: bool) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    result = df.copy()
    status_map = _latest_review_status_map(db, run_id, is_summary=is_summary)

    status_column = "status"
    if status_column not in result.columns:
        result[status_column] = "NEW"
    else:
        result[status_column] = result[status_column].fillna("NEW").astype(str).str.strip().str.upper()

    key_column = "summary_alert_id" if is_summary else "alert_id"
    if key_column in result.columns and status_map:
        keys = result[key_column].fillna("").astype(str).str.strip()
        overrides = keys.map(status_map)
        result[status_column] = overrides.fillna(result[status_column]).astype(str).str.strip().str.upper()

    return result


def _text_tokens(value: Any) -> list[str]:
    if value is None:
        return []
    raw_value = str(value).strip().upper()
    if not raw_value or raw_value in {"NAN", "NONE", "NULL"}:
        return []
    return [token for token in re.split(r"[^A-Z0-9]+", raw_value) if token]


def _series_contains_rule_code(series: pd.Series, token: str) -> pd.Series:
    normalized_token = _normalize_filter_value(token, uppercase=True)
    if normalized_token is None:
        return pd.Series([True] * len(series), index=series.index)
    normalized_series = series.fillna("").astype(str).str.strip().str.upper()
    exact_mask = normalized_series == normalized_token
    if exact_mask.any():
        return exact_mask
    return normalized_series.str.contains(re.escape(normalized_token), na=False, regex=True)


def _series_contains_token(series: pd.Series, token: str) -> pd.Series:
    normalized_token = _normalize_filter_value(token, uppercase=True)
    if normalized_token is None:
        return pd.Series([True] * len(series), index=series.index)
    return series.fillna("").astype(str).apply(lambda value: normalized_token in _text_tokens(value))


def _load_alerts_df_for_run(run_id: str) -> pd.DataFrame:
    alerts_path = _alerts_file(run_id)
    if not alerts_path.exists():
        return pd.DataFrame()
    return _load_csv(alerts_path)


def _build_mcc_lookup(alerts_df: pd.DataFrame) -> dict[tuple[Any, Any, Any, Any], dict[str, int]]:
    if alerts_df.empty:
        return {}

    prepared = alerts_df.copy()
    if "transaction_datetime" in prepared.columns:
        prepared["_summary_date"] = pd.to_datetime(prepared["transaction_datetime"], errors="coerce", utc=True).dt.date
    else:
        prepared["_summary_date"] = None

    if "merchant_rubro_proxy" in prepared.columns:
        prepared["_merchant_rubro_proxy"] = prepared["merchant_rubro_proxy"].fillna("UNKNOWN").astype(str).str.strip().replace({"": "UNKNOWN"})
    else:
        prepared["_merchant_rubro_proxy"] = "UNKNOWN"

    if not {"source_run", "customer_hash", "rule_code"}.issubset(prepared.columns):
        return {}

    lookup: dict[tuple[Any, Any, Any, Any], dict[str, int]] = {}
    for key, group in prepared.groupby(["source_run", "customer_hash", "rule_code", "_summary_date"], dropna=False, sort=False):
        counts = (
            group["_merchant_rubro_proxy"]
            .astype(str)
            .map(lambda value: value.strip().upper())
            .replace({"": "UNKNOWN"})
            .value_counts()
            .to_dict()
        )
        normalized_key = (
            _normalize_filter_value(key[0]),
            _normalize_filter_value(key[1]),
            _normalize_filter_value(key[2], uppercase=True),
            key[3],
        )
        lookup[normalized_key] = {str(code): int(count) for code, count in counts.items() if str(code).strip().upper() not in {"NAN", "NONE", "NULL"}}
    return lookup


def _enrich_summary_with_alerts(summary_df: pd.DataFrame, alerts_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty or alerts_df.empty:
        return summary_df.copy()

    result = summary_df.copy()
    alerts_work = alerts_df.copy()
    if "transaction_datetime" in alerts_work.columns:
        alerts_work["_summary_date"] = pd.to_datetime(alerts_work["transaction_datetime"], errors="coerce", utc=True).dt.date
    else:
        alerts_work["_summary_date"] = None

    if "merchant_rubro_proxy" in alerts_work.columns:
        alerts_work["_merchant_rubro_proxy"] = alerts_work["merchant_rubro_proxy"].fillna("UNKNOWN").astype(str).str.strip().replace({"": "UNKNOWN"})
    else:
        alerts_work["_merchant_rubro_proxy"] = "UNKNOWN"

    tx_lookup: dict[str, Any] = {}
    if "transaction_id" in alerts_work.columns:
        tx_lookup = alerts_work.loc[alerts_work["transaction_id"].notna(), ["transaction_id", "_summary_date"]].copy()
        tx_lookup["transaction_id"] = tx_lookup["transaction_id"].astype(str)
        tx_lookup = tx_lookup.set_index("transaction_id")["_summary_date"].to_dict()

    mcc_lookup = _build_mcc_lookup(alerts_work)

    if "merchant_rubro_proxy" not in result.columns:
        result["merchant_rubro_proxy"] = None
    if "merchant_rubro_values" not in result.columns:
        result["merchant_rubro_values"] = None
    if "top_merchant_rubro_proxy" not in result.columns:
        result["top_merchant_rubro_proxy"] = None

    enriched_rows: list[dict[str, Any]] = []
    for _, row in result.iterrows():
        row_dict = row.to_dict()
        existing_proxy = _normalize_filter_value(row_dict.get("merchant_rubro_proxy"), uppercase=True)
        existing_values = _normalize_filter_value(row_dict.get("merchant_rubro_values"), uppercase=True)
        existing_top = _normalize_filter_value(row_dict.get("top_merchant_rubro_proxy"), uppercase=True)
        if existing_proxy and existing_proxy != "UNKNOWN":
            row_dict["merchant_rubro_proxy"] = existing_proxy
            row_dict["merchant_rubro_values"] = existing_values or row_dict.get("merchant_rubro_values")
            row_dict["top_merchant_rubro_proxy"] = existing_top or existing_proxy
            enriched_rows.append(row_dict)
            continue

        source_run = _normalize_filter_value(row_dict.get("source_run"))
        customer_hash = _normalize_filter_value(row_dict.get("customer_hash"))
        rule_code = _normalize_filter_value(row_dict.get("rule_code"), uppercase=True)
        representative_transaction_id = _normalize_filter_value(row_dict.get("representative_transaction_id"))
        summary_date = tx_lookup.get(str(representative_transaction_id)) if representative_transaction_id else None
        counts = mcc_lookup.get((source_run, customer_hash, rule_code, summary_date)) or mcc_lookup.get((source_run, customer_hash, rule_code, None)) or {}

        sorted_counts = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        mcc_values = [str(code).strip().upper() for code, _count in sorted_counts if str(code).strip().upper() not in {"UNKNOWN", "NAN", "NONE", "NULL"}]
        if not mcc_values and counts.get("UNKNOWN"):
            mcc_values = ["UNKNOWN"]

        top_mcc = mcc_values[0] if mcc_values else None
        if len(mcc_values) == 1:
            row_dict["merchant_rubro_proxy"] = mcc_values[0]
        elif top_mcc is not None:
            row_dict["merchant_rubro_proxy"] = top_mcc
        row_dict["top_merchant_rubro_proxy"] = top_mcc
        row_dict["merchant_rubro_values"] = "|".join(mcc_values) if mcc_values else None
        if row_dict.get("merchant_rubro_proxy") in {None, "", "UNKNOWN"} and top_mcc is not None:
            row_dict["merchant_rubro_proxy"] = top_mcc
        enriched_rows.append(row_dict)

    return pd.DataFrame(enriched_rows)


def _load_summary_for_run(run_id: str, db: Session) -> pd.DataFrame:
    summary_df = _load_csv(_summary_file(run_id))
    summary_df = _merge_statuses(summary_df, db, run_id, is_summary=True)
    return _enrich_summary_with_alerts(summary_df, _load_alerts_df_for_run(run_id))


def _match_summary_filter(df: pd.DataFrame, column: str, value: Optional[str]) -> pd.Series:
    normalized_value = _normalize_filter_value(value, uppercase=column in {"risk_level", "status", "country_code", "rule_code", "merchant_rubro_proxy"})
    if normalized_value is None:
        return pd.Series([True] * len(df), index=df.index)

    if column == "rule_code":
        if column not in df.columns:
            return pd.Series([False] * len(df), index=df.index)
        return _series_contains_rule_code(df[column], normalized_value)

    if column == "country_code":
        if "country_code" in df.columns and df["country_code"].notna().any():
            return df["country_code"].fillna("").astype(str).str.strip().str.upper() == normalized_value
        if "countries_detected" in df.columns:
            return _series_contains_token(df["countries_detected"], normalized_value)
        return pd.Series([False] * len(df), index=df.index)

    if column == "merchant_rubro_proxy":
        available_columns = [candidate for candidate in ["merchant_rubro_proxy", "merchant_rubro_values", "top_merchant_rubro_proxy"] if candidate in df.columns]
        if not available_columns:
            return pd.Series([False] * len(df), index=df.index)
        combined_mask = pd.Series([False] * len(df), index=df.index)
        for candidate in available_columns:
            combined_mask = combined_mask | _series_contains_token(df[candidate], normalized_value)
        return combined_mask

    if column not in df.columns:
        return pd.Series([False] * len(df), index=df.index)

    series = df[column].fillna("").astype(str).str.strip()
    if column in {"risk_level", "status"}:
        series = series.str.upper()
    return series == normalized_value


def _sorted_option_values(series: pd.Series, *, limit: Optional[int] = None) -> list[str]:
    if series.empty:
        return []
    normalized = series.fillna("").astype(str).map(lambda value: value.strip())
    normalized = normalized[~normalized.str.lower().isin({"", "nan", "none", "null"})]
    if normalized.empty:
        return []
    counts = normalized.value_counts()
    items = sorted(counts.items(), key=lambda item: (-int(item[1]), str(item[0]).upper()))
    values = [str(key) for key, _count in items]
    return values[:limit] if limit is not None else values


def _collect_option_values(series: pd.Series, *, limit: Optional[int] = None, tokenize: bool = False, uppercase: bool = True) -> list[str]:
    if series.empty:
        return []

    counts: Counter[str] = Counter()
    for raw_value in series.tolist():
        if raw_value is None:
            continue
        if isinstance(raw_value, float) and pd.isna(raw_value):
            continue

        values = _text_tokens(raw_value) if tokenize else [str(raw_value).strip()]
        for value in values:
            normalized = str(value).strip()
            if not normalized or normalized.lower() in {"nan", "none", "null"}:
                continue
            if uppercase:
                normalized = normalized.upper()
            counts[normalized] += 1

    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    values = [key for key, _count in ordered]
    return values[:limit] if limit is not None else values


def _country_options_from_summary(df: pd.DataFrame) -> list[str]:
    values: list[str] = []
    if "country_code" in df.columns and df["country_code"].notna().any():
        values.extend(_collect_option_values(df["country_code"], tokenize=True, uppercase=True))
    if "countries_detected" in df.columns and df["countries_detected"].notna().any():
        values.extend(token for token in _collect_option_values(df["countries_detected"], tokenize=True, uppercase=True) if len(token) == 2 or token.isalpha())
    return _sorted_option_values(pd.Series(values))


def _merchant_options_from_summary(df: pd.DataFrame) -> list[str]:
    tokens: list[str] = []
    for candidate in ["merchant_rubro_proxy", "top_merchant_rubro_proxy", "merchant_rubro_values"]:
        if candidate not in df.columns or not df[candidate].notna().any():
            continue
        tokens.extend(_collect_option_values(df[candidate], tokenize=True, uppercase=True))
    return _sorted_option_values(pd.Series(tokens))


def _summary_filter_options(df: pd.DataFrame) -> dict[str, list[str]]:
    summary_rule_codes = _collect_option_values(
        df["rule_code"] if "rule_code" in df.columns else pd.Series(dtype=str),
        uppercase=True,
    )
    summary_rule_codes = sorted(summary_rule_codes)

    return {
        "rule_code": summary_rule_codes,
        "risk_level": _sorted_option_values(df["risk_level"] if "risk_level" in df.columns else pd.Series(dtype=str)),
        "status": _collect_option_values(df["status"] if "status" in df.columns else pd.Series(dtype=str), uppercase=True),
        "country_code": _country_options_from_summary(df),
        "merchant_rubro_proxy": _merchant_options_from_summary(df),
        "customer_hash": _collect_option_values(df["customer_hash"] if "customer_hash" in df.columns else pd.Series(dtype=str), limit=100, uppercase=False),
    }


def _summary_filter_options_cache_file(run_id: str) -> Path:
    return _processed_dir() / f"summary_filter_options_run_{_run_token_from_run_id(run_id)}.json"


def _summary_filter_options_signature(run_id: str, db: Session) -> dict[str, Any]:
    summary_path = _summary_file(run_id)
    alerts_path = _alerts_file(run_id)
    return {
        "cache_version": 2,
        "summary_mtime_ns": summary_path.stat().st_mtime_ns if summary_path.exists() else None,
        "alerts_mtime_ns": alerts_path.stat().st_mtime_ns if alerts_path.exists() else None,
        "review_signature": _review_signature(db, run_id),
    }


def _build_summary_filter_options(run_id: str, db: Session) -> tuple[dict[str, Any], str]:
    start_time = time.perf_counter()
    summary_df = _load_summary_for_run(run_id, db)
    summary_options = _summary_filter_options(summary_df)
    source = "summary"

    review_statuses = _collect_option_values(
        pd.Series(list(_latest_review_status_map(db, run_id, is_summary=True).values()) + list(_latest_review_status_map(db, run_id, is_summary=False).values())),
        uppercase=True,
    )
    combined_statuses = [value for value in ["NEW", "IN_REVIEW", "DISMISSED", "FALSE_POSITIVE", "CONFIRMED_FRAUD"] if value not in review_statuses]
    summary_options["status"] = _collect_option_values(
        pd.Series(summary_options["status"] + review_statuses + combined_statuses),
        uppercase=True,
    )

    payload = {"run_id": run_id, **summary_options}
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    logger.info(
        "summary-filter-options run_id=%s source=%s elapsed_ms=%.1f rules=%d countries=%d mcc=%d",
        run_id,
        source,
        elapsed_ms,
        len(payload["rule_code"]),
        len(payload["country_code"]),
        len(payload["merchant_rubro_proxy"]),
    )
    return payload, source


def _apply_filters(df: pd.DataFrame, filters: Dict[str, Optional[str]], *, allowed_missing: Iterable[str] = ()) -> pd.DataFrame:
    result = df.copy()
    for column, value in filters.items():
        normalized_value = _normalize_filter_value(value, uppercase=column in {"risk_level", "status", "country_code", "rule_code", "merchant_rubro_proxy"})
        if normalized_value is None:
            continue
        if column not in result.columns:
            if column in allowed_missing:
                result = result.loc[_match_summary_filter(result, column, normalized_value)]
                continue
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Filter not available in file: {column}")

        result = result.loc[_match_summary_filter(result, column, normalized_value)]
    return result


def _paginate(df: pd.DataFrame, page: int, page_size: int) -> tuple[pd.DataFrame, int, int, int]:
    safe_page = max(1, page)
    safe_page_size = min(max(1, page_size), 200)
    total_items = int(len(df))
    total_pages = int(math.ceil(total_items / safe_page_size)) if total_items else 0
    start = (safe_page - 1) * safe_page_size
    end = start + safe_page_size
    return df.iloc[start:end], safe_page, safe_page_size, total_pages


def _run_file_state(run_id: str) -> RunListItem:
    source = _processed_file(run_id)
    token = _run_token_from_run_id(run_id)
    alerts = _processed_dir() / f"alerts_run_{token}.csv"
    summary = _processed_dir() / f"alerts_summary_run_{token}.csv"
    report = _processed_dir() / f"rules_report_run_{token}.md"
    stat = source.stat()
    return RunListItem(
        run_id=run_id,
        filename=source.name,
        path=_to_posix_path(source),
        created_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        size_bytes=stat.st_size,
        has_alerts=alerts.exists(),
        has_summary=summary.exists(),
        has_report=report.exists(),
        alerts_file=alerts.name if alerts.exists() else None,
        summary_file=summary.name if summary.exists() else None,
        report_file=report.name if report.exists() else None,
    )


def _sort_run_path(path: Path) -> int:
    return int(_run_token_from_run_id(path.stem))


@router.get("/preprocessed-runs", response_model=list[RunListItem])
def list_preprocessed_runs() -> list[RunListItem]:
    processed_dir = _processed_dir()
    if not processed_dir.exists():
        return []

    runs: list[RunListItem] = []
    for path in sorted(processed_dir.glob("preprocessed_run_*.csv"), key=_sort_run_path):
        try:
            runs.append(_run_file_state(path.stem))
        except FileNotFoundError:
            continue
    return runs


@router.post("/analyze", response_model=RuleAnalyzeResponse)
def analyze_rules(request: RuleAnalyzeRequest) -> RuleAnalyzeResponse:
    source_path = _processed_file(request.preprocessed_run_id)
    if not source_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Preprocessed run not found: {request.preprocessed_run_id}",
        )

    run_token = _run_token_from_run_id(request.preprocessed_run_id)
    alerts_path = _processed_dir() / f"alerts_run_{run_token}.csv"
    summary_path = _processed_dir() / f"alerts_summary_run_{run_token}.csv"
    report_path = _processed_dir() / f"rules_report_run_{run_token}.md"

    if not request.force and alerts_path.exists() and summary_path.exists() and report_path.exists():
        source_df = pd.read_csv(source_path)
        alerts_df = pd.read_csv(alerts_path)
        summary_df = pd.read_csv(summary_path)
        return RuleAnalyzeResponse(
            status="ALREADY_EXISTS",
            source_run=request.preprocessed_run_id,
            total_transactions=int(len(source_df)),
            alerts_file=alerts_path.name,
            summary_file=summary_path.name,
            report_file=report_path.name,
            total_alerts=int(len(alerts_df)),
            total_summary_alerts=int(len(summary_df)),
            message="Artifacts already exist; rerun skipped.",
        )

    result = rule_engine_service.generate_alerts_from_preprocessed_csv(
        str(source_path),
        config={**(request.config or {}), "source_run": request.preprocessed_run_id},
    )
    source_df = pd.read_csv(source_path)
    return RuleAnalyzeResponse(
        status="COMPLETED",
        source_run=request.preprocessed_run_id,
        total_transactions=int(len(source_df)),
        alerts_file=Path(result["alerts_path"]).name,
        summary_file=Path(result["summary_path"]).name,
        report_file=Path(result["report_path"]).name,
        total_alerts=int(result["summary"].get("alerts_generated", 0)),
        total_summary_alerts=int(len(result.get("summary_df", []))),
        message="Rule engine executed successfully.",
    )


@router.get("/summary", response_model=PaginatedAlertSummaryResponse)
def get_summary(
    run_id: str = Query(...),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1),
    rule_code: Optional[str] = Query(None),
    risk_level: Optional[str] = Query(None),
    status_filter: Optional[str] = Query(None, alias="status"),
    country_code: Optional[str] = Query(None),
    merchant_rubro_proxy: Optional[str] = Query(None),
    customer_hash: Optional[str] = Query(None),
    db: Session = Depends(get_db),
) -> PaginatedAlertSummaryResponse:
    df = _load_summary_for_run(run_id, db)
    total_before_filter = int(len(df))
    filters = {
        "rule_code": rule_code,
        "risk_level": risk_level,
        "status": status_filter,
        "country_code": country_code,
        "merchant_rubro_proxy": merchant_rubro_proxy,
        "customer_hash": customer_hash,
    }
    normalized_filters = {
        key: _normalize_filter_value(
            value,
            uppercase=key in {"risk_level", "status", "country_code", "rule_code", "merchant_rubro_proxy"},
        )
        for key, value in filters.items()
    }
    normalized_rule_code = normalized_filters.get("rule_code")

    if normalized_rule_code:
        logger.info(
            "summary rule filter requested run_id=%s rule_code=%s total_before=%d",
            run_id,
            normalized_rule_code,
            total_before_filter,
        )

    filtered = _apply_filters(df, normalized_filters, allowed_missing={"country_code", "merchant_rubro_proxy"})
    total_after_filter = int(len(filtered))

    if normalized_rule_code:
        if total_after_filter == 0:
            available_rules: list[str] = []
            if "rule_code" in df.columns:
                available_rules = sorted(_collect_option_values(df["rule_code"], uppercase=True))
            logger.info(
                "summary rule filter result run_id=%s rule_code=%s total_after=%d available_rules=%s",
                run_id,
                normalized_rule_code,
                total_after_filter,
                available_rules,
            )
        else:
            logger.info(
                "summary rule filter result run_id=%s rule_code=%s total_after=%d",
                run_id,
                normalized_rule_code,
                total_after_filter,
            )

    page_df, safe_page, safe_page_size, total_pages = _paginate(filtered, page, page_size)

    records = _sanitize_records(_clean_records(page_df))
    
    return PaginatedAlertSummaryResponse(
        run_id=run_id,
        page=safe_page,
        page_size=safe_page_size,
        total_items=int(len(filtered)),
        total_pages=total_pages,
        items=[AlertSummaryItem(**row) for row in records],
    )


@router.get("/summary-filter-options")
def get_summary_filter_options(
    run_id: str = Query(...),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    summary_path = _summary_file(run_id)
    if not summary_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Summary not found for run_id={run_id}")

    cache_file = _summary_filter_options_cache_file(run_id)
    signature = _summary_filter_options_signature(run_id, db)
    cached = _read_json_file(cache_file)
    if cached and cached.get("signature") == signature and isinstance(cached.get("payload"), dict):
        payload = cached["payload"]
        logger.info(
            "summary-filter-options run_id=%s source=cache rules=%d countries=%d mcc=%d",
            run_id,
            len(payload.get("rule_code", [])),
            len(payload.get("country_code", [])),
            len(payload.get("merchant_rubro_proxy", [])),
        )
        return payload

    payload, source = _build_summary_filter_options(run_id, db)
    _write_json_file(cache_file, {"signature": signature, "source": source, "payload": payload})
    return payload


@router.get("/alerts", response_model=PaginatedAlertsResponse)
def get_alerts(
    run_id: str = Query(...),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1),
    rule_code: Optional[str] = Query(None),
    risk_level: Optional[str] = Query(None),
    status_filter: Optional[str] = Query(None, alias="status"),
    country_code: Optional[str] = Query(None),
    pos_entry_mode: Optional[str] = Query(None),
    merchant_rubro_proxy: Optional[str] = Query(None),
    customer_hash: Optional[str] = Query(None),
    transaction_id: Optional[str] = Query(None),
    db: Session = Depends(get_db),
) -> PaginatedAlertsResponse:
    df = _load_csv(_alerts_file(run_id))
    df = _merge_statuses(df, db, run_id, is_summary=False)
    filters = {
        "rule_code": rule_code,
        "risk_level": risk_level,
        "status": status_filter,
        "country_code": country_code,
        "pos_entry_mode": pos_entry_mode,
        "merchant_rubro_proxy": merchant_rubro_proxy,
        "customer_hash": customer_hash,
        "transaction_id": transaction_id,
    }
    normalized_filters = {key: _normalize_filter_value(value, uppercase=key in {"risk_level", "status", "country_code"}) for key, value in filters.items()}
    filtered = _apply_filters(df, normalized_filters)
    page_df, safe_page, safe_page_size, total_pages = _paginate(filtered, page, page_size)

    records = _sanitize_records(_clean_records(page_df))
    
    return PaginatedAlertsResponse(
        run_id=run_id,
        page=safe_page,
        page_size=safe_page_size,
        total_items=int(len(filtered)),
        total_pages=total_pages,
        items=[AlertItem(**row) for row in records],
    )


@router.get("/alerts/{alert_id}", response_model=AlertItem)
def get_alert_detail(alert_id: str, run_id: Optional[str] = Query(None), db: Session = Depends(get_db)) -> AlertItem:
    if run_id:
        candidates = [_alerts_file(run_id)]
    else:
        candidates = sorted(_processed_dir().glob("alerts_run_*.csv"), key=lambda item: item.stat().st_mtime, reverse=True)

    for path in candidates:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "alert_id" not in df.columns:
            continue
        match = df.loc[df["alert_id"].astype(str) == str(alert_id)]
        if not match.empty:
            record = _clean_records(match.head(1))[0]
            # Determine run_id if not provided
            if not run_id:
                run_id = path.stem.replace("alerts_run_", "preprocessed_run_")
            # Merge with review status
            merged_record = rule_alert_review_service.merge_status_with_item(db, run_id, record, is_summary=False)
            return AlertItem(**merged_record)

    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Alert not found")


@router.get("/report")
def get_report(run_id: str = Query(...)) -> Dict[str, Any]:
    path = _report_file(run_id)
    if not path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Report not found for run_id={run_id}")
    return {"run_id": run_id, "report": path.read_text(encoding="utf-8")}


@router.get("/metrics", response_model=RuleMetricsResponse)
def get_metrics(run_id: str = Query(...)) -> RuleMetricsResponse:
    alerts_df = _load_csv(_alerts_file(run_id))
    summary_path = _summary_file(run_id)
    summary_df = _load_csv(summary_path) if summary_path.exists() else pd.DataFrame()

    def _counts(df: pd.DataFrame, column: str) -> Dict[str, int]:
        if df.empty or column not in df.columns:
            return {}
        return {str(key): int(value) for key, value in df[column].astype(str).value_counts().to_dict().items()}

    top_customers: list[dict[str, Any]] = []
    if not alerts_df.empty and "customer_hash" in alerts_df.columns:
        customer_counts = (
            alerts_df.groupby(alerts_df["customer_hash"].astype(str))
            .size()
            .sort_values(ascending=False)
            .head(10)
        )
        top_customers = [{"customer_hash": str(index), "alert_count": int(count)} for index, count in customer_counts.items()]

    return RuleMetricsResponse(
        run_id=run_id,
        total_alerts=int(len(alerts_df)),
        total_summary_alerts=int(len(summary_df)),
        alerts_by_rule=_counts(alerts_df, "rule_code"),
        alerts_by_risk_level=_counts(alerts_df, "risk_level"),
        alerts_by_mcc=_counts(alerts_df, "merchant_rubro_proxy"),
        alerts_by_country=_counts(alerts_df, "country_code"),
        top_customers=top_customers,
    )


# ============================================================
# PHASE B.3: Human Review and Alert Status Management
# ============================================================


@router.patch("/alerts/{alert_id}/status", response_model=AlertStatusUpdateResponse)
def update_alert_status(
    alert_id: str,
    request: AlertStatusUpdateRequest,
    db: Session = Depends(get_db),
) -> AlertStatusUpdateResponse:
    """
    Update the status of a detailed alert.
    
    Allowed statuses: NEW, IN_REVIEW, DISMISSED, FALSE_POSITIVE, CONFIRMED_FRAUD
    Status changes are recorded in DB, original CSV files are never modified.
    """
    try:
        result = rule_alert_review_service.create_or_update_alert_review(
            db,
            source_run=request.run_id,
            rule_code="",  # Will be populated from CSV if needed
            new_status=request.new_status,
            alert_id=alert_id,
            analyst_notes=request.analyst_notes,
            reviewed_by_id=None,  # Can be integrated with auth system
        )
        return AlertStatusUpdateResponse(
            status="OK",
            alert_id=result["alert_id"],
            run_id=result["source_run"],
            new_status=result["new_status"],
            reviewed_at=result["reviewed_at"],
            message="Alert status updated successfully.",
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.patch("/summary/{summary_alert_id}/status", response_model=AlertStatusUpdateResponse)
def update_summary_alert_status(
    summary_alert_id: str,
    request: AlertStatusUpdateRequest,
    db: Session = Depends(get_db),
) -> AlertStatusUpdateResponse:
    """
    Update the status of a summary (grouped) alert.
    
    Allowed statuses: NEW, IN_REVIEW, DISMISSED, FALSE_POSITIVE, CONFIRMED_FRAUD
    Status changes are recorded in DB, original CSV files are never modified.
    """
    try:
        result = rule_alert_review_service.create_or_update_alert_review(
            db,
            source_run=request.run_id,
            rule_code="",  # Will be populated from CSV if needed
            new_status=request.new_status,
            summary_alert_id=summary_alert_id,
            analyst_notes=request.analyst_notes,
            reviewed_by_id=None,  # Can be integrated with auth system
        )
        return AlertStatusUpdateResponse(
            status="OK",
            summary_alert_id=result["summary_alert_id"],
            run_id=result["source_run"],
            new_status=result["new_status"],
            reviewed_at=result["reviewed_at"],
            message="Summary alert status updated successfully.",
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/alerts/{alert_id}/history", response_model=AlertReviewHistoryResponse)
def get_alert_history(
    alert_id: str,
    run_id: str = Query(...),
    db: Session = Depends(get_db),
) -> AlertReviewHistoryResponse:
    """Get review history for a detailed alert."""
    try:
        history = rule_alert_review_service.get_alert_review_history(db, run_id, alert_id=alert_id)
        return AlertReviewHistoryResponse(
            alert_id=alert_id,
            run_id=run_id,
            history=[
                {
                    "id": h["id"],
                    "source_run": h["source_run"],
                    "alert_id": h["alert_id"],
                    "summary_alert_id": h["summary_alert_id"],
                    "rule_code": h["rule_code"],
                    "previous_status": h["previous_status"],
                    "new_status": h["new_status"],
                    "analyst_notes": h["analyst_notes"],
                    "reviewed_by_id": h["reviewed_by_id"],
                    "reviewed_at": h["reviewed_at"],
                }
                for h in history
            ],
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/summary/{summary_alert_id}/history", response_model=AlertReviewHistoryResponse)
def get_summary_alert_history(
    summary_alert_id: str,
    run_id: str = Query(...),
    db: Session = Depends(get_db),
) -> AlertReviewHistoryResponse:
    """Get review history for a summary (grouped) alert."""
    try:
        history = rule_alert_review_service.get_alert_review_history(db, run_id, summary_alert_id=summary_alert_id)
        return AlertReviewHistoryResponse(
            summary_alert_id=summary_alert_id,
            run_id=run_id,
            history=[
                {
                    "id": h["id"],
                    "source_run": h["source_run"],
                    "alert_id": h["alert_id"],
                    "summary_alert_id": h["summary_alert_id"],
                    "rule_code": h["rule_code"],
                    "previous_status": h["previous_status"],
                    "new_status": h["new_status"],
                    "analyst_notes": h["analyst_notes"],
                    "reviewed_by_id": h["reviewed_by_id"],
                    "reviewed_at": h["reviewed_at"],
                }
                for h in history
            ],
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/reviews", response_model=PaginatedReviewsResponse)
def list_reviews(
    run_id: str = Query(...),
    status: Optional[str] = Query(None, description="Filter by status: NEW, IN_REVIEW, DISMISSED, FALSE_POSITIVE, CONFIRMED_FRAUD"),
    rule_code: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1),
    db: Session = Depends(get_db),
) -> PaginatedReviewsResponse:
    """List all reviews for a run with optional filtering."""
    try:
        result = rule_alert_review_service.list_all_reviews(
            db,
            source_run=run_id,
            status=status,
            rule_code=rule_code,
            page=page,
            page_size=page_size,
        )
        return PaginatedReviewsResponse(
            run_id=run_id,
            page=result["page"],
            page_size=result["page_size"],
            total_items=result["total_items"],
            total_pages=result["total_pages"],
            items=[
                {
                    "id": item["id"],
                    "source_run": item["source_run"],
                    "alert_id": item["alert_id"],
                    "summary_alert_id": item["summary_alert_id"],
                    "rule_code": item["rule_code"],
                    "previous_status": item.get("previous_status"),
                    "new_status": item["new_status"],
                    "analyst_notes": item["analyst_notes"],
                    "reviewed_by_id": item["reviewed_by_id"],
                    "reviewed_at": item["reviewed_at"],
                }
                for item in result["items"]
            ],
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
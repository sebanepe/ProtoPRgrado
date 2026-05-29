from __future__ import annotations

import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query, status

from backend.app.schemas.rules import (
    AlertItem,
    AlertSummaryItem,
    PaginatedAlertSummaryResponse,
    PaginatedAlertsResponse,
    RuleAnalyzeRequest,
    RuleAnalyzeResponse,
    RuleMetricsResponse,
    RunListItem,
)
from backend.app.services import rule_engine_service


router = APIRouter(prefix="/api/rules", tags=["rules", "alert_rules", "fraud_rules"])


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


def _clean_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []
    cleaned = df.where(pd.notnull(df), None)
    return cleaned.to_dict(orient="records")


def _apply_filters(df: pd.DataFrame, filters: Dict[str, Optional[str]], *, allowed_missing: Iterable[str] = ()) -> pd.DataFrame:
    result = df.copy()
    for column, value in filters.items():
        if value is None:
            continue
        if column not in result.columns:
            if column in allowed_missing:
                continue
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Filter not available in file: {column}")
        result = result.loc[result[column].astype(str) == str(value)]
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
) -> PaginatedAlertSummaryResponse:
    df = _load_csv(_summary_file(run_id))
    filters = {
        "rule_code": rule_code,
        "risk_level": risk_level,
        "status": status_filter,
        "country_code": country_code,
        "merchant_rubro_proxy": merchant_rubro_proxy,
        "customer_hash": customer_hash,
    }
    filtered = _apply_filters(df, filters, allowed_missing={"country_code", "merchant_rubro_proxy"})
    page_df, safe_page, safe_page_size, total_pages = _paginate(filtered, page, page_size)
    return PaginatedAlertSummaryResponse(
        run_id=run_id,
        page=safe_page,
        page_size=safe_page_size,
        total_items=int(len(filtered)),
        total_pages=total_pages,
        items=[AlertSummaryItem(**row) for row in _clean_records(page_df)],
    )


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
) -> PaginatedAlertsResponse:
    df = _load_csv(_alerts_file(run_id))
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
    filtered = _apply_filters(df, filters)
    page_df, safe_page, safe_page_size, total_pages = _paginate(filtered, page, page_size)
    return PaginatedAlertsResponse(
        run_id=run_id,
        page=safe_page,
        page_size=safe_page_size,
        total_items=int(len(filtered)),
        total_pages=total_pages,
        items=[AlertItem(**row) for row in _clean_records(page_df)],
    )


@router.get("/alerts/{alert_id}", response_model=AlertItem)
def get_alert_detail(alert_id: str, run_id: Optional[str] = Query(None)) -> AlertItem:
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
            return AlertItem(**_clean_records(match.head(1))[0])

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
from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from fastapi import APIRouter, Depends, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.models.models import FraudAlert, ModelResult, RuleAlertReview, Transaction
from backend.app.services import supervised_service
from backend.app.services.anomaly_service import AnomalyService

router = APIRouter(prefix="/dashboard", tags=["dashboard"])
api_router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])

SENSITIVE_COLUMNS = {
    "pan",
    "pan_card",
    "pan_tarjeta",
    "card_pan",
    "tarjeta",
    "authorization_code",
    "reference_number",
    "is_fraud",
    "confirmed_fraud",
    "target_is_fraud",
}


def _processed_dir() -> Path:
    return Path(os.environ.get("PROJECT_PROCESSED_DIR") or os.path.join(os.getcwd(), "data", "processed"))


def _models_dir() -> Path:
    return Path(os.environ.get("PROJECT_MODELS_DIR") or os.path.join(os.getcwd(), "data", "models"))


def _run_token(run_id: Optional[str]) -> Optional[str]:
    if not run_id:
        return None
    match = re.search(r"run_(\d+)", str(run_id))
    if match:
        return match.group(1)
    match = re.search(r"(\d+)$", str(run_id))
    return match.group(1) if match else None


def _default_source_run() -> str:
    summaries = sorted(_processed_dir().glob("alerts_summary_run_*.csv"), key=lambda item: item.stat().st_mtime, reverse=True)
    if summaries:
        token = _run_token(summaries[0].stem)
        if token:
            return f"preprocessed_run_{token}"
    return "preprocessed_run_26"


def _source_run_aliases(source_run: str) -> set[str]:
    token = _run_token(source_run)
    aliases = {source_run}
    if token:
        aliases.update({token, f"run_{token}", f"preprocessed_run_{token}"})
    return aliases


def _summary_path(source_run: str) -> Path:
    token = _run_token(source_run)
    if token:
        return _processed_dir() / f"alerts_summary_run_{token}.csv"
    return _processed_dir() / f"alerts_summary_{source_run}.csv"


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"merchant_rubro_proxy": str}, low_memory=False)


def _clean_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def _safe_record(row: dict[str, Any]) -> dict[str, Any]:
    return {key: _clean_value(value) for key, value in row.items() if str(key).lower() not in SENSITIVE_COLUMNS}


def _apply_latest_review_statuses(df: pd.DataFrame, db: Session, source_run: str) -> pd.DataFrame:
    if df.empty or "summary_alert_id" not in df.columns:
        if "status" not in df.columns:
            df["status"] = "NEW"
        return df

    result = df.copy()
    if "status" not in result.columns:
        result["status"] = "NEW"

    try:
        reviews = (
            db.query(RuleAlertReview)
            .filter(RuleAlertReview.source_run.in_(_source_run_aliases(source_run)))
            .filter(RuleAlertReview.summary_alert_id.isnot(None))
            .order_by(RuleAlertReview.reviewed_at.asc(), RuleAlertReview.id.asc())
            .all()
        )
    except Exception:
        return result
    status_by_summary_id = {
        str(review.summary_alert_id): review.new_status
        for review in reviews
        if review.summary_alert_id and review.new_status
    }
    if status_by_summary_id:
        alert_ids = result["summary_alert_id"].astype(str)
        result.loc[alert_ids.isin(status_by_summary_id), "status"] = alert_ids.map(status_by_summary_id)
    result["status"] = result["status"].fillna("NEW").astype(str).str.upper()
    return result


def _risk_column(df: pd.DataFrame) -> Optional[str]:
    for column in ("max_risk_score", "risk_score", "score"):
        if column in df.columns:
            return column
    return None


def _build_alerts_evolution(summary_df: pd.DataFrame) -> list[dict[str, Any]]:
    if summary_df.empty:
        return []

    date_column = next((column for column in ("created_at", "window_start", "transaction_datetime") if column in summary_df.columns), None)
    if not date_column:
        return []

    frame = summary_df.copy()
    frame["_date"] = pd.to_datetime(frame[date_column], errors="coerce", utc=True).dt.strftime("%Y-%m-%d")
    frame = frame.dropna(subset=["_date"])
    if frame.empty:
        return []

    evolution: list[dict[str, Any]] = []
    for date, group in frame.groupby("_date", sort=True):
        item: dict[str, Any] = {"date": str(date), "count": int(len(group)), "high": 0, "medium": 0, "low": 0}
        if "risk_level" in group.columns:
            counts = group["risk_level"].astype(str).str.upper().value_counts().to_dict()
            item.update(
                {
                    "high": int(counts.get("HIGH", 0)),
                    "medium": int(counts.get("MEDIUM", 0)),
                    "low": int(counts.get("LOW", 0)),
                }
            )
        evolution.append(item)
    return evolution[-30:]


def _build_recent_alerts(summary_df: pd.DataFrame) -> list[dict[str, Any]]:
    if summary_df.empty:
        return []

    frame = summary_df.copy()
    sort_column = next((column for column in ("created_at", "window_start", "window_end") if column in frame.columns), None)
    if sort_column:
        frame["_sort_date"] = pd.to_datetime(frame[sort_column], errors="coerce", utc=True)
        frame = frame.sort_values("_sort_date", ascending=False, na_position="last")

    rows: list[dict[str, Any]] = []
    for record in frame.head(5).to_dict(orient="records"):
        rows.append(
            _safe_record(
                {
                    "alert_id": record.get("summary_alert_id") or record.get("alert_id"),
                    "rule_code": record.get("rule_code"),
                    "customer_hash": record.get("customer_hash"),
                    "risk_score": record.get("max_risk_score", record.get("risk_score")),
                    "risk_level": record.get("risk_level"),
                    "status": record.get("status", "NEW"),
                    "created_at": record.get("created_at") or record.get("window_start"),
                }
            )
        )
    return rows


def _display_model_name(value: Optional[str]) -> str:
    normalized = str(value or "Isolation Forest").replace("_", " ").strip()
    if normalized.lower() in {"isolation forest", "isolationforest", "unsupervised anomaly detection"}:
        return "Isolation Forest"
    return " ".join(part.capitalize() for part in normalized.split())


def _select_anomaly_run(anomaly_run: Optional[str], source_run: str, anomaly_service: AnomalyService) -> Optional[str]:
    if anomaly_run:
        return anomaly_run

    token = _run_token(source_run)
    if token:
        candidate = f"run_{token}"
        if anomaly_service._find_anomaly_scores_file(candidate):
            return candidate

    runs = anomaly_service.list_anomaly_runs()
    return runs[0]["anomaly_run_id"] if runs else None


def _build_active_model(anomaly_run: Optional[str], anomaly_service: AnomalyService, warnings: list[str]) -> tuple[Optional[dict[str, Any]], Optional[int]]:
    if not anomaly_run:
        warnings.append("No hay modelo no supervisado disponible.")
        return None, None
    try:
        metrics = anomaly_service.get_anomaly_metrics(anomaly_run)
        return (
            {
                "model_name": _display_model_name(metrics.get("model_name") or metrics.get("algorithm")),
                "run_id": anomaly_run,
                "anomaly_count": int(metrics.get("anomaly_count") or 0),
                "anomaly_rate": float(metrics.get("anomaly_rate") or 0),
            },
            int(metrics.get("total_transactions") or 0),
        )
    except Exception as exc:
        warnings.append(f"No se pudieron cargar metricas de anomalias para {anomaly_run}: {exc}")
        return None, None


@router.get('/summary')
def dashboard_summary(db: Session = Depends(get_db)):
    # Transactions analyzed
    tx_count = db.query(func.count(Transaction.id)).scalar() or 0

    # Alerts (total) and average risk
    alerts_q = db.query(FraudAlert)
    alerts_count = db.query(func.count(FraudAlert.id)).scalar() or 0
    avg_risk = db.query(func.avg(FraudAlert.risk_score)).scalar() or 0.0

    # Active model
    active = db.query(ModelResult).filter(ModelResult.is_active == True).order_by(ModelResult.created_at.desc()).first()
    active_model = active.model_name if active else None

    # Alert trend (last 7 days)
    trend_rows = (
        db.query(func.date(FraudAlert.created_at).label('d'), func.count(FraudAlert.id).label('c'))
        .group_by(func.date(FraudAlert.created_at))
        .order_by(func.date(FraudAlert.created_at))
        .limit(14)
        .all()
    )
    alert_trend = [{"date": str(r.d), "count": int(r.c)} for r in trend_rows]

    # Fraud ratio from transactions
    fraud_count = db.query(func.count(Transaction.id)).filter(Transaction.is_fraud == True).scalar() or 0
    total_tx = tx_count or 1
    fraud_pct = int((fraud_count / total_tx) * 100) if total_tx else 0
    normal_pct = 100 - fraud_pct

    # Recent alerts
    recent = (
        db.query(FraudAlert).order_by(FraudAlert.created_at.desc()).limit(10).all()
    )
    recent_list = [
        {
            "alert_id": a.id,
            "transaction_id": a.transaction_id,
            "score": a.risk_score,
            "channel": a.transaction.channel if a.transaction else None,
            "amount": float(a.transaction.amount) if a.transaction and a.transaction.amount is not None else None,
            "status": a.status,
            "date": a.created_at.isoformat() if a.created_at else None,
        }
        for a in recent
    ]

    return {
        "transactions": int(tx_count),
        "alerts": int(alerts_count),
        "risk": float(avg_risk) if avg_risk is not None else 0.0,
        "model": active_model or "--",
        "alertTrend": alert_trend,
        "fraudRatio": {"fraud": fraud_pct, "normal": normal_pct},
        "recentAlerts": recent_list,
        # Backwards-compat keys expected by frontend mock
        "recent_alerts": recent_list,
        # Backwards-compatible keys expected by integration tests
        "total_transactions": int(tx_count),
        "active_alerts": int(alerts_count),
        "average_risk": float(avg_risk) if avg_risk is not None else 0.0,
        "active_model": active_model or "--",
    }


@api_router.get("/overview")
def dashboard_overview(
    source_run: Optional[str] = Query(None),
    anomaly_run: Optional[str] = Query(None),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    resolved_source_run = source_run or _default_source_run()
    anomaly_service = AnomalyService(processed_dir=str(_processed_dir()), models_dir=str(_models_dir()))
    resolved_anomaly_run = _select_anomaly_run(anomaly_run, resolved_source_run, anomaly_service)
    warnings: list[str] = []

    active_model, total_transactions = _build_active_model(resolved_anomaly_run, anomaly_service, warnings)

    summary_df = pd.DataFrame()
    summary_file = _summary_path(resolved_source_run)
    if summary_file.exists():
        try:
            summary_df = _read_csv(summary_file)
            summary_df = _apply_latest_review_statuses(summary_df, db, resolved_source_run)
        except Exception as exc:
            warnings.append(f"No se pudo leer el resumen de alertas {summary_file.name}: {exc}")
            summary_df = pd.DataFrame()
    else:
        warnings.append(f"No existe archivo de alertas agrupadas para {resolved_source_run}.")

    if total_transactions is None:
        total_transactions = 0

    active_alerts = 0
    average_risk_score = None
    if not summary_df.empty:
        if "status" in summary_df.columns:
            active_alerts = int(summary_df["status"].astype(str).str.upper().isin({"NEW", "IN_REVIEW"}).sum())
        risk_column = _risk_column(summary_df)
        if risk_column:
            average_risk_score = float(pd.to_numeric(summary_df[risk_column], errors="coerce").dropna().mean())

    try:
        label_summary = supervised_service.get_human_label_summary(db, source_run=resolved_source_run)
    except Exception:
        warnings.append("No se pudieron cargar revisiones humanas; se muestra distribucion vacia.")
        label_summary = {
            "total_reviews": 0,
            "confirmed_fraud": 0,
            "dismissed": 0,
            "new": 0,
            "in_review": 0,
            "false_positive_excluded": 0,
            "usable_total_labels": 0,
        }
    review_distribution = {
        "confirmed_fraud": int(label_summary.get("confirmed_fraud") or 0),
        "dismissed": int(label_summary.get("dismissed") or 0),
        "in_review": int(label_summary.get("in_review") or 0),
        "new": int(label_summary.get("new") or 0),
        "false_positive_excluded": int(label_summary.get("false_positive_excluded") or 0),
        "total_reviews": int(label_summary.get("total_reviews") or 0),
        "usable_total_labels": int(label_summary.get("usable_total_labels") or 0),
    }

    return {
        "source_run": resolved_source_run,
        "anomaly_run": resolved_anomaly_run,
        "total_transactions": int(total_transactions or 0),
        "active_alerts": active_alerts,
        "average_risk_score": average_risk_score,
        "active_model": active_model,
        "review_distribution": review_distribution,
        "alerts_evolution": _build_alerts_evolution(summary_df),
        "recent_alerts": _build_recent_alerts(summary_df),
        "warnings": warnings,
    }

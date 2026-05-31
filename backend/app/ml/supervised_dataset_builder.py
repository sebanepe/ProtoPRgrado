from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from sqlalchemy.orm import Session

from backend.app.database import SessionLocal
from backend.app.models.models import ArtifactRegistry, RuleAlertReview, RuleRun, SupervisedDatasetRun
from backend.app.services import artifact_registry_service as artifacts
from backend.app.services.supervised_service import (
    RECOMMENDED_NEGATIVE_REQUIRED,
    RECOMMENDED_POSITIVE_REQUIRED,
    STRONG_NEGATIVE_TARGET,
    STRONG_POSITIVE_TARGET,
    TECHNICAL_NEGATIVE_REQUIRED,
    TECHNICAL_POSITIVE_REQUIRED,
)


LABEL_POLICY = "HUMAN_REVIEW_CONFIRMED_FRAUD_DISMISSED"
TARGET_SOURCE = "HUMAN_REVIEW"
USABLE_LABELS = {"CONFIRMED_FRAUD": 1, "DISMISSED": 0}
EXCLUDED_STATUSES = {"NEW", "IN_REVIEW", "FALSE_POSITIVE"}
FALLBACK_WARNING = "Se uso fallback de archivos porque no existe registro de artefactos en DB."
NO_LABELS_MESSAGE = "No existen etiquetas humanas usables para construir el dataset supervisado."
METHODOLOGY_WARNING = (
    "Este dataset fue construido unicamente con etiquetas humanas de revision. "
    "CONFIRMED_FRAUD se usa como clase positiva y DISMISSED como clase negativa. "
    "Las reglas automaticas, anomaly_flag y risk_score no se utilizan como etiquetas de fraude."
)
SENSITIVE_COLUMNS = {"PAN_TARJETA", "TARJETA", "pan_card", "raw_card", "masked_card"}
FORBIDDEN_OUTPUT_COLUMNS = SENSITIVE_COLUMNS | {"is_fraud", "confirmed_fraud"}


def _json_dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True)


def _status(value: Any) -> str:
    text = str(value or "").strip().upper()
    return text or "UNKNOWN"


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(value)
    except Exception:
        return default


def _truthy_contains(*values: Any, terms: str) -> bool:
    haystack = " ".join(str(value or "") for value in values).upper()
    return any(term in haystack for term in terms.upper().split("|"))


def _duration_minutes(start: Any, end: Any) -> Optional[float]:
    start_dt = pd.to_datetime(start, errors="coerce", utc=True)
    end_dt = pd.to_datetime(end, errors="coerce", utc=True)
    if pd.isna(start_dt) or pd.isna(end_dt):
        return None
    return round(float((end_dt - start_dt).total_seconds() / 60.0), 3)


def _readiness(positive_count: int, negative_count: int) -> dict[str, bool]:
    return {
        "technical_ready": positive_count >= TECHNICAL_POSITIVE_REQUIRED and negative_count >= TECHNICAL_NEGATIVE_REQUIRED,
        "recommended_ready": positive_count >= RECOMMENDED_POSITIVE_REQUIRED and negative_count >= RECOMMENDED_NEGATIVE_REQUIRED,
        "strong_ready": positive_count >= STRONG_POSITIVE_TARGET and negative_count >= STRONG_NEGATIVE_TARGET,
    }


def _dataset_status(readiness: dict[str, bool], total: int) -> str:
    if total <= 0:
        return "NOT_CREATED_INSUFFICIENT_HUMAN_LABELS"
    if readiness["strong_ready"]:
        return "READY_STRONG"
    if readiness["recommended_ready"]:
        return "READY_RECOMMENDED"
    if readiness["technical_ready"]:
        return "READY_TECHNICAL"
    return "CREATED_INSUFFICIENT_LABELS"


def _processed_dir(output_dir: str | Path | None = None) -> Path:
    return Path(output_dir) if output_dir else artifacts.default_processed_dir()


def _resolve_registered_file(db: Session, source_run: str, artifact_type: str, rule_attr: str) -> tuple[Optional[Path], str]:
    artifact = artifacts.get_artifact_by_type(db, source_run, artifact_type)
    if artifact and artifact.status == "AVAILABLE" and Path(artifact.file_path).exists():
        return Path(artifact.file_path), "artifact_registry"

    rule_run = db.query(RuleRun).filter(RuleRun.source_run == artifacts.normalize_source_run(source_run)).first()
    if rule_run:
        value = getattr(rule_run, rule_attr, None)
        if value and Path(value).exists():
            return Path(value), "rule_runs"
    return None, "fallback"


def _resolve_input_files(
    db: Session,
    source_run: str,
    run_token: str,
    processed_dir: Path,
) -> tuple[Path, Path, str, list[str]]:
    warnings: list[str] = []
    summary_path, summary_source = _resolve_registered_file(db, source_run, artifacts.ARTIFACT_RULE_SUMMARY_CSV, "summary_file")
    alerts_path, alerts_source = _resolve_registered_file(db, source_run, artifacts.ARTIFACT_RULE_ALERTS_CSV, "alerts_file")

    used_sources = {summary_source, alerts_source}
    if summary_path is None:
        summary_path = processed_dir / f"alerts_summary_run_{run_token}.csv"
    if alerts_path is None:
        alerts_path = processed_dir / f"alerts_run_{run_token}.csv"
    if "fallback" in used_sources:
        warnings.append(FALLBACK_WARNING)

    return summary_path, alerts_path, "fallback" if "fallback" in used_sources else "+".join(sorted(used_sources)), warnings


def _reviews_dataframe(db: Session, source_run: str) -> tuple[pd.DataFrame, dict[str, int]]:
    rows = (
        db.query(RuleAlertReview)
        .filter(RuleAlertReview.source_run == artifacts.normalize_source_run(source_run))
        .order_by(RuleAlertReview.reviewed_at.desc(), RuleAlertReview.id.desc())
        .all()
    )
    counts = {"total": len(rows), "NEW": 0, "IN_REVIEW": 0, "FALSE_POSITIVE": 0}
    payload: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for review in rows:
        status = _status(review.new_status)
        if status in counts:
            counts[status] += 1
        if status not in USABLE_LABELS or not review.summary_alert_id:
            continue
        key = (review.source_run, review.summary_alert_id)
        if key in seen:
            continue
        seen.add(key)
        payload.append(
            {
                "source_run": artifacts.normalize_source_run(review.source_run),
                "summary_alert_id": review.summary_alert_id,
                "human_review_status": status,
                "reviewed_at": review.reviewed_at.isoformat() if review.reviewed_at else None,
                "reviewed_by": review.reviewed_by_id,
                "human_review_comment": review.analyst_notes,
                "target_human_label": USABLE_LABELS[status],
                "target_label_meaning": status,
            }
        )
    return pd.DataFrame(payload), counts


def _summary_dataframe(summary_path: Path, source_run: str) -> pd.DataFrame:
    df = pd.read_csv(summary_path)
    if "source_run" in df.columns:
        df["source_run"] = df["source_run"].apply(lambda value: artifacts.normalize_source_run(value))
    else:
        df["source_run"] = artifacts.normalize_source_run(source_run)
    df["summary_alert_id"] = df["summary_alert_id"].astype(str)
    return df


def _build_rows(merged: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        countries = str(row.get("countries_detected") or "")
        countries_count = len([item for item in countries.split("|") if item])
        rule_code = row.get("rule_code")
        rule_name = row.get("rule_name")
        max_score = row.get("max_score", row.get("max_risk_score"))
        tx_detected = row.get("transactions_detected", row.get("count_transactions"))
        output = {
            "source_run": row.get("source_run"),
            "summary_alert_id": row.get("summary_alert_id"),
            "representative_transaction_id": row.get("representative_transaction_id"),
            "customer_hash": row.get("customer_hash"),
            "rule_code": rule_code,
            "rule_name": rule_name,
            "risk_level": row.get("risk_level"),
            "max_score": max_score,
            "transactions_detected": tx_detected,
            "countries_detected": countries,
            "merchant_rubro_proxy": row.get("merchant_rubro_proxy", row.get("top_merchant_rubro_proxy")),
            "merchant_rubro_values": row.get("merchant_rubro_values"),
            "window_start": row.get("window_start"),
            "window_end": row.get("window_end"),
            "duration_minutes": row.get("duration_minutes")
            if "duration_minutes" in row
            else _duration_minutes(row.get("window_start"), row.get("window_end")),
            "status": row.get("status"),
            "countries_count": countries_count,
            "has_multiple_countries": countries_count > 1,
            "is_high_risk_rule": str(row.get("risk_level") or "").upper() == "HIGH",
            "is_velocity_rule": _truthy_contains(rule_code, rule_name, terms="VELOCITY|FREQUENCY|BURST|RAPID"),
            "is_double_country_rule": _truthy_contains(rule_code, rule_name, terms="DOUBLE_COUNTRY|MULTIPLE_COUNTRY|COUNTRY"),
            "is_mcc_risk_rule": _truthy_contains(rule_code, rule_name, terms="MCC|RUBRO|MERCHANT"),
            "is_card_present_rule": _truthy_contains(rule_code, rule_name, terms="CARD_PRESENT|PRESENT"),
            "is_card_absent_rule": _truthy_contains(rule_code, rule_name, terms="CARD_ABSENT|NOT_PRESENT|ABSENT"),
            "is_internet_related": _truthy_contains(rule_code, rule_name, row.get("merchant_rubro_values"), terms="INTERNET|ONLINE|WEB|ECOM"),
            "is_atm_or_cash_related": _truthy_contains(rule_code, rule_name, row.get("merchant_rubro_values"), terms="ATM|CASH|CAJERO"),
            "human_review_status": row.get("human_review_status"),
            "reviewed_at": row.get("reviewed_at"),
            "reviewed_by": row.get("reviewed_by"),
            "human_review_comment": row.get("human_review_comment"),
            "target_human_label": _safe_int(row.get("target_human_label")),
            "target_label_source": TARGET_SOURCE,
            "target_label_meaning": row.get("target_label_meaning"),
        }
        for column in FORBIDDEN_OUTPUT_COLUMNS:
            output.pop(column, None)
        rows.append(output)
    return pd.DataFrame(rows)


def _write_report(
    *,
    report_path: Path,
    source_run: str,
    run_token: str,
    total_reviews: int,
    counts: dict[str, int],
    positive_count: int,
    negative_count: int,
    row_count: int,
    readiness: dict[str, bool],
    dataset_path: Optional[Path],
    artifact_dataset_id: Optional[int],
    artifact_report_id: Optional[int],
    supervised_dataset_run_id: Optional[int],
    source_mode: str,
    warnings: list[str],
) -> None:
    generated_at = datetime.now(timezone.utc).isoformat()
    lines = [
        "# Reporte dataset supervisado humano",
        "",
        f"- source_run: {source_run}",
        f"- run_token: {run_token}",
        f"- generado_en: {generated_at}",
        f"- total revisiones leidas: {total_reviews}",
        f"- total revisiones usables: {positive_count + negative_count}",
        f"- total CONFIRMED_FRAUD: {positive_count}",
        f"- total DISMISSED: {negative_count}",
        f"- total excluidas NEW: {counts.get('NEW', 0)}",
        f"- total excluidas IN_REVIEW: {counts.get('IN_REVIEW', 0)}",
        f"- total excluidas FALSE_POSITIVE: {counts.get('FALSE_POSITIVE', 0)}",
        f"- total filas dataset: {row_count}",
        f"- distribucion target: 1={positive_count}, 0={negative_count}",
        f"- readiness minimo tecnico 20/20: {readiness['technical_ready']}",
        f"- readiness recomendado 50/120: {readiness['recommended_ready']}",
        f"- readiness fuerte 70/180: {readiness['strong_ready']}",
        f"- archivo generado: {dataset_path.name if dataset_path else 'NO_CREADO'}",
        f"- artifact_registry dataset id: {artifact_dataset_id}",
        f"- artifact_registry report id: {artifact_report_id}",
        f"- supervised_dataset_runs id: {supervised_dataset_run_id}",
        f"- origen archivos: {source_mode}",
        f"- advertencias: {warnings}",
        "",
        "## Advertencia metodologica",
        "",
        METHODOLOGY_WARNING,
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def _upsert_supervised_run(
    db: Session,
    *,
    source_run: str,
    run_token: str,
    dataset_path: Optional[Path],
    report_path: Optional[Path],
    positive_count: int,
    negative_count: int,
    readiness: dict[str, bool],
    status: str,
    metadata: dict[str, Any],
) -> SupervisedDatasetRun:
    existing = (
        db.query(SupervisedDatasetRun)
        .filter(SupervisedDatasetRun.source_run == source_run, SupervisedDatasetRun.run_token == run_token)
        .order_by(SupervisedDatasetRun.id.desc())
        .first()
    )
    payload = {
        "source_run": source_run,
        "run_token": run_token,
        "dataset_file": str(dataset_path) if dataset_path else None,
        "report_file": str(report_path) if report_path else None,
        "label_policy": LABEL_POLICY,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "usable_total_count": positive_count + negative_count,
        "technical_ready": readiness["technical_ready"],
        "recommended_ready": readiness["recommended_ready"],
        "strong_ready": readiness["strong_ready"],
        "status": status,
        "metadata_json": _json_dumps(metadata),
    }
    if existing is None:
        existing = SupervisedDatasetRun(**payload)
        db.add(existing)
    else:
        for key, value in payload.items():
            setattr(existing, key, value)
    db.commit()
    db.refresh(existing)
    return existing


def build_human_supervised_alert_dataset(
    source_run: str,
    db: Optional[Session] = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    own_session = db is None
    session = db or SessionLocal()
    warnings: list[str] = []
    try:
        normalized = artifacts.normalize_source_run(source_run)
        run_token = artifacts.normalize_run_token(normalized)
        processed = _processed_dir(output_dir)
        processed.mkdir(parents=True, exist_ok=True)

        summary_path, alerts_path, source_mode, source_warnings = _resolve_input_files(session, normalized, run_token, processed)
        warnings.extend(source_warnings)
        if not summary_path.exists():
            return {
                "status": "NOT_CREATED",
                "verdict": "DATASET_NOT_CREATED_SOURCE_SUMMARY_MISSING",
                "source_run": normalized,
                "message": f"No existe archivo de resumen de alertas: {summary_path}",
                "warnings": warnings,
            }

        summary_df = _summary_dataframe(summary_path, normalized)
        reviews_df, review_counts = _reviews_dataframe(session, normalized)
        positive_count = int((reviews_df.get("target_human_label", pd.Series(dtype=int)) == 1).sum()) if not reviews_df.empty else 0
        negative_count = int((reviews_df.get("target_human_label", pd.Series(dtype=int)) == 0).sum()) if not reviews_df.empty else 0
        readiness = _readiness(positive_count, negative_count)

        if reviews_df.empty:
            status = "NOT_CREATED_INSUFFICIENT_HUMAN_LABELS"
            supervised_run = _upsert_supervised_run(
                session,
                source_run=normalized,
                run_token=run_token,
                dataset_path=None,
                report_path=None,
                positive_count=0,
                negative_count=0,
                readiness=readiness,
                status=status,
                metadata={
                    "excluded_new": review_counts.get("NEW", 0),
                    "excluded_in_review": review_counts.get("IN_REVIEW", 0),
                    "excluded_false_positive": review_counts.get("FALSE_POSITIVE", 0),
                    "source_summary_file": str(summary_path),
                    "source_reviews_table": "rule_alert_reviews",
                    "join_key": "source_run + summary_alert_id",
                },
            )
            return {
                "status": "NOT_CREATED",
                "verdict": "DATASET_NOT_CREATED_INSUFFICIENT_HUMAN_LABELS",
                "source_run": normalized,
                "usable_positive_labels": 0,
                "usable_negative_labels": 0,
                "usable_total_labels": 0,
                "message": NO_LABELS_MESSAGE,
                "supervised_dataset_run_id": supervised_run.id,
                "warnings": warnings,
            }

        merged = summary_df.merge(reviews_df, on=["source_run", "summary_alert_id"], how="inner")
        dataset_df = _build_rows(merged)
        for column in FORBIDDEN_OUTPUT_COLUMNS:
            if column in dataset_df.columns:
                dataset_df = dataset_df.drop(columns=[column])

        dataset_path = processed / f"supervised_human_alert_dataset_run_{run_token}.csv"
        report_path = processed / f"supervised_human_dataset_report_run_{run_token}.md"
        dataset_df.to_csv(dataset_path, index=False)
        status = _dataset_status(readiness, len(dataset_df))
        metadata = {
            "excluded_new": review_counts.get("NEW", 0),
            "excluded_in_review": review_counts.get("IN_REVIEW", 0),
            "excluded_false_positive": review_counts.get("FALSE_POSITIVE", 0),
            "source_summary_file": str(summary_path),
            "source_alerts_file": str(alerts_path),
            "source_reviews_table": "rule_alert_reviews",
            "join_key": "source_run + summary_alert_id",
        }
        supervised_run = _upsert_supervised_run(
            session,
            source_run=normalized,
            run_token=run_token,
            dataset_path=dataset_path,
            report_path=report_path,
            positive_count=positive_count,
            negative_count=negative_count,
            readiness=readiness,
            status=status,
            metadata=metadata,
        )
        dataset_artifact = artifacts.register_or_update_artifact(
            session,
            artifact_type=artifacts.ARTIFACT_SUPERVISED_DATASET,
            phase=artifacts.PHASE_C4,
            source_run=normalized,
            run_token=run_token,
            file_path=dataset_path,
            metadata={
                "label_policy": LABEL_POLICY,
                "positive_count": positive_count,
                "negative_count": negative_count,
                "usable_total_count": positive_count + negative_count,
                **readiness,
            },
        )
        _write_report(
            report_path=report_path,
            source_run=normalized,
            run_token=run_token,
            total_reviews=review_counts["total"],
            counts=review_counts,
            positive_count=positive_count,
            negative_count=negative_count,
            row_count=len(dataset_df),
            readiness=readiness,
            dataset_path=dataset_path,
            artifact_dataset_id=dataset_artifact.id,
            artifact_report_id=None,
            supervised_dataset_run_id=supervised_run.id,
            source_mode=source_mode,
            warnings=warnings,
        )
        report_artifact = artifacts.register_or_update_artifact(
            session,
            artifact_type=artifacts.ARTIFACT_SUPERVISED_REPORT,
            phase=artifacts.PHASE_C4,
            source_run=normalized,
            run_token=run_token,
            file_path=report_path,
            metadata={"label_policy": LABEL_POLICY, "supervised_dataset_run_id": supervised_run.id},
        )
        _write_report(
            report_path=report_path,
            source_run=normalized,
            run_token=run_token,
            total_reviews=review_counts["total"],
            counts=review_counts,
            positive_count=positive_count,
            negative_count=negative_count,
            row_count=len(dataset_df),
            readiness=readiness,
            dataset_path=dataset_path,
            artifact_dataset_id=dataset_artifact.id,
            artifact_report_id=report_artifact.id,
            supervised_dataset_run_id=supervised_run.id,
            source_mode=source_mode,
            warnings=warnings,
        )
        report_artifact = artifacts.register_or_update_artifact(
            session,
            artifact_type=artifacts.ARTIFACT_SUPERVISED_REPORT,
            phase=artifacts.PHASE_C4,
            source_run=normalized,
            run_token=run_token,
            file_path=report_path,
            metadata={"label_policy": LABEL_POLICY, "supervised_dataset_run_id": supervised_run.id},
        )

        return {
            "status": "COMPLETED",
            "verdict": "HUMAN_SUPERVISED_DATASET_CREATED",
            "source_run": normalized,
            "dataset_file": dataset_path.name,
            "dataset_path": str(dataset_path),
            "report_file": report_path.name,
            "report_path": str(report_path),
            "usable_positive_labels": positive_count,
            "usable_negative_labels": negative_count,
            "usable_total_labels": positive_count + negative_count,
            **readiness,
            "artifact_registry_dataset_id": dataset_artifact.id,
            "artifact_registry_report_id": report_artifact.id,
            "supervised_dataset_run_id": supervised_run.id,
            "warnings": warnings,
        }
    except Exception as exc:
        if session:
            session.rollback()
        return {
            "status": "ERROR",
            "verdict": "HUMAN_SUPERVISED_DATASET_BUILD_FAILED",
            "source_run": source_run,
            "message": str(exc),
            "warnings": warnings,
        }
    finally:
        if own_session:
            session.close()


def get_latest_supervised_dataset_run(db: Session, source_run: str) -> Optional[SupervisedDatasetRun]:
    normalized = artifacts.normalize_source_run(source_run)
    return (
        db.query(SupervisedDatasetRun)
        .filter(SupervisedDatasetRun.source_run == normalized)
        .order_by(SupervisedDatasetRun.updated_at.desc(), SupervisedDatasetRun.id.desc())
        .first()
    )


def supervised_dataset_run_to_dict(item: SupervisedDatasetRun) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if item.metadata_json:
        try:
            parsed = json.loads(item.metadata_json)
            metadata = parsed if isinstance(parsed, dict) else {}
        except Exception:
            metadata = {}
    return {
        "id": item.id,
        "source_run": item.source_run,
        "run_token": item.run_token,
        "dataset_file": item.dataset_file,
        "report_file": item.report_file,
        "label_policy": item.label_policy,
        "positive_count": item.positive_count,
        "negative_count": item.negative_count,
        "usable_total_count": item.usable_total_count,
        "technical_ready": item.technical_ready,
        "recommended_ready": item.recommended_ready,
        "strong_ready": item.strong_ready,
        "status": item.status,
        "metadata_json": metadata,
        "created_at": item.created_at.isoformat() if item.created_at else None,
        "updated_at": item.updated_at.isoformat() if item.updated_at else None,
    }

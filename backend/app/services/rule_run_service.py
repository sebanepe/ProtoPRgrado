from __future__ import annotations

import json
from typing import Any, Optional

from sqlalchemy.orm import Session

from backend.app.models.models import RuleRun
from backend.app.services import artifact_registry_service as artifacts


def rule_run_to_dict(rule_run: RuleRun) -> dict[str, Any]:
    metadata = {}
    if rule_run.metadata_json:
        try:
            metadata = json.loads(rule_run.metadata_json)
        except Exception:
            metadata = {}
    return {
        "id": rule_run.id,
        "source_run": rule_run.source_run,
        "run_token": rule_run.run_token,
        "alerts_file": rule_run.alerts_file,
        "summary_file": rule_run.summary_file,
        "report_file": rule_run.report_file,
        "detailed_alert_count": rule_run.detailed_alert_count,
        "grouped_alert_count": rule_run.grouped_alert_count,
        "status": rule_run.status,
        "metadata_json": metadata,
        "created_at": rule_run.created_at.isoformat() if rule_run.created_at else None,
        "updated_at": rule_run.updated_at.isoformat() if rule_run.updated_at else None,
    }


def list_rule_runs(db: Session, source_run: Optional[str] = None) -> list[RuleRun]:
    query = db.query(RuleRun)
    if source_run:
        query = query.filter(RuleRun.source_run == artifacts.normalize_source_run(source_run))
    return query.order_by(RuleRun.created_at.desc(), RuleRun.id.desc()).all()


def register_rule_run_from_artifacts(db: Session, source_run: str) -> RuleRun:
    normalized = artifacts.normalize_source_run(source_run)
    run_token = artifacts.normalize_run_token(normalized)
    alerts_artifact = artifacts.get_artifact_by_type(db, normalized, artifacts.ARTIFACT_RULE_ALERTS_CSV)
    summary_artifact = artifacts.get_artifact_by_type(db, normalized, artifacts.ARTIFACT_RULE_SUMMARY_CSV)
    report_artifact = artifacts.get_artifact_by_type(db, normalized, artifacts.ARTIFACT_RULE_REPORT)

    detailed_count = int(alerts_artifact.row_count or 0) if alerts_artifact and alerts_artifact.status == "AVAILABLE" else 0
    grouped_count = int(summary_artifact.row_count or 0) if summary_artifact and summary_artifact.status == "AVAILABLE" else 0
    missing = [
        name
        for name, artifact in {
            "RULE_ALERTS_CSV": alerts_artifact,
            "RULE_SUMMARY_CSV": summary_artifact,
            "RULE_REPORT": report_artifact,
        }.items()
        if artifact is None or artifact.status != "AVAILABLE"
    ]

    existing = db.query(RuleRun).filter(RuleRun.source_run == normalized).first()
    payload = {
        "run_token": run_token,
        "alerts_file": alerts_artifact.file_path if alerts_artifact else None,
        "summary_file": summary_artifact.file_path if summary_artifact else None,
        "report_file": report_artifact.file_path if report_artifact else None,
        "detailed_alert_count": detailed_count,
        "grouped_alert_count": grouped_count,
        "status": "AVAILABLE" if not missing else "PARTIAL",
        "metadata_json": json.dumps({"missing_artifacts": missing}, ensure_ascii=True, sort_keys=True),
    }
    if existing is None:
        existing = RuleRun(source_run=normalized, **payload)
        db.add(existing)
    else:
        for key, value in payload.items():
            setattr(existing, key, value)
    db.commit()
    db.refresh(existing)
    return existing

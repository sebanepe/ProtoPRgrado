from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.models.models import SupervisedDatasetRun
from backend.app.services import artifact_registry_service, model_registry_service, rule_run_service


router = APIRouter(prefix="/api/artifacts", tags=["artifacts", "traceability"])


class RegisterExistingArtifactsRequest(BaseModel):
    source_run: str


def _supervised_dataset_to_dict(item: SupervisedDatasetRun) -> dict[str, Any]:
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
        "created_at": item.created_at.isoformat() if item.created_at else None,
        "updated_at": item.updated_at.isoformat() if item.updated_at else None,
    }


@router.post("/register-existing")
def register_existing_artifacts(
    request: RegisterExistingArtifactsRequest,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    scan = artifact_registry_service.scan_existing_artifacts(db, request.source_run)
    rule_run = rule_run_service.register_rule_run_from_artifacts(db, request.source_run)
    model = model_registry_service.register_unsupervised_model_from_artifacts(db, request.source_run)
    return {
        **scan,
        "rule_run": rule_run_service.rule_run_to_dict(rule_run),
        "model_registry": model_registry_service.model_registry_to_dict(model),
    }


@router.get("")
def list_artifacts(
    source_run: Optional[str] = Query(None),
    phase: Optional[str] = Query(None),
    artifact_type: Optional[str] = Query(None),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    items = artifact_registry_service.list_artifacts(
        db,
        source_run=source_run,
        phase=phase,
        artifact_type=artifact_type,
    )
    return {"count": len(items), "items": [artifact_registry_service.artifact_to_dict(item) for item in items]}


@router.get("/rule-runs")
def list_rule_runs(
    source_run: Optional[str] = Query(None),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    items = rule_run_service.list_rule_runs(db, source_run=source_run)
    return {"count": len(items), "items": [rule_run_service.rule_run_to_dict(item) for item in items]}


@router.get("/model-registry")
def list_model_registry(
    source_run: Optional[str] = Query(None),
    model_family: Optional[str] = Query(None),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    items = model_registry_service.list_model_registry(db, source_run=source_run, model_family=model_family)
    return {"count": len(items), "items": [model_registry_service.model_registry_to_dict(item) for item in items]}


@router.get("/traceability")
def get_traceability(source_run: str = Query(...), db: Session = Depends(get_db)) -> dict[str, Any]:
    normalized = artifact_registry_service.normalize_source_run(source_run)
    run_token = artifact_registry_service.normalize_run_token(normalized)

    def artifact_payload(artifact_type: str) -> dict[str, Any] | None:
        artifact = artifact_registry_service.get_artifact_by_type(db, normalized, artifact_type)
        return artifact_registry_service.artifact_to_dict(artifact) if artifact else None

    rule_runs = rule_run_service.list_rule_runs(db, source_run=normalized)
    models = model_registry_service.list_model_registry(db, source_run=normalized, model_family="UNSUPERVISED")
    supervised_runs = (
        db.query(SupervisedDatasetRun)
        .filter(SupervisedDatasetRun.source_run == normalized)
        .order_by(SupervisedDatasetRun.created_at.desc(), SupervisedDatasetRun.id.desc())
        .all()
    )
    warnings: list[str] = []
    for label, payload in {
        "preprocessed_csv": artifact_payload(artifact_registry_service.ARTIFACT_PREPROCESSED_CSV),
        "alerts_csv": artifact_payload(artifact_registry_service.ARTIFACT_RULE_ALERTS_CSV),
        "summary_csv": artifact_payload(artifact_registry_service.ARTIFACT_RULE_SUMMARY_CSV),
        "anomaly_scores": artifact_payload(artifact_registry_service.ARTIFACT_ANOMALY_SCORES_CSV),
        "model_pickle": artifact_payload(artifact_registry_service.ARTIFACT_MODEL_PICKLE),
    }.items():
        if payload is None or payload.get("status") != "AVAILABLE":
            warnings.append(f"Missing or unavailable artifact: {label}")

    return {
        "source_run": normalized,
        "run_token": run_token,
        "phase_a": {
            "preprocessed_csv": artifact_payload(artifact_registry_service.ARTIFACT_PREPROCESSED_CSV),
            "report": artifact_payload(artifact_registry_service.ARTIFACT_PREPROCESSING_REPORT),
        },
        "phase_b": {
            "alerts_csv": artifact_payload(artifact_registry_service.ARTIFACT_RULE_ALERTS_CSV),
            "summary_csv": artifact_payload(artifact_registry_service.ARTIFACT_RULE_SUMMARY_CSV),
            "rules_report": artifact_payload(artifact_registry_service.ARTIFACT_RULE_REPORT),
            "rule_run": rule_run_service.rule_run_to_dict(rule_runs[0]) if rule_runs else None,
        },
        "phase_c3": {
            "model": model_registry_service.model_registry_to_dict(models[0]) if models else None,
            "scores": artifact_payload(artifact_registry_service.ARTIFACT_ANOMALY_SCORES_CSV),
            "metadata": artifact_payload(artifact_registry_service.ARTIFACT_MODEL_METADATA),
            "report": artifact_payload(artifact_registry_service.ARTIFACT_ANOMALY_REPORT),
            "feature_set": artifact_payload(artifact_registry_service.ARTIFACT_UNSUPERVISED_FEATURE_SET),
        },
        "phase_c4": {
            "supervised_dataset_runs": [_supervised_dataset_to_dict(item) for item in supervised_runs],
        },
        "warnings": warnings,
    }

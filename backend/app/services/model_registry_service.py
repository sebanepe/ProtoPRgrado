from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from sqlalchemy.orm import Session

from backend.app.models.models import ModelRegistry
from backend.app.services import artifact_registry_service as artifacts


def _load_json_file(path: Optional[str]) -> dict[str, Any]:
    if not path:
        return {}
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def model_registry_to_dict(model: ModelRegistry) -> dict[str, Any]:
    metrics = {}
    if model.metrics_json:
        try:
            metrics = json.loads(model.metrics_json)
        except Exception:
            metrics = {}
    return {
        "id": model.id,
        "model_family": model.model_family,
        "algorithm": model.algorithm,
        "source_run": model.source_run,
        "run_token": model.run_token,
        "model_file": model.model_file,
        "metadata_file": model.metadata_file,
        "report_file": model.report_file,
        "scores_file": model.scores_file,
        "feature_file": model.feature_file,
        "metrics_json": metrics,
        "status": model.status,
        "is_active": model.is_active,
        "created_at": model.created_at.isoformat() if model.created_at else None,
        "updated_at": model.updated_at.isoformat() if model.updated_at else None,
    }


def list_model_registry(
    db: Session,
    *,
    source_run: Optional[str] = None,
    model_family: Optional[str] = None,
) -> list[ModelRegistry]:
    query = db.query(ModelRegistry)
    if source_run:
        query = query.filter(ModelRegistry.source_run == artifacts.normalize_source_run(source_run))
    if model_family:
        query = query.filter(ModelRegistry.model_family == model_family)
    return query.order_by(ModelRegistry.created_at.desc(), ModelRegistry.id.desc()).all()


def register_unsupervised_model_from_artifacts(db: Session, source_run: str) -> ModelRegistry:
    normalized = artifacts.normalize_source_run(source_run)
    run_token = artifacts.normalize_run_token(normalized)

    model_artifact = artifacts.get_artifact_by_type(db, normalized, artifacts.ARTIFACT_MODEL_PICKLE)
    metadata_artifact = artifacts.get_artifact_by_type(db, normalized, artifacts.ARTIFACT_MODEL_METADATA)
    scores_artifact = artifacts.get_artifact_by_type(db, normalized, artifacts.ARTIFACT_ANOMALY_SCORES_CSV)
    report_artifact = artifacts.get_artifact_by_type(db, normalized, artifacts.ARTIFACT_ANOMALY_REPORT)
    feature_artifact = artifacts.get_artifact_by_type(db, normalized, artifacts.ARTIFACT_UNSUPERVISED_FEATURE_SET)

    metadata = _load_json_file(metadata_artifact.file_path if metadata_artifact and metadata_artifact.status == "AVAILABLE" else None)
    algorithm = str(metadata.get("algorithm") or metadata.get("model_type") or "isolation_forest").lower()
    missing = [
        name
        for name, artifact in {
            "MODEL_PICKLE": model_artifact,
            "MODEL_METADATA": metadata_artifact,
            "ANOMALY_SCORES_CSV": scores_artifact,
            "ANOMALY_REPORT": report_artifact,
            "UNSUPERVISED_FEATURE_SET": feature_artifact,
        }.items()
        if artifact is None or artifact.status != "AVAILABLE"
    ]
    has_active = db.query(ModelRegistry).filter(ModelRegistry.model_family == "UNSUPERVISED", ModelRegistry.is_active.is_(True)).first()

    existing = (
        db.query(ModelRegistry)
        .filter(
            ModelRegistry.model_family == "UNSUPERVISED",
            ModelRegistry.algorithm == algorithm,
            ModelRegistry.source_run == normalized,
        )
        .first()
    )
    payload = {
        "run_token": run_token,
        "model_file": model_artifact.file_path if model_artifact else None,
        "metadata_file": metadata_artifact.file_path if metadata_artifact else None,
        "report_file": report_artifact.file_path if report_artifact else None,
        "scores_file": scores_artifact.file_path if scores_artifact else None,
        "feature_file": feature_artifact.file_path if feature_artifact else None,
        "metrics_json": json.dumps({"metadata": metadata, "missing_artifacts": missing}, ensure_ascii=True, sort_keys=True),
        "status": "AVAILABLE" if not missing else "PARTIAL",
    }
    if existing is None:
        existing = ModelRegistry(
            model_family="UNSUPERVISED",
            algorithm=algorithm,
            source_run=normalized,
            is_active=has_active is None,
            **payload,
        )
        db.add(existing)
    else:
        for key, value in payload.items():
            setattr(existing, key, value)
        if has_active is None:
            existing.is_active = True
    db.commit()
    db.refresh(existing)
    return existing


def register_autoencoder_model(
    db: Session,
    *,
    source_run: str,
    model_file: str,
    metadata_file: str,
    scores_file: str,
    feature_file: str,
    report_file: str,
    metrics: dict[str, Any],
) -> ModelRegistry:
    normalized = artifacts.normalize_source_run(source_run)
    run_token = artifacts.normalize_run_token(normalized)
    existing = (
        db.query(ModelRegistry)
        .filter(
            ModelRegistry.model_family == "UNSUPERVISED",
            ModelRegistry.algorithm == "autoencoder_pytorch",
            ModelRegistry.source_run == normalized,
        )
        .first()
    )
    payload = {
        "run_token": run_token,
        "model_file": model_file,
        "metadata_file": metadata_file,
        "report_file": report_file,
        "scores_file": scores_file,
        "feature_file": feature_file,
        "metrics_json": json.dumps(metrics, ensure_ascii=True, sort_keys=True),
        "status": "AVAILABLE",
        "is_active": False,
    }
    if existing is None:
        existing = ModelRegistry(
            model_family="UNSUPERVISED",
            algorithm="autoencoder_pytorch",
            source_run=normalized,
            **payload,
        )
        db.add(existing)
    else:
        for key, value in payload.items():
            setattr(existing, key, value)
    db.commit()
    db.refresh(existing)
    return existing


def register_supervised_human_model(
    db: Session,
    *,
    source_run: str,
    algorithm: str,
    model_file: str,
    metadata_file: str,
    report_file: str,
    predictions_file: str,
    feature_file: str,
    metrics: dict[str, Any],
) -> ModelRegistry:
    normalized = artifacts.normalize_source_run(source_run)
    run_token = artifacts.normalize_run_token(normalized)
    existing = (
        db.query(ModelRegistry)
        .filter(
            ModelRegistry.model_family == "SUPERVISED_HUMAN",
            ModelRegistry.algorithm == algorithm,
            ModelRegistry.source_run == normalized,
        )
        .first()
    )
    payload = {
        "run_token": run_token,
        "model_file": model_file,
        "metadata_file": metadata_file,
        "report_file": report_file,
        "scores_file": predictions_file,
        "feature_file": feature_file,
        "metrics_json": json.dumps(metrics, ensure_ascii=True, sort_keys=True),
        "status": "AVAILABLE",
        "is_active": False,
    }
    if existing is None:
        existing = ModelRegistry(
            model_family="SUPERVISED_HUMAN",
            algorithm=algorithm,
            source_run=normalized,
            **payload,
        )
        db.add(existing)
    else:
        for key, value in payload.items():
            setattr(existing, key, value)
    db.commit()
    db.refresh(existing)
    return existing

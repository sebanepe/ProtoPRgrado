from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Optional

from sqlalchemy.orm import Session

from backend.app.models.models import ArtifactRegistry


ARTIFACT_PREPROCESSED_CSV = "PREPROCESSED_CSV"
ARTIFACT_PREPROCESSING_REPORT = "PREPROCESSING_REPORT"
ARTIFACT_RULE_ALERTS_CSV = "RULE_ALERTS_CSV"
ARTIFACT_RULE_SUMMARY_CSV = "RULE_SUMMARY_CSV"
ARTIFACT_RULE_REPORT = "RULE_REPORT"
ARTIFACT_RULE_FILTER_CACHE = "RULE_FILTER_CACHE"
ARTIFACT_ANOMALY_SCORES_CSV = "ANOMALY_SCORES_CSV"
ARTIFACT_ANOMALY_REPORT = "ANOMALY_REPORT"
ARTIFACT_UNSUPERVISED_FEATURE_SET = "UNSUPERVISED_FEATURE_SET"
ARTIFACT_MODEL_PICKLE = "MODEL_PICKLE"
ARTIFACT_MODEL_METADATA = "MODEL_METADATA"
ARTIFACT_MODEL_ARTIFACT = "MODEL_ARTIFACT"
ARTIFACT_MODEL_SCALER = "MODEL_SCALER"
ARTIFACT_MODEL_REPORT = "MODEL_REPORT"
ARTIFACT_MODEL_PREDICTIONS = "MODEL_PREDICTIONS"
ARTIFACT_SUPERVISED_DATASET = "SUPERVISED_DATASET"
ARTIFACT_SUPERVISED_REPORT = "SUPERVISED_REPORT"
ARTIFACT_SUPERVISED_PREDICTIONS = "SUPERVISED_PREDICTIONS"
ARTIFACT_SUPERVISED_INFERENCE_SCORES = "SUPERVISED_INFERENCE_SCORES"

ARTIFACT_SCORING_RESULTS  = "SCORING_RESULTS_CSV"
ARTIFACT_SCORING_REPORT   = "SCORING_REPORT"
ARTIFACT_SCORING_METADATA = "SCORING_METADATA"

PHASE_A  = "PHASE_A"
PHASE_B  = "PHASE_B"
PHASE_C3 = "PHASE_C3"
PHASE_C4 = "PHASE_C4"
PHASE_D1 = "PHASE_D1"


def normalize_run_token(source_run: Any) -> str:
    value = str(source_run or "").strip()
    match = re.search(r"run_(\d+)", value)
    if match:
        return match.group(1)
    match = re.search(r"(\d+)$", value)
    if match:
        return match.group(1)
    return value or "UNKNOWN"


def normalize_source_run(source_run: Any) -> str:
    value = str(source_run or "").strip()
    if not value:
        raise ValueError("source_run is required")
    token = normalize_run_token(value)
    if value.startswith("preprocessed_run_"):
        return value
    if token and token != "UNKNOWN":
        return f"preprocessed_run_{token}"
    return value


def default_processed_dir() -> Path:
    return Path(os.environ.get("PROJECT_PROCESSED_DIR") or Path.cwd() / "data" / "processed")


def default_models_dir() -> Path:
    return Path(os.environ.get("PROJECT_MODELS_DIR") or Path.cwd() / "data" / "models")


def calculate_file_checksum(file_path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(file_path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_file_size(file_path: str | os.PathLike[str]) -> int:
    return int(Path(file_path).stat().st_size)


def count_csv_rows(file_path: str | os.PathLike[str]) -> int:
    path = Path(file_path)
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        line_count = sum(1 for _ in handle)
    return max(line_count - 1, 0)


def _json_dumps(payload: Optional[dict[str, Any]]) -> Optional[str]:
    if payload is None:
        return None
    return json.dumps(payload, ensure_ascii=True, sort_keys=True)


def _json_loads(value: Optional[str]) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def artifact_to_dict(artifact: ArtifactRegistry) -> dict[str, Any]:
    return {
        "id": artifact.id,
        "artifact_type": artifact.artifact_type,
        "phase": artifact.phase,
        "source_run": artifact.source_run,
        "run_token": artifact.run_token,
        "file_path": artifact.file_path,
        "file_name": artifact.file_name,
        "row_count": artifact.row_count,
        "checksum": artifact.checksum,
        "file_size_bytes": artifact.file_size_bytes,
        "status": artifact.status,
        "metadata_json": _json_loads(artifact.metadata_json),
        "created_at": artifact.created_at.isoformat() if artifact.created_at else None,
        "updated_at": artifact.updated_at.isoformat() if artifact.updated_at else None,
    }


def register_artifact(
    db: Session,
    *,
    artifact_type: str,
    phase: str,
    source_run: str,
    file_path: str | os.PathLike[str],
    run_token: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> ArtifactRegistry:
    source_run = normalize_source_run(source_run)
    run_token = run_token or normalize_run_token(source_run)
    path = Path(file_path)
    exists = path.exists()
    row_count = count_csv_rows(path) if exists and path.suffix.lower() == ".csv" else None
    artifact = ArtifactRegistry(
        artifact_type=artifact_type,
        phase=phase,
        source_run=source_run,
        run_token=run_token,
        file_path=str(path),
        file_name=path.name,
        row_count=row_count,
        checksum=calculate_file_checksum(path) if exists else None,
        file_size_bytes=get_file_size(path) if exists else None,
        status="AVAILABLE" if exists else "MISSING",
        metadata_json=_json_dumps(metadata),
    )
    db.add(artifact)
    db.commit()
    db.refresh(artifact)
    return artifact


def register_or_update_artifact(
    db: Session,
    *,
    artifact_type: str,
    phase: str,
    source_run: str,
    file_path: str | os.PathLike[str],
    run_token: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> ArtifactRegistry:
    source_run = normalize_source_run(source_run)
    run_token = run_token or normalize_run_token(source_run)
    path = Path(file_path)
    existing = (
        db.query(ArtifactRegistry)
        .filter(
            ArtifactRegistry.source_run == source_run,
            ArtifactRegistry.artifact_type == artifact_type,
            ArtifactRegistry.file_path == str(path),
        )
        .first()
    )
    exists = path.exists()
    row_count = count_csv_rows(path) if exists and path.suffix.lower() == ".csv" else None
    if existing is None:
        return register_artifact(
            db,
            artifact_type=artifact_type,
            phase=phase,
            source_run=source_run,
            run_token=run_token,
            file_path=path,
            metadata=metadata,
        )

    existing.phase = phase
    existing.run_token = run_token
    existing.file_name = path.name
    existing.row_count = row_count
    existing.checksum = calculate_file_checksum(path) if exists else None
    existing.file_size_bytes = get_file_size(path) if exists else None
    existing.status = "AVAILABLE" if exists else "MISSING"
    existing.metadata_json = _json_dumps(metadata)
    db.commit()
    db.refresh(existing)
    return existing


def get_artifacts_by_source_run(db: Session, source_run: str) -> list[ArtifactRegistry]:
    normalized = normalize_source_run(source_run)
    return (
        db.query(ArtifactRegistry)
        .filter(ArtifactRegistry.source_run == normalized)
        .order_by(ArtifactRegistry.phase.asc(), ArtifactRegistry.artifact_type.asc())
        .all()
    )


def get_artifacts_by_phase(db: Session, phase: str) -> list[ArtifactRegistry]:
    return db.query(ArtifactRegistry).filter(ArtifactRegistry.phase == phase).order_by(ArtifactRegistry.created_at.desc()).all()


def get_artifact_by_type(db: Session, source_run: str, artifact_type: str) -> Optional[ArtifactRegistry]:
    normalized = normalize_source_run(source_run)
    return (
        db.query(ArtifactRegistry)
        .filter(ArtifactRegistry.source_run == normalized, ArtifactRegistry.artifact_type == artifact_type)
        .order_by(ArtifactRegistry.updated_at.desc(), ArtifactRegistry.id.desc())
        .first()
    )


def list_artifacts(
    db: Session,
    *,
    source_run: Optional[str] = None,
    phase: Optional[str] = None,
    artifact_type: Optional[str] = None,
) -> list[ArtifactRegistry]:
    query = db.query(ArtifactRegistry)
    if source_run:
        query = query.filter(ArtifactRegistry.source_run == normalize_source_run(source_run))
    if phase:
        query = query.filter(ArtifactRegistry.phase == phase)
    if artifact_type:
        query = query.filter(ArtifactRegistry.artifact_type == artifact_type)
    return query.order_by(ArtifactRegistry.created_at.desc(), ArtifactRegistry.id.desc()).all()


def artifact_specs(source_run: str, processed_dir: Path | None = None, models_dir: Path | None = None) -> list[dict[str, Any]]:
    source_run = normalize_source_run(source_run)
    run_token = normalize_run_token(source_run)
    processed = processed_dir or default_processed_dir()
    models = models_dir or default_models_dir()
    return [
        {"artifact_type": ARTIFACT_PREPROCESSED_CSV, "phase": PHASE_A, "path": processed / f"preprocessed_run_{run_token}.csv"},
        {"artifact_type": ARTIFACT_PREPROCESSING_REPORT, "phase": PHASE_A, "path": processed / f"preprocessing_report_run_{run_token}.md"},
        {"artifact_type": ARTIFACT_RULE_ALERTS_CSV, "phase": PHASE_B, "path": processed / f"alerts_run_{run_token}.csv"},
        {"artifact_type": ARTIFACT_RULE_SUMMARY_CSV, "phase": PHASE_B, "path": processed / f"alerts_summary_run_{run_token}.csv"},
        {"artifact_type": ARTIFACT_RULE_REPORT, "phase": PHASE_B, "path": processed / f"rules_report_run_{run_token}.md"},
        {"artifact_type": ARTIFACT_RULE_FILTER_CACHE, "phase": PHASE_B, "path": processed / f"summary_filter_options_run_{run_token}.json"},
        {"artifact_type": ARTIFACT_UNSUPERVISED_FEATURE_SET, "phase": PHASE_C3, "path": processed / f"unsupervised_feature_set_run_{run_token}.csv"},
        {"artifact_type": ARTIFACT_ANOMALY_SCORES_CSV, "phase": PHASE_C3, "path": processed / f"anomaly_scores_run_{run_token}.csv"},
        {"artifact_type": ARTIFACT_ANOMALY_REPORT, "phase": PHASE_C3, "path": processed / f"anomaly_report_run_{run_token}.md"},
        {"artifact_type": ARTIFACT_MODEL_PICKLE, "phase": PHASE_C3, "path": models / f"isolation_forest_run_{run_token}.pkl"},
        {"artifact_type": ARTIFACT_MODEL_METADATA, "phase": PHASE_C3, "path": models / f"isolation_forest_run_{run_token}_metadata.json"},
    ]


def scan_existing_artifacts(
    db: Session,
    source_run: str,
    *,
    processed_dir: str | os.PathLike[str] | None = None,
    models_dir: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    normalized = normalize_source_run(source_run)
    run_token = normalize_run_token(normalized)
    registered: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    warnings: list[str] = []
    processed = Path(processed_dir) if processed_dir else None
    models = Path(models_dir) if models_dir else None

    for spec in artifact_specs(normalized, processed, models):
        path = spec["path"]
        metadata = {"scan_source": "register_existing_artifacts"}
        artifact = register_or_update_artifact(
            db,
            artifact_type=spec["artifact_type"],
            phase=spec["phase"],
            source_run=normalized,
            run_token=run_token,
            file_path=path,
            metadata=metadata,
        )
        payload = artifact_to_dict(artifact)
        if artifact.status == "AVAILABLE":
            registered.append(payload)
        else:
            missing.append(payload)
            warnings.append(f"Missing {artifact.artifact_type}: {artifact.file_path}")

    return {
        "source_run": normalized,
        "run_token": run_token,
        "registered_count": len(registered),
        "missing_count": len(missing),
        "registered": registered,
        "missing": missing,
        "warnings": warnings,
    }

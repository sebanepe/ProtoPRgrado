from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from sqlalchemy.orm import Session

from backend.app.ml.supervised_dataset_builder import (
    build_human_supervised_alert_dataset,
    get_latest_supervised_dataset_run,
)
from backend.app.ml.validate_human_supervised_dataset import (
    INSUFFICIENT_CLASSES,
    READY,
    validate_human_supervised_dataset,
)
from backend.app.services import artifact_registry_service as artifacts
from backend.app.services import supervised_service


def _dataset_path(source_run: str, db: Session) -> Path:
    run = get_latest_supervised_dataset_run(db, source_run)
    if run and run.dataset_file:
        return Path(run.dataset_file)
    token = artifacts.normalize_run_token(source_run)
    return artifacts.default_processed_dir() / f"supervised_human_alert_dataset_run_{token}.csv"


def run_training_preflight(
    source_run: str,
    db: Session,
    *,
    build_if_missing: bool = False,
) -> dict[str, Any]:
    normalized = artifacts.normalize_source_run(source_run)
    label_summary = supervised_service.get_human_label_summary(db, source_run=normalized)
    human_labels = {
        "confirmed_fraud": label_summary["confirmed_fraud"],
        "dismissed": label_summary["dismissed"],
        "usable_total": label_summary["usable_total_labels"],
        "technical_ready": label_summary["technical_ready"],
        "recommended_ready": label_summary["recommended_ready"],
        "strong_ready": label_summary["strong_ready"],
    }
    warnings: list[str] = []
    build_result: Optional[dict[str, Any]] = None

    path = _dataset_path(normalized, db)
    if not path.exists() and build_if_missing and human_labels["technical_ready"]:
        build_result = build_human_supervised_alert_dataset(normalized, db=db)
        warnings.extend(build_result.get("warnings", []))
        path = _dataset_path(normalized, db)

    dataset_exists = path.exists()
    validation = validate_human_supervised_dataset(path) if dataset_exists else {
        "verdict": "FILE_NOT_FOUND",
        "row_count": 0,
        "positive_count": 0,
        "negative_count": 0,
        "errors": [],
        "warnings": [],
    }
    warnings.extend(validation.get("warnings", []))

    dataset = {
        "exists": dataset_exists,
        "file": path.name,
        "path": str(path),
        "rows": int(validation.get("row_count") or 0),
        "positive_count": int(validation.get("positive_count") or 0),
        "negative_count": int(validation.get("negative_count") or 0),
        "verdict": validation.get("verdict"),
        "errors": validation.get("errors", []),
    }

    dataset_artifact = artifacts.get_artifact_by_type(db, normalized, artifacts.ARTIFACT_SUPERVISED_DATASET)
    report_artifact = artifacts.get_artifact_by_type(db, normalized, artifacts.ARTIFACT_SUPERVISED_REPORT)
    run = get_latest_supervised_dataset_run(db, normalized)

    blocking_reason = None
    if not human_labels["technical_ready"]:
        blocking_reason = "INSUFFICIENT_HUMAN_LABELS"
    elif not dataset_exists:
        blocking_reason = "SUPERVISED_DATASET_NOT_FOUND"
    elif validation.get("verdict") == INSUFFICIENT_CLASSES:
        blocking_reason = "SINGLE_CLASS_DATASET"
    elif validation.get("verdict") != READY:
        blocking_reason = "SUPERVISED_DATASET_INVALID"

    return {
        "source_run": normalized,
        "human_labels": human_labels,
        "dataset": dataset,
        "artifact_registry": {
            "supervised_dataset_registered": bool(dataset_artifact and dataset_artifact.status == "AVAILABLE"),
            "supervised_report_registered": bool(report_artifact and report_artifact.status == "AVAILABLE"),
        },
        "supervised_dataset_runs": {
            "registered": run is not None,
            "status": run.status if run else None,
        },
        "can_train": blocking_reason is None,
        "blocking_reason": blocking_reason,
        "warnings": warnings,
        "build_attempted": build_result is not None,
        "build_result": build_result,
    }

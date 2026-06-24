from __future__ import annotations

from collections import defaultdict
from typing import Any, Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.app.models.models import (
    ArtifactRegistry,
    Dataset,
    PreprocessingRun,
    RuleAlertReview,
    RuleRun,
    User,
)
from backend.app.services.artifact_registry_service import (
    ARTIFACT_PREPROCESSED_CSV,
    ARTIFACT_RULE_ALERTS_CSV,
    ARTIFACT_RULE_SUMMARY_CSV,
)

_RELEVANT_ARTIFACT_TYPES = [
    ARTIFACT_PREPROCESSED_CSV,
    ARTIFACT_RULE_ALERTS_CSV,
    ARTIFACT_RULE_SUMMARY_CSV,
]

_REVIEW_STATUSES = [
    "NEW",
    "IN_REVIEW",
    "DISMISSED",
    "FALSE_POSITIVE",
    "CONFIRMED_FRAUD",
]

# Sentinel: rule engine artifacts exist in artifact_registry but no rule_runs record.
# This happens when the rule engine ran but the registration step failed or was skipped.
_STATUS_DERIVED = "DERIVADO"


def _iso(dt) -> Optional[str]:
    return dt.isoformat() if dt is not None else None


def _user_label(user_by_id: dict[int, str], user_id: Optional[int]) -> Optional[str]:
    if user_id is None:
        return None
    return user_by_id.get(user_id)


def _count_reviews_by_status(db: Session, source_run: str) -> dict[str, dict[str, int]]:
    """
    Returns latest-status counts per alert type for a given source_run.
    Uses MAX(id) per alert_id / summary_alert_id as tiebreaker (monotonic insert order).
    Produces: {"detailed": {status: count}, "grouped": {status: count}}
    """
    results = {
        "detailed": {s: 0 for s in _REVIEW_STATUSES},
        "grouped": {s: 0 for s in _REVIEW_STATUSES},
    }

    # Detailed alerts (keyed by alert_id)
    latest_ids_detail = (
        db.query(func.max(RuleAlertReview.id))
        .filter(
            RuleAlertReview.source_run == source_run,
            RuleAlertReview.alert_id.isnot(None),
        )
        .group_by(RuleAlertReview.alert_id)
        .subquery()
    )
    detail_counts = (
        db.query(RuleAlertReview.new_status, func.count().label("cnt"))
        .filter(RuleAlertReview.id.in_(db.query(latest_ids_detail)))
        .group_by(RuleAlertReview.new_status)
        .all()
    )
    for status, cnt in detail_counts:
        if status in results["detailed"]:
            results["detailed"][status] = cnt

    # Grouped / summary alerts (keyed by summary_alert_id)
    latest_ids_grouped = (
        db.query(func.max(RuleAlertReview.id))
        .filter(
            RuleAlertReview.source_run == source_run,
            RuleAlertReview.summary_alert_id.isnot(None),
        )
        .group_by(RuleAlertReview.summary_alert_id)
        .subquery()
    )
    grouped_counts = (
        db.query(RuleAlertReview.new_status, func.count().label("cnt"))
        .filter(RuleAlertReview.id.in_(db.query(latest_ids_grouped)))
        .group_by(RuleAlertReview.new_status)
        .all()
    )
    for status, cnt in grouped_counts:
        if status in results["grouped"]:
            results["grouped"][status] = cnt

    return results


def get_import_alert_summary(db: Session) -> list[dict[str, Any]]:
    """
    Read-only cross-phase traceability pivot.

    Returns one row per (Dataset, PreprocessingRun) combination.
    Datasets with no preprocessing run appear as a single row with null downstream fields.

    Relationship chain:
      datasets
        → preprocessing_runs (via input_dataset_id FK)
          → rule_runs (via source_run = "preprocessed_run_" + str(pr.id))
            fallback: artifact_registry (RULE_ALERTS_CSV / RULE_SUMMARY_CSV)
            → rule_alert_reviews (via source_run — review states only)
            → artifact_registry (via source_run + artifact_type)

    Fallback for missing rule_runs record: if no RuleRun exists but RULE_ALERTS_CSV
    or RULE_SUMMARY_CSV are AVAILABLE in artifact_registry, the counts are derived
    from artifact.row_count and rule_run_status is set to "DERIVADO".

    Never exposes: is_fraud, confirmed_fraud (as generated field), PAN_TARJETA,
    TARJETA, password*, pan_card, raw_card.
    Never modifies data. Never creates alerts or reviews.
    """
    # --- Batch load to avoid N+1 ---
    datasets = db.query(Dataset).order_by(Dataset.created_at.desc()).all()
    if not datasets:
        return []

    dataset_ids = [d.id for d in datasets]

    # All preprocessing runs grouped by dataset id
    all_prep_runs = (
        db.query(PreprocessingRun)
        .filter(PreprocessingRun.input_dataset_id.in_(dataset_ids))
        .order_by(PreprocessingRun.id.asc())
        .all()
    )
    prep_by_dataset: dict[int, list[PreprocessingRun]] = defaultdict(list)
    for pr in all_prep_runs:
        prep_by_dataset[pr.input_dataset_id].append(pr)

    # All source_run strings from preprocessing runs
    all_source_runs = [f"preprocessed_run_{pr.id}" for pr in all_prep_runs]

    # All rule runs indexed by source_run
    rule_run_by_source: dict[str, RuleRun] = {}
    if all_source_runs:
        for rr in (
            db.query(RuleRun)
            .filter(RuleRun.source_run.in_(all_source_runs))
            .all()
        ):
            rule_run_by_source[rr.source_run] = rr

    # All relevant artifacts indexed by (source_run, artifact_type) → (status, row_count)
    # row_count is needed to derive alert counts when no RuleRun record exists.
    artifact_data: dict[tuple[str, str], tuple[str, int]] = {}
    if all_source_runs:
        for ar in (
            db.query(ArtifactRegistry)
            .filter(
                ArtifactRegistry.source_run.in_(all_source_runs),
                ArtifactRegistry.artifact_type.in_(_RELEVANT_ARTIFACT_TYPES),
            )
            .all()
        ):
            key = (ar.source_run, ar.artifact_type)
            existing = artifact_data.get(key)
            # Keep the AVAILABLE entry if one exists; otherwise take the first
            if existing is None or existing[0] != "AVAILABLE":
                artifact_data[key] = (ar.status, ar.row_count or 0)

    # Batch-load users referenced by datasets and preprocessing runs
    user_ids: set[int] = set()
    for d in datasets:
        if d.uploaded_by_id:
            user_ids.add(d.uploaded_by_id)
    for pr in all_prep_runs:
        if pr.executed_by_id:
            user_ids.add(pr.executed_by_id)

    user_by_id: dict[int, str] = {}
    if user_ids:
        for u in db.query(User).filter(User.id.in_(user_ids)).all():
            label = u.full_name or u.email or str(u.id)
            user_by_id[u.id] = label

    # --- Assemble rows ---
    rows: list[dict[str, Any]] = []

    for dataset in datasets:
        prep_runs = prep_by_dataset.get(dataset.id, [])

        if not prep_runs:
            rows.append(_build_row(dataset, None, None, {}, artifact_data, user_by_id, None))
            continue

        for pr in prep_runs:
            source_run = f"preprocessed_run_{pr.id}"
            rule_run = rule_run_by_source.get(source_run)

            review_counts: dict[str, dict[str, int]] = {}
            if rule_run or _has_rule_artifacts(artifact_data, source_run):
                review_counts = _count_reviews_by_status(db, source_run)

            rows.append(_build_row(dataset, pr, rule_run, review_counts, artifact_data, user_by_id, source_run))

    return rows


def _has_rule_artifacts(artifact_data: dict, source_run: str) -> bool:
    """True if RULE_ALERTS_CSV or RULE_SUMMARY_CSV is AVAILABLE for this source_run."""
    for atype in (ARTIFACT_RULE_ALERTS_CSV, ARTIFACT_RULE_SUMMARY_CSV):
        entry = artifact_data.get((source_run, atype))
        if entry and entry[0] == "AVAILABLE":
            return True
    return False


def _build_row(
    dataset: Dataset,
    pr: Optional[PreprocessingRun],
    rule_run: Optional[RuleRun],
    review_counts: dict[str, dict[str, int]],
    artifact_data: dict[tuple[str, str], tuple[str, int]],
    user_by_id: dict[int, str],
    source_run: Optional[str],
) -> dict[str, Any]:
    detail_r = review_counts.get("detailed", {s: 0 for s in _REVIEW_STATUSES})
    grouped_r = review_counts.get("grouped", {s: 0 for s in _REVIEW_STATUSES})

    # Resolve rule run fields, with fallback to artifact-derived counts
    if rule_run is not None:
        rule_run_id = rule_run.id
        rule_run_status = rule_run.status
        rule_run_created_at = _iso(rule_run.created_at)
        detailed_alert_count = rule_run.detailed_alert_count
        grouped_alert_count = rule_run.grouped_alert_count
    elif source_run and _has_rule_artifacts(artifact_data, source_run):
        # No rule_run DB record, but artifacts exist in artifact_registry.
        # Derive counts from artifact row_count (set when files were registered).
        rule_run_id = None
        rule_run_status = _STATUS_DERIVED
        rule_run_created_at = None
        alerts_entry = artifact_data.get((source_run, ARTIFACT_RULE_ALERTS_CSV), ("MISSING", 0))
        summary_entry = artifact_data.get((source_run, ARTIFACT_RULE_SUMMARY_CSV), ("MISSING", 0))
        detailed_alert_count = alerts_entry[1] if alerts_entry[0] == "AVAILABLE" else 0
        grouped_alert_count = summary_entry[1] if summary_entry[0] == "AVAILABLE" else 0
    else:
        rule_run_id = None
        rule_run_status = None
        rule_run_created_at = None
        detailed_alert_count = 0
        grouped_alert_count = 0

    def art_status(atype: str) -> str:
        entry = artifact_data.get((source_run, atype)) if source_run else None
        return entry[0] if entry else "MISSING"

    return {
        # Dataset
        "dataset_id": dataset.id,
        "dataset_name": dataset.name,
        "dataset_filename": dataset.original_filename,
        "dataset_status": dataset.status,
        "dataset_total_records": dataset.total_records,
        "dataset_created_at": _iso(dataset.created_at),
        "dataset_uploaded_by": _user_label(user_by_id, dataset.uploaded_by_id),
        # Preprocessing run
        "preprocessing_run_id": pr.id if pr else None,
        "preprocessing_run_status": pr.status if pr else None,
        "preprocessing_processed_records": pr.processed_records if pr else None,
        "preprocessing_started_at": _iso(pr.started_at) if pr else None,
        "preprocessing_finished_at": _iso(pr.finished_at) if pr else None,
        "preprocessing_executed_by": _user_label(user_by_id, pr.executed_by_id) if pr else None,
        # Rule run
        "rule_run_id": rule_run_id,
        "rule_run_status": rule_run_status,
        "rule_run_created_at": rule_run_created_at,
        "detailed_alert_count": detailed_alert_count,
        "grouped_alert_count": grouped_alert_count,
        # Review counts — detailed alerts (per analyst action on alert_id)
        "detailed_new_count": detail_r.get("NEW", 0),
        "detailed_in_review_count": detail_r.get("IN_REVIEW", 0),
        "detailed_dismissed_count": detail_r.get("DISMISSED", 0),
        "detailed_false_positive_count": detail_r.get("FALSE_POSITIVE", 0),
        "detailed_confirmed_by_review_count": detail_r.get("CONFIRMED_FRAUD", 0),
        # Review counts — grouped/summary alerts (per analyst action on summary_alert_id)
        "grouped_new_count": grouped_r.get("NEW", 0),
        "grouped_in_review_count": grouped_r.get("IN_REVIEW", 0),
        "grouped_dismissed_count": grouped_r.get("DISMISSED", 0),
        "grouped_false_positive_count": grouped_r.get("FALSE_POSITIVE", 0),
        "grouped_confirmed_by_review_count": grouped_r.get("CONFIRMED_FRAUD", 0),
        # Artifact health
        "artifact_preprocessed_csv": art_status(ARTIFACT_PREPROCESSED_CSV),
        "artifact_rule_alerts_csv": art_status(ARTIFACT_RULE_ALERTS_CSV),
        "artifact_rule_summary_csv": art_status(ARTIFACT_RULE_SUMMARY_CSV),
    }

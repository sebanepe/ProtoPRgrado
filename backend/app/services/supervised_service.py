from __future__ import annotations

from typing import Any, Optional

from sqlalchemy.orm import Session

from backend.app.models.models import RuleAlertReview


TECHNICAL_POSITIVE_REQUIRED = 20
TECHNICAL_NEGATIVE_REQUIRED = 20
RECOMMENDED_POSITIVE_REQUIRED = 50
RECOMMENDED_NEGATIVE_REQUIRED = 120
STRONG_POSITIVE_TARGET = 70
STRONG_NEGATIVE_TARGET = 180

USABLE_POSITIVE_STATUS = "CONFIRMED_FRAUD"
USABLE_NEGATIVE_STATUS = "DISMISSED"
EXCLUDED_FALSE_POSITIVE_STATUS = "FALSE_POSITIVE"
EXCLUDED_NEW_STATUS = "NEW"
EXCLUDED_IN_REVIEW_STATUS = "IN_REVIEW"


def _normalize_status(value: Any) -> str:
    if value is None:
        return "UNKNOWN"
    text = str(value).strip().upper()
    return text or "UNKNOWN"


def _missing(current: int, required: int) -> int:
    return max(required - current, 0)


def _readiness_verdict(technical_ready: bool, recommended_ready: bool, strong_ready: bool) -> str:
    if strong_ready:
        return "HUMAN_LABELS_STRONG_READY"
    if recommended_ready:
        return "HUMAN_LABELS_RECOMMENDED_READY"
    if technical_ready:
        return "HUMAN_LABELS_TECHNICALLY_READY"
    return "INSUFFICIENT_HUMAN_LABELS"


def get_human_label_summary(db: Session, source_run: Optional[str] = None) -> dict[str, Any]:
    query = db.query(RuleAlertReview)
    if source_run:
        query = query.filter(RuleAlertReview.source_run == source_run)

    status_counts: dict[str, int] = {}
    total_reviews = 0
    for (status_value,) in query.with_entities(RuleAlertReview.new_status).all():
        total_reviews += 1
        status = _normalize_status(status_value)
        status_counts[status] = status_counts.get(status, 0) + 1

    confirmed_fraud = status_counts.get(USABLE_POSITIVE_STATUS, 0)
    dismissed = status_counts.get(USABLE_NEGATIVE_STATUS, 0)
    new = status_counts.get(EXCLUDED_NEW_STATUS, 0)
    in_review = status_counts.get(EXCLUDED_IN_REVIEW_STATUS, 0)
    false_positive_excluded = status_counts.get(EXCLUDED_FALSE_POSITIVE_STATUS, 0)

    usable_positive_labels = confirmed_fraud
    usable_negative_labels = dismissed
    usable_total_labels = usable_positive_labels + usable_negative_labels
    excluded_total = max(total_reviews - usable_total_labels, 0)

    technical_ready = confirmed_fraud >= TECHNICAL_POSITIVE_REQUIRED and dismissed >= TECHNICAL_NEGATIVE_REQUIRED
    recommended_ready = confirmed_fraud >= RECOMMENDED_POSITIVE_REQUIRED and dismissed >= RECOMMENDED_NEGATIVE_REQUIRED
    strong_ready = confirmed_fraud >= STRONG_POSITIVE_TARGET and dismissed >= STRONG_NEGATIVE_TARGET
    verdict = _readiness_verdict(technical_ready, recommended_ready, strong_ready)

    return {
        "source_run": source_run,
        "total_reviews": total_reviews,
        "confirmed_fraud": confirmed_fraud,
        "dismissed": dismissed,
        "new": new,
        "in_review": in_review,
        "false_positive_excluded": false_positive_excluded,
        "usable_positive_labels": usable_positive_labels,
        "usable_negative_labels": usable_negative_labels,
        "usable_total_labels": usable_total_labels,
        "excluded_total": excluded_total,
        "technical_min_positive_required": TECHNICAL_POSITIVE_REQUIRED,
        "technical_min_negative_required": TECHNICAL_NEGATIVE_REQUIRED,
        "recommended_positive_required": RECOMMENDED_POSITIVE_REQUIRED,
        "recommended_negative_required": RECOMMENDED_NEGATIVE_REQUIRED,
        "strong_positive_target": STRONG_POSITIVE_TARGET,
        "strong_negative_target": STRONG_NEGATIVE_TARGET,
        "technical_ready": technical_ready,
        "recommended_ready": recommended_ready,
        "strong_ready": strong_ready,
        "missing_for_technical": {
            "positive": _missing(confirmed_fraud, TECHNICAL_POSITIVE_REQUIRED),
            "negative": _missing(dismissed, TECHNICAL_NEGATIVE_REQUIRED),
        },
        "missing_for_recommended": {
            "positive": _missing(confirmed_fraud, RECOMMENDED_POSITIVE_REQUIRED),
            "negative": _missing(dismissed, RECOMMENDED_NEGATIVE_REQUIRED),
        },
        "missing_for_strong": {
            "positive": _missing(confirmed_fraud, STRONG_POSITIVE_TARGET),
            "negative": _missing(dismissed, STRONG_NEGATIVE_TARGET),
        },
        "verdict": verdict,
    }


def get_human_readiness(db: Session, source_run: Optional[str] = None) -> dict[str, Any]:
    summary = get_human_label_summary(db, source_run=source_run)
    verdict = summary["verdict"]
    if verdict == "HUMAN_LABELS_STRONG_READY":
        message = "Existen etiquetas humanas suficientes para una preparacion fuerte del entrenamiento supervisado."
    elif verdict == "HUMAN_LABELS_RECOMMENDED_READY":
        message = "Existen etiquetas humanas suficientes para una preparacion recomendada del entrenamiento supervisado."
    elif verdict == "HUMAN_LABELS_TECHNICALLY_READY":
        message = "Existen etiquetas humanas suficientes para un entrenamiento supervisado tecnico minimo."
    else:
        message = "No existen suficientes etiquetas humanas para entrenar un modelo supervisado."

    return {
        "source_run": summary["source_run"],
        "technical_ready": summary["technical_ready"],
        "recommended_ready": summary["recommended_ready"],
        "strong_ready": summary["strong_ready"],
        "verdict": verdict,
        "message": message,
        "current": {
            "positive": summary["usable_positive_labels"],
            "negative": summary["usable_negative_labels"],
            "total": summary["usable_total_labels"],
        },
        "requirements": {
            "technical": {
                "positive": TECHNICAL_POSITIVE_REQUIRED,
                "negative": TECHNICAL_NEGATIVE_REQUIRED,
            },
            "recommended": {
                "positive": RECOMMENDED_POSITIVE_REQUIRED,
                "negative": RECOMMENDED_NEGATIVE_REQUIRED,
            },
            "strong": {
                "positive": STRONG_POSITIVE_TARGET,
                "negative": STRONG_NEGATIVE_TARGET,
            },
        },
        "missing": {
            "technical": summary["missing_for_technical"],
            "recommended": summary["missing_for_recommended"],
            "strong": summary["missing_for_strong"],
        },
    }

from __future__ import annotations

from backend.app.models.models import RuleAlertReview
from backend.app.services.supervised_service import get_human_label_summary, get_human_readiness


def add_reviews(db_session, source_run: str, status: str, count: int, *, rule_code: str = "RULE_TEST") -> None:
    for index in range(count):
        db_session.add(
            RuleAlertReview(
                source_run=source_run,
                summary_alert_id=f"{source_run}-S-{status}-{index}",
                rule_code=rule_code,
                new_status=status,
            )
        )
    db_session.commit()


def test_human_label_summary_empty_counts_zero(db_session):
    summary = get_human_label_summary(db_session, source_run="preprocessed_run_empty_c41")

    assert summary["total_reviews"] == 0
    assert summary["confirmed_fraud"] == 0
    assert summary["dismissed"] == 0
    assert summary["usable_total_labels"] == 0
    assert summary["excluded_total"] == 0
    assert summary["technical_ready"] is False
    assert summary["verdict"] == "INSUFFICIENT_HUMAN_LABELS"


def test_confirmed_fraud_counts_as_usable_positive(db_session):
    source_run = "preprocessed_run_positive_c41"
    add_reviews(db_session, source_run, "CONFIRMED_FRAUD", 3)

    summary = get_human_label_summary(db_session, source_run=source_run)

    assert summary["confirmed_fraud"] == 3
    assert summary["usable_positive_labels"] == 3
    assert summary["usable_negative_labels"] == 0


def test_dismissed_counts_as_usable_negative(db_session):
    source_run = "preprocessed_run_negative_c41"
    add_reviews(db_session, source_run, "DISMISSED", 4)

    summary = get_human_label_summary(db_session, source_run=source_run)

    assert summary["dismissed"] == 4
    assert summary["usable_negative_labels"] == 4
    assert summary["usable_positive_labels"] == 0


def test_excluded_statuses_are_not_usable_labels(db_session):
    source_run = "preprocessed_run_excluded_c41"
    add_reviews(db_session, source_run, "FALSE_POSITIVE", 2)
    add_reviews(db_session, source_run, "NEW", 3)
    add_reviews(db_session, source_run, "IN_REVIEW", 4)

    summary = get_human_label_summary(db_session, source_run=source_run)

    assert summary["false_positive_excluded"] == 2
    assert summary["new"] == 3
    assert summary["in_review"] == 4
    assert summary["usable_positive_labels"] == 0
    assert summary["usable_negative_labels"] == 0
    assert summary["usable_total_labels"] == 0
    assert summary["excluded_total"] == 9


def test_technical_ready_requires_20_positive_and_20_negative(db_session):
    source_run = "preprocessed_run_technical_c41"
    add_reviews(db_session, source_run, "CONFIRMED_FRAUD", 20)
    add_reviews(db_session, source_run, "DISMISSED", 20)

    summary = get_human_label_summary(db_session, source_run=source_run)

    assert summary["technical_ready"] is True
    assert summary["recommended_ready"] is False
    assert summary["strong_ready"] is False
    assert summary["missing_for_technical"] == {"positive": 0, "negative": 0}
    assert summary["verdict"] == "HUMAN_LABELS_TECHNICALLY_READY"


def test_recommended_ready_requires_50_positive_and_120_negative(db_session):
    source_run = "preprocessed_run_recommended_c41"
    add_reviews(db_session, source_run, "CONFIRMED_FRAUD", 50)
    add_reviews(db_session, source_run, "DISMISSED", 120)

    summary = get_human_label_summary(db_session, source_run=source_run)

    assert summary["technical_ready"] is True
    assert summary["recommended_ready"] is True
    assert summary["strong_ready"] is False
    assert summary["verdict"] == "HUMAN_LABELS_RECOMMENDED_READY"


def test_strong_ready_requires_70_positive_and_180_negative(db_session):
    source_run = "preprocessed_run_strong_c41"
    add_reviews(db_session, source_run, "CONFIRMED_FRAUD", 70)
    add_reviews(db_session, source_run, "DISMISSED", 180)

    summary = get_human_label_summary(db_session, source_run=source_run)

    assert summary["technical_ready"] is True
    assert summary["recommended_ready"] is True
    assert summary["strong_ready"] is True
    assert summary["verdict"] == "HUMAN_LABELS_STRONG_READY"


def test_readiness_insufficient_and_technical_verdicts(db_session):
    empty = get_human_readiness(db_session, source_run="preprocessed_run_readiness_empty_c41")
    assert empty["verdict"] == "INSUFFICIENT_HUMAN_LABELS"
    assert empty["current"] == {"positive": 0, "negative": 0, "total": 0}

    source_run = "preprocessed_run_readiness_technical_c41"
    add_reviews(db_session, source_run, "CONFIRMED_FRAUD", 20)
    add_reviews(db_session, source_run, "DISMISSED", 20)
    ready = get_human_readiness(db_session, source_run=source_run)

    assert ready["verdict"] == "HUMAN_LABELS_TECHNICALLY_READY"
    assert ready["technical_ready"] is True
    assert ready["requirements"]["technical"] == {"positive": 20, "negative": 20}


def test_non_human_fields_are_not_labels(db_session):
    source_run = "preprocessed_run_no_proxy_c41"
    db_session.add(
        RuleAlertReview(
            source_run=source_run,
            summary_alert_id="summary-rule-score-anomaly",
            rule_code="CONFIRMED_FRAUD",
            new_status="NEW",
            analyst_notes="risk_score=0.99 anomaly_flag=1",
        )
    )
    db_session.commit()

    summary = get_human_label_summary(db_session, source_run=source_run)

    assert summary["usable_total_labels"] == 0
    assert summary["confirmed_fraud"] == 0
    assert summary["dismissed"] == 0
    assert "is_fraud" not in summary
    assert "confirmed_fraud_auto" not in summary
    assert "anomaly_flag" not in summary
    assert "rule_code" not in summary

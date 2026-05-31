from __future__ import annotations

from backend.app.models.models import RuleAlertReview


def add_reviews(db_session, source_run: str, status: str, count: int) -> None:
    for index in range(count):
        db_session.add(
            RuleAlertReview(
                source_run=source_run,
                summary_alert_id=f"{source_run}-S-{status}-{index}",
                rule_code="RULE_TEST",
                new_status=status,
            )
        )
    db_session.commit()


def test_human_label_summary_endpoint_returns_zero_counts(test_client):
    response = test_client.get(
        "/api/supervised/human-label-summary",
        params={"source_run": "preprocessed_run_empty_route_c41"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["source_run"] == "preprocessed_run_empty_route_c41"
    assert payload["total_reviews"] == 0
    assert payload["usable_total_labels"] == 0
    assert payload["verdict"] == "INSUFFICIENT_HUMAN_LABELS"


def test_human_label_summary_endpoint_counts_only_human_usable_labels(test_client, db_session):
    source_run = "preprocessed_run_counts_route_c41"
    add_reviews(db_session, source_run, "CONFIRMED_FRAUD", 1)
    add_reviews(db_session, source_run, "DISMISSED", 2)
    add_reviews(db_session, source_run, "FALSE_POSITIVE", 3)
    add_reviews(db_session, source_run, "NEW", 4)
    add_reviews(db_session, source_run, "IN_REVIEW", 5)

    response = test_client.get("/api/supervised/human-label-summary", params={"source_run": source_run})

    assert response.status_code == 200
    payload = response.json()
    assert payload["confirmed_fraud"] == 1
    assert payload["dismissed"] == 2
    assert payload["false_positive_excluded"] == 3
    assert payload["new"] == 4
    assert payload["in_review"] == 5
    assert payload["usable_positive_labels"] == 1
    assert payload["usable_negative_labels"] == 2
    assert payload["usable_total_labels"] == 3
    assert payload["excluded_total"] == 12
    assert "is_fraud" not in payload
    assert "anomaly_flag" not in payload
    assert "rule_code" not in payload


def test_human_readiness_endpoint_returns_insufficient(test_client):
    response = test_client.get(
        "/api/supervised/human-readiness",
        params={"source_run": "preprocessed_run_insufficient_route_c41"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["verdict"] == "INSUFFICIENT_HUMAN_LABELS"
    assert payload["technical_ready"] is False
    assert payload["current"] == {"positive": 0, "negative": 0, "total": 0}
    assert payload["missing"]["technical"] == {"positive": 20, "negative": 20}


def test_human_readiness_endpoint_returns_technical_ready(test_client, db_session):
    source_run = "preprocessed_run_technical_route_c41"
    add_reviews(db_session, source_run, "CONFIRMED_FRAUD", 20)
    add_reviews(db_session, source_run, "DISMISSED", 20)

    response = test_client.get("/api/supervised/human-readiness", params={"source_run": source_run})

    assert response.status_code == 200
    payload = response.json()
    assert payload["verdict"] == "HUMAN_LABELS_TECHNICALLY_READY"
    assert payload["technical_ready"] is True
    assert payload["recommended_ready"] is False
    assert payload["strong_ready"] is False


def test_human_readiness_endpoint_returns_recommended_ready(test_client, db_session):
    source_run = "preprocessed_run_recommended_route_c41"
    add_reviews(db_session, source_run, "CONFIRMED_FRAUD", 50)
    add_reviews(db_session, source_run, "DISMISSED", 120)

    response = test_client.get("/api/supervised/human-readiness", params={"source_run": source_run})

    assert response.status_code == 200
    payload = response.json()
    assert payload["verdict"] == "HUMAN_LABELS_RECOMMENDED_READY"
    assert payload["technical_ready"] is True
    assert payload["recommended_ready"] is True
    assert payload["strong_ready"] is False


def test_human_readiness_endpoint_returns_strong_ready(test_client, db_session):
    source_run = "preprocessed_run_strong_route_c41"
    add_reviews(db_session, source_run, "CONFIRMED_FRAUD", 70)
    add_reviews(db_session, source_run, "DISMISSED", 180)

    response = test_client.get("/api/supervised/human-readiness", params={"source_run": source_run})

    assert response.status_code == 200
    payload = response.json()
    assert payload["verdict"] == "HUMAN_LABELS_STRONG_READY"
    assert payload["technical_ready"] is True
    assert payload["recommended_ready"] is True
    assert payload["strong_ready"] is True

from __future__ import annotations

from pathlib import Path

import pandas as pd

from backend.app.models.models import RuleAlertReview
from backend.app.services import artifact_registry_service as artifacts


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


def write_summary_artifacts(db_session, tmp_path: Path, token: str) -> None:
    source_run = f"preprocessed_run_{token}"
    summary = tmp_path / f"alerts_summary_run_{token}.csv"
    alerts = tmp_path / f"alerts_run_{token}.csv"
    pd.DataFrame(
        [
            {
                "summary_alert_id": f"{token}-S-pos",
                "source_run": token,
                "customer_hash": "cust-pos",
                "rule_code": "RULE_DOUBLE_COUNTRY",
                "rule_name": "Double Country",
                "risk_level": "HIGH",
                "max_risk_score": 80,
                "count_transactions": 2,
                "countries_detected": "BO|PE",
                "merchant_rubro_proxy": "5814",
                "merchant_rubro_values": "5814",
                "window_start": "2026-01-01T00:00:00Z",
                "window_end": "2026-01-01T00:30:00Z",
                "representative_transaction_id": "tx-pos",
                "status": "NEW",
                "TARJETA": "sensitive",
            },
            {
                "summary_alert_id": f"{token}-S-neg",
                "source_run": token,
                "customer_hash": "cust-neg",
                "rule_code": "RULE_MCC_RISK",
                "rule_name": "MCC Risk",
                "risk_level": "MEDIUM",
                "max_risk_score": 55,
                "count_transactions": 1,
                "countries_detected": "BO",
                "merchant_rubro_proxy": "7995",
                "merchant_rubro_values": "7995",
                "window_start": "2026-01-02T00:00:00Z",
                "window_end": "2026-01-02T00:30:00Z",
                "representative_transaction_id": "tx-neg",
                "status": "NEW",
            },
        ]
    ).to_csv(summary, index=False)
    pd.DataFrame([{"alert_id": "a1"}]).to_csv(alerts, index=False)
    artifacts.register_or_update_artifact(
        db_session,
        artifact_type=artifacts.ARTIFACT_RULE_SUMMARY_CSV,
        phase=artifacts.PHASE_B,
        source_run=source_run,
        file_path=summary,
    )
    artifacts.register_or_update_artifact(
        db_session,
        artifact_type=artifacts.ARTIFACT_RULE_ALERTS_CSV,
        phase=artifacts.PHASE_B,
        source_run=source_run,
        file_path=alerts,
    )


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


def test_build_human_dataset_route_and_related_endpoints(test_client, db_session, tmp_path):
    token = "931"
    source_run = f"preprocessed_run_{token}"
    write_summary_artifacts(db_session, tmp_path, token)
    db_session.add(
        RuleAlertReview(
            source_run=source_run,
            summary_alert_id=f"{token}-S-pos",
            rule_code="RULE_DOUBLE_COUNTRY",
            new_status="CONFIRMED_FRAUD",
        )
    )
    db_session.add(
        RuleAlertReview(
            source_run=source_run,
            summary_alert_id=f"{token}-S-neg",
            rule_code="RULE_MCC_RISK",
            new_status="DISMISSED",
        )
    )
    db_session.add(
        RuleAlertReview(
            source_run=source_run,
            summary_alert_id=f"{token}-S-fp",
            rule_code="RULE_MCC_RISK",
            new_status="FALSE_POSITIVE",
        )
    )
    db_session.commit()

    build = test_client.post("/api/supervised/build-human-dataset", json={"source_run": source_run})
    assert build.status_code == 200
    payload = build.json()
    assert payload["verdict"] == "HUMAN_SUPERVISED_DATASET_CREATED"
    assert payload["usable_positive_labels"] == 1
    assert payload["usable_negative_labels"] == 1

    summary = test_client.get("/api/supervised/human-dataset-summary", params={"source_run": source_run})
    assert summary.status_code == 200
    assert summary.json()["supervised_dataset_run_id"] == payload["supervised_dataset_run_id"]

    preview = test_client.get("/api/supervised/human-dataset-preview", params={"source_run": source_run, "limit": 5})
    assert preview.status_code == 200
    rows = preview.json()["rows"]
    assert len(rows) == 2
    assert "TARJETA" not in rows[0]
    assert {row["target_human_label"] for row in rows} == {0, 1}

    report = test_client.get("/api/supervised/human-dataset-report", params={"source_run": source_run})
    assert report.status_code == 200
    assert "CONFIRMED_FRAUD se usa como clase positiva" in report.json()["markdown"]

    validation = test_client.get("/api/supervised/human-dataset-validate", params={"source_run": source_run})
    assert validation.status_code == 200
    assert validation.json()["verdict"] == "HUMAN_SUPERVISED_DATASET_READY"


def test_build_human_dataset_route_handles_no_usable_labels(test_client, db_session, tmp_path):
    token = "932"
    source_run = f"preprocessed_run_{token}"
    write_summary_artifacts(db_session, tmp_path, token)
    add_reviews(db_session, source_run, "IN_REVIEW", 1)

    response = test_client.post("/api/supervised/build-human-dataset", json={"source_run": source_run})

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "NOT_CREATED"
    assert payload["verdict"] == "DATASET_NOT_CREATED_INSUFFICIENT_HUMAN_LABELS"

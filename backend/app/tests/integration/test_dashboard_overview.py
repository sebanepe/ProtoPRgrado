import json

import pandas as pd

from backend.app.models.models import RuleAlertReview


def _write_dashboard_artifacts(tmp_path):
    processed_dir = tmp_path / "processed"
    models_dir = tmp_path / "models"
    processed_dir.mkdir()
    models_dir.mkdir()

    pd.DataFrame(
        [
            {
                "summary_alert_id": "26-S-001",
                "source_run": "26",
                "customer_hash": "cust_a",
                "rule_code": "RULE_A",
                "risk_level": "HIGH",
                "max_risk_score": 90,
                "window_start": "2026-04-14T10:00:00+00:00",
                "created_at": "2026-05-30T20:00:00+00:00",
                "status": "NEW",
                "pan_card": "4111111111111111",
                "is_fraud": True,
                "confirmed_fraud": True,
            },
            {
                "summary_alert_id": "26-S-002",
                "source_run": "26",
                "customer_hash": "cust_b",
                "rule_code": "RULE_B",
                "risk_level": "MEDIUM",
                "max_risk_score": 70,
                "window_start": "2026-04-15T10:00:00+00:00",
                "created_at": "2026-05-31T20:00:00+00:00",
                "status": "NEW",
            },
        ]
    ).to_csv(processed_dir / "alerts_summary_run_26.csv", index=False)

    pd.DataFrame(
        [
            {
                "anomaly_run_id": "anomaly_run_26",
                "source_run": "26",
                "transaction_id": "tx_1",
                "customer_hash": "cust_a",
                "transaction_datetime": "2026-04-14 10:00:00",
                "amount": 100,
                "country_code": "BO",
                "pos_entry_mode": "5",
                "merchant_rubro_proxy": "6011",
                "anomaly_model_name": "isolation_forest",
                "anomaly_score": 0.1,
                "anomaly_rank": 1,
                "anomaly_flag": 1,
                "anomaly_percentile": 100,
            },
            {
                "anomaly_run_id": "anomaly_run_26",
                "source_run": "26",
                "transaction_id": "tx_2",
                "customer_hash": "cust_b",
                "transaction_datetime": "2026-04-15 10:00:00",
                "amount": 20,
                "country_code": "BO",
                "pos_entry_mode": "5",
                "merchant_rubro_proxy": "5411",
                "anomaly_model_name": "isolation_forest",
                "anomaly_score": 0.0,
                "anomaly_rank": 2,
                "anomaly_flag": 0,
                "anomaly_percentile": 50,
            },
        ]
    ).to_csv(processed_dir / "anomaly_scores_run_26.csv", index=False)

    (models_dir / "isolation_forest_run_26_metadata.json").write_text(
        json.dumps(
            {
                "source_run": "preprocessed_run_26",
                "model_type": "isolation_forest",
                "algorithm": "isolation_forest",
                "total_rows": 2,
                "anomaly_count": 1,
                "anomaly_rate": 0.5,
            }
        ),
        encoding="utf-8",
    )

    return processed_dir, models_dir


def test_dashboard_overview_returns_real_artifact_metrics(test_client, db_session, monkeypatch, tmp_path):
    processed_dir, models_dir = _write_dashboard_artifacts(tmp_path)
    monkeypatch.setenv("PROJECT_PROCESSED_DIR", str(processed_dir))
    monkeypatch.setenv("PROJECT_MODELS_DIR", str(models_dir))

    db_session.add(
        RuleAlertReview(
            source_run="preprocessed_run_26",
            summary_alert_id="26-S-002",
            rule_code="RULE_B",
            new_status="DISMISSED",
        )
    )
    db_session.commit()

    response = test_client.get(
        "/api/dashboard/overview",
        params={"source_run": "preprocessed_run_26", "anomaly_run": "run_26"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["total_transactions"] == 2
    assert data["active_alerts"] == 1
    assert data["average_risk_score"] == 80
    assert data["active_model"]["model_name"] == "Isolation Forest"
    assert data["active_model"]["run_id"] == "run_26"
    assert data["active_model"]["anomaly_count"] == 1
    assert data["review_distribution"]["dismissed"] == 1
    assert data["recent_alerts"]
    assert data["alerts_evolution"]
    serialized = json.dumps(data)
    assert "4111111111111111" not in serialized
    assert "pan_card" not in serialized
    assert "is_fraud" not in serialized
    assert data["review_distribution"]["confirmed_fraud"] == 0


def test_dashboard_overview_returns_partial_data_when_alert_file_missing(test_client, monkeypatch, tmp_path):
    processed_dir, models_dir = _write_dashboard_artifacts(tmp_path)
    (processed_dir / "alerts_summary_run_26.csv").unlink()
    monkeypatch.setenv("PROJECT_PROCESSED_DIR", str(processed_dir))
    monkeypatch.setenv("PROJECT_MODELS_DIR", str(models_dir))

    response = test_client.get("/api/dashboard/overview", params={"source_run": "preprocessed_run_26", "anomaly_run": "run_26"})

    assert response.status_code == 200
    data = response.json()
    assert data["total_transactions"] == 2
    assert data["recent_alerts"] == []
    assert data["warnings"]

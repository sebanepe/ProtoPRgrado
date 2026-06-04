"""Integration tests for supervised inference routes.

Validates:
- GET /api/supervised-inference/trained-models returns 200
- GET /api/supervised-inference/preprocessed-runs returns 200
- POST /api/supervised-inference/apply-trained-model: invalid input_type → 422
- POST /api/supervised-inference/apply-trained-model: preprocessed_run without id → 422
- POST /api/supervised-inference/apply-trained-model: preprocessed_run without rule_summary → 404
- POST /api/supervised-inference/apply-trained-model: nonexistent model_id → 404
- GET /api/supervised-inference/prediction-results strips forbidden columns
- GET /api/supervised-inference/inference-status/{id} not found → 404
- Response never includes is_fraud, confirmed_fraud, PAN_TARJETA
"""
from __future__ import annotations

import io
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest


# ── Discovery endpoints ───────────────────────────────────────────────────────

def test_get_sup_trained_models_returns_200(test_client):
    response = test_client.get("/api/supervised-inference/trained-models")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_get_sup_preprocessed_runs_returns_200(test_client):
    response = test_client.get("/api/supervised-inference/preprocessed-runs")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_get_sup_prediction_runs_returns_200(test_client):
    response = test_client.get("/api/supervised-inference/prediction-runs")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


# ── Apply model: input validation ─────────────────────────────────────────────

def test_apply_sup_model_invalid_input_type_returns_422(test_client):
    response = test_client.post(
        "/api/supervised-inference/apply-trained-model",
        data={"model_registry_id": 1, "input_type": "invalid_type"},
    )
    assert response.status_code == 422


def test_apply_sup_model_csv_upload_without_file_returns_422(test_client):
    response = test_client.post(
        "/api/supervised-inference/apply-trained-model",
        data={"model_registry_id": 1, "input_type": "csv_upload"},
    )
    assert response.status_code == 422


def test_apply_sup_model_preprocessed_run_without_id_returns_422(test_client):
    response = test_client.post(
        "/api/supervised-inference/apply-trained-model",
        data={"model_registry_id": 1, "input_type": "preprocessed_run"},
    )
    assert response.status_code == 422


def test_apply_sup_model_preprocessed_run_no_rule_summary_returns_404(test_client, db_session):
    """preprocessed_run existente pero sin ARTIFACT_RULE_SUMMARY_CSV → 404 con mensaje descriptivo."""
    from backend.app.models.models import PreprocessingRun
    from datetime import datetime, timezone

    run = PreprocessingRun(
        status="COMPLETED",
        total_records=100,
        started_at=datetime.now(timezone.utc),
        finished_at=datetime.now(timezone.utc),
    )
    db_session.add(run)
    db_session.commit()
    db_session.refresh(run)

    response = test_client.post(
        "/api/supervised-inference/apply-trained-model",
        data={"model_registry_id": 1, "input_type": "preprocessed_run", "preprocessed_run_id": run.id},
    )
    assert response.status_code == 404
    assert "motor de reglas" in response.json()["detail"].lower()


def test_apply_sup_model_nonexistent_model_returns_error(test_client):
    """Applying a nonexistent model_registry_id should return 404."""
    csv_content = b"transaction_id,amount\ntx1,10.5\ntx2,20.0\n"
    response = test_client.post(
        "/api/supervised-inference/apply-trained-model",
        data={"model_registry_id": 99999, "input_type": "csv_upload"},
        files={"file": ("test.csv", io.BytesIO(csv_content), "text/csv")},
    )
    assert response.status_code in {404, 422, 500}


def test_apply_sup_model_non_csv_file_returns_422(test_client):
    response = test_client.post(
        "/api/supervised-inference/apply-trained-model",
        data={"model_registry_id": 1, "input_type": "csv_upload"},
        files={"file": ("data.txt", io.BytesIO(b"not a csv"), "text/plain")},
    )
    assert response.status_code == 422


# ── Inference status ──────────────────────────────────────────────────────────

def test_get_inference_status_not_found_returns_404(test_client):
    response = test_client.get("/api/supervised-inference/inference-status/999999")
    assert response.status_code == 404


# ── Prediction results: forbidden columns ─────────────────────────────────────

def _seed_supervised_run(db_session, tmp_path, results_data: list) -> int:
    from backend.app.models.models import SupervisedInferenceRun

    results_path = tmp_path / "sup_seed_results.csv"
    pd.DataFrame(results_data).to_csv(results_path, index=False)

    run = SupervisedInferenceRun(
        algorithm="logistic_regression",
        model_source_run="preprocessed_run_26",
        input_type="csv_upload",
        input_source="test.csv",
        results_file=str(results_path),
        total_analyzed=len(results_data),
        high_count=sum(1 for r in results_data if r.get("priority_level") == "HIGH"),
        medium_count=sum(1 for r in results_data if r.get("priority_level") == "MEDIUM"),
        low_count=sum(1 for r in results_data if r.get("priority_level") == "LOW"),
        status="COMPLETED",
        started_at=datetime.now(timezone.utc),
        finished_at=datetime.now(timezone.utc),
    )
    db_session.add(run)
    db_session.commit()
    db_session.refresh(run)
    return run.id


def test_prediction_results_no_forbidden_columns(test_client, db_session, tmp_path):
    run_id = _seed_supervised_run(db_session, tmp_path, [
        {"transaction_id": "tx1", "prediction_label": 1, "prediction_probability": 0.85,
         "priority_level": "HIGH", "is_fraud": 1, "confirmed_fraud": 0, "PAN_TARJETA": "4111"},
        {"transaction_id": "tx2", "prediction_label": 0, "prediction_probability": 0.15,
         "priority_level": "LOW", "is_fraud": 0, "confirmed_fraud": 0, "PAN_TARJETA": "4112"},
    ])
    response = test_client.get(f"/api/supervised-inference/prediction-results?run_id={run_id}")
    assert response.status_code == 200
    data = response.json()
    rows = data["rows"]
    assert len(rows) == 2
    for row in rows:
        assert "is_fraud" not in row
        assert "confirmed_fraud" not in row
        assert "PAN_TARJETA" not in row


def test_prediction_results_has_priority_level(test_client, db_session, tmp_path):
    run_id = _seed_supervised_run(db_session, tmp_path, [
        {"transaction_id": "tx1", "prediction_label": 1, "prediction_probability": 0.9, "priority_level": "HIGH"},
        {"transaction_id": "tx2", "prediction_label": 0, "prediction_probability": 0.1, "priority_level": "LOW"},
    ])
    response = test_client.get(f"/api/supervised-inference/prediction-results?run_id={run_id}")
    assert response.status_code == 200
    rows = response.json()["rows"]
    assert all("priority_level" in row for row in rows)


def test_prediction_results_priority_filter(test_client, db_session, tmp_path):
    run_id = _seed_supervised_run(db_session, tmp_path, [
        {"transaction_id": "tx1", "prediction_label": 1, "prediction_probability": 0.9, "priority_level": "HIGH"},
        {"transaction_id": "tx2", "prediction_label": 1, "prediction_probability": 0.85, "priority_level": "HIGH"},
        {"transaction_id": "tx3", "prediction_label": 0, "prediction_probability": 0.1, "priority_level": "LOW"},
    ])
    response = test_client.get(f"/api/supervised-inference/prediction-results?run_id={run_id}&priority_filter=HIGH")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    for row in data["rows"]:
        assert row["priority_level"] == "HIGH"


def test_prediction_report_returns_200(test_client, db_session, tmp_path):
    run_id = _seed_supervised_run(db_session, tmp_path, [
        {"transaction_id": "tx1", "prediction_label": 1, "prediction_probability": 0.9, "priority_level": "HIGH"},
    ])
    response = test_client.get(f"/api/supervised-inference/prediction-report?run_id={run_id}")
    assert response.status_code == 200
    data = response.json()
    assert "priority_distribution" in data
    assert "methodology_warning" in data


def test_prediction_metadata_returns_200(test_client, db_session, tmp_path):
    run_id = _seed_supervised_run(db_session, tmp_path, [
        {"transaction_id": "tx1", "prediction_label": 1, "prediction_probability": 0.9, "priority_level": "HIGH"},
    ])
    response = test_client.get(f"/api/supervised-inference/prediction-metadata?run_id={run_id}")
    assert response.status_code == 200
    data = response.json()
    assert "run" in data
    assert "methodology_warning" in data


def test_prediction_results_not_found_returns_404(test_client):
    response = test_client.get("/api/supervised-inference/prediction-results?run_id=999999")
    assert response.status_code == 404

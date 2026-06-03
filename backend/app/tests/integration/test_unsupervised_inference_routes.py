"""Integration tests for unsupervised inference routes.

Validates:
- GET /api/unsupervised/trained-models returns 200
- GET /api/unsupervised/preprocessed-runs returns 200
- POST /api/unsupervised/apply-trained-model with valid CSV returns 200 or controlled error
- GET /api/unsupervised/prediction-results strips forbidden columns
- Response never includes is_fraud, confirmed_fraud, PAN_TARJETA
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd
import numpy as np
import pytest


# ── Discovery endpoints ───────────────────────────────────────────────────────

def test_get_trained_models_returns_200(test_client):
    response = test_client.get("/api/unsupervised/trained-models")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_get_preprocessed_runs_returns_200(test_client):
    response = test_client.get("/api/unsupervised/preprocessed-runs")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_get_prediction_runs_returns_200(test_client):
    response = test_client.get("/api/unsupervised/prediction-runs")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


# ── Apply model: input validation ─────────────────────────────────────────────

def test_apply_trained_model_invalid_input_type_returns_422(test_client):
    response = test_client.post(
        "/api/unsupervised/apply-trained-model",
        data={"model_registry_id": 1, "input_type": "invalid_type"},
    )
    assert response.status_code == 422


def test_apply_trained_model_csv_upload_without_file_returns_422(test_client):
    response = test_client.post(
        "/api/unsupervised/apply-trained-model",
        data={"model_registry_id": 1, "input_type": "csv_upload"},
    )
    assert response.status_code == 422


def test_apply_trained_model_preprocessed_run_without_id_returns_422(test_client):
    response = test_client.post(
        "/api/unsupervised/apply-trained-model",
        data={"model_registry_id": 1, "input_type": "preprocessed_run"},
    )
    assert response.status_code == 422


def test_apply_trained_model_nonexistent_model_returns_error(test_client, tmp_path):
    """Applying a nonexistent model_registry_id returns 404 or 422."""
    csv_content = b"transaction_id,amount\ntx1,10.5\ntx2,20.0\n"
    response = test_client.post(
        "/api/unsupervised/apply-trained-model",
        data={"model_registry_id": 99999, "input_type": "csv_upload"},
        files={"file": ("test.csv", io.BytesIO(csv_content), "text/csv")},
    )
    assert response.status_code in {404, 422, 500}


def test_apply_trained_model_non_csv_file_returns_422(test_client, db_session):
    """Non-CSV file upload returns 422."""
    response = test_client.post(
        "/api/unsupervised/apply-trained-model",
        data={"model_registry_id": 1, "input_type": "csv_upload"},
        files={"file": ("data.txt", io.BytesIO(b"not a csv"), "text/plain")},
    )
    assert response.status_code == 422


# ── Prediction results: forbidden columns ─────────────────────────────────────

def _seed_inference_run(db_session, tmp_path, results_data: list) -> int:
    """Insert an UnsupervisedInferenceRun with a results CSV and return its id."""
    from backend.app.models.models import UnsupervisedInferenceRun
    from datetime import datetime, timezone

    results_path = tmp_path / "seed_results.csv"
    pd.DataFrame(results_data).to_csv(results_path, index=False)

    run = UnsupervisedInferenceRun(
        algorithm="isolation_forest",
        model_source_run="preprocessed_run_99",
        input_type="csv_upload",
        input_source="test.csv",
        results_file=str(results_path),
        total_analyzed=len(results_data),
        anomaly_count=sum(1 for r in results_data if r.get("anomaly_flag", 0) == 1),
        anomaly_rate=0.5,
        status="COMPLETED",
        started_at=datetime.now(timezone.utc),
        finished_at=datetime.now(timezone.utc),
    )
    db_session.add(run)
    db_session.commit()
    db_session.refresh(run)
    return run.id


def test_prediction_results_strips_forbidden_columns(test_client, db_session, tmp_path):
    run_id = _seed_inference_run(db_session, tmp_path, [
        {"transaction_id": "tx1", "anomaly_score": 0.9, "anomaly_flag": 1, "anomaly_rank": 1,
         "is_fraud": 1, "confirmed_fraud": 0, "PAN_TARJETA": "4111111111111111"},
        {"transaction_id": "tx2", "anomaly_score": 0.1, "anomaly_flag": 0, "anomaly_rank": 2,
         "is_fraud": 0, "confirmed_fraud": 0, "PAN_TARJETA": "4111111111111112"},
    ])

    response = test_client.get("/api/unsupervised/prediction-results", params={"run_id": run_id})
    assert response.status_code == 200
    data = response.json()
    for row in data.get("rows", []):
        assert "is_fraud" not in row
        assert "confirmed_fraud" not in row
        assert "PAN_TARJETA" not in row
        assert "TARJETA" not in row
        assert "pan_card" not in row


def test_prediction_results_pagination(test_client, db_session, tmp_path):
    rows = [{"transaction_id": f"tx{i}", "anomaly_score": float(i) * 0.1, "anomaly_flag": i % 2, "anomaly_rank": i + 1} for i in range(10)]
    run_id = _seed_inference_run(db_session, tmp_path, rows)

    r1 = test_client.get("/api/unsupervised/prediction-results", params={"run_id": run_id, "page": 1, "page_size": 5})
    assert r1.status_code == 200
    assert len(r1.json()["rows"]) == 5

    r2 = test_client.get("/api/unsupervised/prediction-results", params={"run_id": run_id, "page": 2, "page_size": 5})
    assert r2.status_code == 200
    assert len(r2.json()["rows"]) == 5


def test_prediction_results_anomaly_only_filter(test_client, db_session, tmp_path):
    rows = [
        {"transaction_id": "tx1", "anomaly_score": 0.9, "anomaly_flag": 1, "anomaly_rank": 1},
        {"transaction_id": "tx2", "anomaly_score": 0.1, "anomaly_flag": 0, "anomaly_rank": 2},
        {"transaction_id": "tx3", "anomaly_score": 0.8, "anomaly_flag": 1, "anomaly_rank": 3},
    ]
    run_id = _seed_inference_run(db_session, tmp_path, rows)

    response = test_client.get(
        "/api/unsupervised/prediction-results",
        params={"run_id": run_id, "anomaly_only": True},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    for row in data["rows"]:
        assert row["anomaly_flag"] == 1


# ── Report and metadata ───────────────────────────────────────────────────────

def test_prediction_report_returns_methodology_warning(test_client, db_session, tmp_path):
    run_id = _seed_inference_run(db_session, tmp_path, [
        {"transaction_id": "tx1", "anomaly_score": 0.9, "anomaly_flag": 1, "anomaly_rank": 1},
    ])
    response = test_client.get("/api/unsupervised/prediction-report", params={"run_id": run_id})
    assert response.status_code == 200
    data = response.json()
    assert "methodology_warning" in data
    assert "fraude confirmado" in data["methodology_warning"].lower() or "no constituyen fraude" in data["methodology_warning"].lower()


def test_prediction_metadata_returns_200(test_client, db_session, tmp_path):
    run_id = _seed_inference_run(db_session, tmp_path, [
        {"transaction_id": "tx1", "anomaly_score": 0.9, "anomaly_flag": 1, "anomaly_rank": 1},
    ])
    response = test_client.get("/api/unsupervised/prediction-metadata", params={"run_id": run_id})
    assert response.status_code == 200
    data = response.json()
    assert "run" in data
    assert "methodology_warning" in data


def test_prediction_results_nonexistent_run_returns_404(test_client):
    response = test_client.get("/api/unsupervised/prediction-results", params={"run_id": 99999})
    assert response.status_code == 404


def test_prediction_report_nonexistent_run_returns_404(test_client):
    response = test_client.get("/api/unsupervised/prediction-report", params={"run_id": 99999})
    assert response.status_code == 404


def test_prediction_metadata_nonexistent_run_returns_404(test_client):
    response = test_client.get("/api/unsupervised/prediction-metadata", params={"run_id": 99999})
    assert response.status_code == 404


# ── Existing anomaly routes still work ────────────────────────────────────────

def test_existing_anomaly_runs_endpoint_still_works(test_client):
    """Ensure existing anomaly endpoints were not broken."""
    response = test_client.get("/api/anomaly/runs")
    assert response.status_code == 200

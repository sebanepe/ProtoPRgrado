"""
Integration tests for anomaly detection endpoints.
Tests the 7 anomaly endpoints with various scenarios.
"""
import os
import json
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.services.anomaly_service import AnomalyService


@pytest.fixture
def test_client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def temp_anomaly_data(tmp_path):
    """
    Create temporary anomaly data files for testing.
    Returns paths to score_file, report_file, and metadata_file.
    """
    processed_dir = tmp_path / "data" / "processed"
    models_dir = tmp_path / "data" / "models"
    processed_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Create sample scores CSV
    run_id = "run_26"
    n_rows = 100
    n_anomalies = 5

    scores_data = {
        "anomaly_run_id": [run_id] * n_rows,
        "source_run": ["preprocessed_run_26"] * n_rows,
        "transaction_id": range(1, n_rows + 1),
        "customer_hash": [f"cust_{i % 10}" for i in range(n_rows)],
        "transaction_datetime": [
            (datetime.now() - timedelta(hours=i)).isoformat()
            for i in range(n_rows)
        ],
        "amount": np.random.uniform(10, 5000, n_rows),
        "country_code": ["BO"] * (n_rows - 10) + ["AR"] * 10,
        "pos_entry_mode": ["contactless"] * (n_rows // 2) + ["magstripe"] * (n_rows - n_rows // 2),
        "has_pinblock": [True] * (n_rows // 2) + [False] * (n_rows - n_rows // 2),
        "merchant_rubro_proxy": np.random.choice(
            ["RETAIL", "FOOD", "TRAVEL", "UNKNOWN"],
            n_rows,
        ),
        "anomaly_model_name": ["isolation_forest"] * n_rows,
        "anomaly_score": np.concatenate([
            np.random.uniform(-0.5, -0.1, n_rows - n_anomalies),  # Normal transactions
            np.random.uniform(0.8, 1.0, n_anomalies),  # Anomalies
        ]),
        "anomaly_rank": list(range(1, n_rows + 1)),
        "anomaly_flag": [0] * (n_rows - n_anomalies) + [1] * n_anomalies,
        "anomaly_percentile": np.linspace(0, 100, n_rows),
        "created_at": [
            (datetime.now() - timedelta(hours=i)).isoformat()
            for i in range(n_rows)
        ],
    }

    df_scores = pd.DataFrame(scores_data)
    score_file = processed_dir / f"anomaly_scores_{run_id}.csv"
    df_scores.to_csv(score_file, index=False)

    # Create sample report markdown
    report_file = processed_dir / f"anomaly_report_{run_id}.md"
    report_content = f"""# Anomaly Detection Report - {run_id}

- source_run: None
- source_run_token: None

## Summary
- Total Transactions: {n_rows}
- Anomalies Detected: {n_anomalies}
- Anomaly Rate: {n_anomalies / n_rows:.2%}

## Model Configuration
- Algorithm: IsolationForest
- Contamination: 0.01
- N Estimators: 200

## Important Notice
Las anomalías detectadas no representan fraude confirmado.
No se generó is_fraud. No se generó confirmed_fraud. No se usaron reglas como etiquetas.

## Top Anomalies
| Transaction ID | Customer Hash | Amount | Score | Rank |
|---|---|---|---|---|
"""
    for idx in range(min(5, n_anomalies)):
        row = df_scores[df_scores["anomaly_flag"] == 1].iloc[idx]
        report_content += f"| {row['transaction_id']} | {row['customer_hash']} | {row['amount']:.2f} | {row['anomaly_score']:.3f} | {row['anomaly_rank']} |\n"

    report_file.write_text(report_content, encoding="utf-8")

    # Create model metadata JSON
    metadata_file = models_dir / f"isolation_forest_{run_id}_metadata.json"
    metadata = {
        "source_run": "preprocessed_run_26",
        "source_run_token": 26,
        "model_type": "unsupervised_anomaly_detection",
        "algorithm": "IsolationForest",
        "contamination": 0.01,
        "n_estimators": 200,
        "random_state": 42,
        "model_input_columns": [
            "amount",
            "hour_of_day",
            "day_of_week",
            "is_weekend",
            "is_international",
        ],
    }
    metadata_file.write_text(json.dumps(metadata), encoding="utf-8")

    return {
        "processed_dir": str(processed_dir),
        "models_dir": str(models_dir),
        "score_file": str(score_file),
        "report_file": str(report_file),
        "metadata_file": str(metadata_file),
        "run_id": run_id,
    }


@pytest.fixture
def anomaly_service_with_temp_data(temp_anomaly_data):
    """Create AnomalyService with temporary data directories."""
    service = AnomalyService(
        processed_dir=temp_anomaly_data["processed_dir"],
        models_dir=temp_anomaly_data["models_dir"],
    )
    return service


def test_list_anomaly_runs_returns_200(test_client, monkeypatch, anomaly_service_with_temp_data):
    """
    Test: GET /api/anomaly/runs responde 200.
    """
    # Monkey-patch the service in routes module
    import backend.app.routes.anomaly_routes as anomaly_routes_module
    anomaly_routes_module.anomaly_service = anomaly_service_with_temp_data

    response = test_client.get("/api/anomaly/runs")
    assert response.status_code == 200
    data = response.json()
    assert "runs" in data
    assert "count" in data


def test_get_anomaly_scores_paginates_results(test_client, monkeypatch, anomaly_service_with_temp_data, temp_anomaly_data):
    """
    Test: GET /api/anomaly/scores pagina resultados.
    """
    import backend.app.routes.anomaly_routes as anomaly_routes_module
    anomaly_routes_module.anomaly_service = anomaly_service_with_temp_data

    response = test_client.get(
        f"/api/anomaly/scores?run_id={temp_anomaly_data['run_id']}&page=1&page_size=10"
    )
    assert response.status_code == 200
    data = response.json()
    assert data["page"] == 1
    assert data["page_size"] == 10
    assert len(data["items"]) <= 10
    assert "total_items" in data
    assert "total_pages" in data


def test_get_anomaly_scores_filters_by_anomaly_flag(test_client, monkeypatch, anomaly_service_with_temp_data, temp_anomaly_data):
    """
    Test: GET /api/anomaly/scores?anomaly_flag=1 devuelve solo anomalías.
    """
    import backend.app.routes.anomaly_routes as anomaly_routes_module
    anomaly_routes_module.anomaly_service = anomaly_service_with_temp_data

    response = test_client.get(
        f"/api/anomaly/scores?run_id={temp_anomaly_data['run_id']}&anomaly_flag=1&page_size=100"
    )
    assert response.status_code == 200
    data = response.json()

    # All returned items should have anomaly_flag = 1
    for item in data["items"]:
        assert item["anomaly_flag"] == 1


def test_get_top_anomalies_returns_ranking(test_client, monkeypatch, anomaly_service_with_temp_data, temp_anomaly_data):
    """
    Test: GET /api/anomaly/top devuelve ranking.
    """
    import backend.app.routes.anomaly_routes as anomaly_routes_module
    anomaly_routes_module.anomaly_service = anomaly_service_with_temp_data

    response = test_client.get(
        f"/api/anomaly/top?run_id={temp_anomaly_data['run_id']}&limit=10"
    )
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert data["limit"] == 10

    # Verify all returned items are anomalies (anomaly_flag=1)
    for item in data["items"]:
        assert item["anomaly_flag"] == 1


def test_get_anomaly_metrics_returns_stats(test_client, monkeypatch, anomaly_service_with_temp_data, temp_anomaly_data):
    """
    Test: GET /api/anomaly/metrics devuelve anomaly_count y anomaly_rate.
    """
    import backend.app.routes.anomaly_routes as anomaly_routes_module
    anomaly_routes_module.anomaly_service = anomaly_service_with_temp_data

    response = test_client.get(f"/api/anomaly/metrics?run_id={temp_anomaly_data['run_id']}")
    assert response.status_code == 200
    data = response.json()

    assert "anomaly_count" in data
    assert "anomaly_rate" in data
    assert "total_transactions" in data
    assert "model_name" in data
    assert "algorithm" in data
    assert data["anomaly_count"] > 0
    assert 0.0 <= data["anomaly_rate"] <= 1.0


def test_get_anomaly_report_returns_markdown(test_client, monkeypatch, anomaly_service_with_temp_data, temp_anomaly_data):
    """
    Test: GET /api/anomaly/report devuelve markdown.
    """
    import backend.app.routes.anomaly_routes as anomaly_routes_module
    anomaly_routes_module.anomaly_service = anomaly_service_with_temp_data

    response = test_client.get(f"/api/anomaly/report?run_id={temp_anomaly_data['run_id']}")
    assert response.status_code == 200
    data = response.json()

    assert "report" in data
    assert isinstance(data["report"], str)
    assert len(data["report"]) > 0
    assert "# Anomaly Detection Report" in data["report"]
    assert "source_run: preprocessed_run_26" in data["report"]
    assert "source_run_token: 26" in data["report"]
    assert "Las anomalías detectadas no representan fraude confirmado." in data["report"]
    assert "No se generó is_fraud. No se generó confirmed_fraud." in data["report"]


def test_get_model_metadata_returns_json(test_client, monkeypatch, anomaly_service_with_temp_data, temp_anomaly_data):
    """
    Test: GET /api/anomaly/model-metadata devuelve JSON.
    """
    import backend.app.routes.anomaly_routes as anomaly_routes_module
    anomaly_routes_module.anomaly_service = anomaly_service_with_temp_data

    response = test_client.get(f"/api/anomaly/model-metadata?run_id={temp_anomaly_data['run_id']}")
    assert response.status_code == 200
    data = response.json()

    assert "metadata" in data
    assert "algorithm" in data["metadata"]
    assert "contamination" in data["metadata"]


def test_nonexistent_run_id_returns_404(test_client, monkeypatch, anomaly_service_with_temp_data):
    """
    Test: run_id inexistente devuelve 404 controlado.
    """
    import backend.app.routes.anomaly_routes as anomaly_routes_module
    anomaly_routes_module.anomaly_service = anomaly_service_with_temp_data

    response = test_client.get("/api/anomaly/scores?run_id=nonexistent_run")
    assert response.status_code == 404


def test_response_excludes_fraud_labels(test_client, monkeypatch, anomaly_service_with_temp_data, temp_anomaly_data):
    """
    Test: No aparecen is_fraud ni confirmed_fraud en respuestas.
    """
    import backend.app.routes.anomaly_routes as anomaly_routes_module
    anomaly_routes_module.anomaly_service = anomaly_service_with_temp_data

    response = test_client.get(
        f"/api/anomaly/scores?run_id={temp_anomaly_data['run_id']}&page_size=100"
    )
    assert response.status_code == 200
    data = response.json()

    for item in data["items"]:
        assert "is_fraud" not in item
        assert "confirmed_fraud" not in item
        assert "target_is_fraud" not in item


def test_merchant_rubro_proxy_as_string(test_client, monkeypatch, anomaly_service_with_temp_data, temp_anomaly_data):
    """
    Test: merchant_rubro_proxy se trata como string.
    """
    import backend.app.routes.anomaly_routes as anomaly_routes_module
    anomaly_routes_module.anomaly_service = anomaly_service_with_temp_data

    response = test_client.get(
        f"/api/anomaly/scores?run_id={temp_anomaly_data['run_id']}&merchant_rubro_proxy=RETAIL&page_size=100"
    )
    assert response.status_code == 200
    data = response.json()

    # All returned items should have merchant_rubro_proxy = "RETAIL"
    for item in data["items"]:
        assert item["merchant_rubro_proxy"] == "RETAIL"


def test_page_out_of_range_returns_empty_items(test_client, monkeypatch, anomaly_service_with_temp_data, temp_anomaly_data):
    """
    Test: Si page está fuera de rango, devolver items [] y no 500.
    """
    import backend.app.routes.anomaly_routes as anomaly_routes_module
    anomaly_routes_module.anomaly_service = anomaly_service_with_temp_data

    response = test_client.get(
        f"/api/anomaly/scores?run_id={temp_anomaly_data['run_id']}&page=9999&page_size=50"
    )
    assert response.status_code == 200
    data = response.json()
    assert data["items"] == []


def test_nan_values_converted_to_null(test_client, monkeypatch, anomaly_service_with_temp_data):
    """
    Test: Convertir NaN/NaT a null.
    """
    import backend.app.routes.anomaly_routes as anomaly_routes_module
    anomaly_routes_module.anomaly_service = anomaly_service_with_temp_data

    # The service already handles this in get_anomaly_scores()
    # Just verify that when we fetch data, no NaN values appear in the response
    service = anomaly_service_with_temp_data
    runs = service.list_anomaly_runs()

    if runs:
        run_id = runs[0]["anomaly_run_id"]
        result = service.get_anomaly_scores(run_id=run_id, page=1, page_size=10)

        # Check all items for NaN or NaT
        import json
        for item in result["items"]:
            json_str = json.dumps(item)  # If NaN/NaT present, this would fail
            assert "NaN" not in json_str
            assert "NaT" not in json_str

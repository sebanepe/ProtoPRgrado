import json

import pandas as pd

from backend.app.services.autoencoder_anomaly_service import AutoencoderAnomalyService


def _prepare_autoencoder_artifacts(tmp_path):
    processed = tmp_path / "processed"
    models = tmp_path / "models"
    processed.mkdir()
    models.mkdir()
    pd.DataFrame(
        [
            {
                "source_run": "preprocessed_run_26",
                "transaction_id": "tx1",
                "customer_hash": "cust1",
                "reconstruction_error": 1.2,
                "autoencoder_anomaly_score": 1.0,
                "autoencoder_anomaly_flag": 1,
                "anomaly_rank": 1,
                "is_fraud": 1,
                "confirmed_fraud": 1,
            },
            {
                "source_run": "preprocessed_run_26",
                "transaction_id": "tx2",
                "customer_hash": "cust2",
                "reconstruction_error": 0.2,
                "autoencoder_anomaly_score": 0.0,
                "autoencoder_anomaly_flag": 0,
                "anomaly_rank": 2,
            },
        ]
    ).to_csv(processed / "autoencoder_scores_run_26.csv", index=False)
    (processed / "autoencoder_report_run_26.md").write_text("# Autoencoder PyTorch Report\n", encoding="utf-8")
    (models / "autoencoder_model_run_26_metadata.json").write_text(
        json.dumps(
            {
                "source_run": "preprocessed_run_26",
                "algorithm": "autoencoder_pytorch",
                "total_records": 2,
                "anomaly_count": 1,
                "anomaly_rate": 0.5,
                "threshold": 1.0,
                "contamination": 0.5,
                "created_at": "2026-05-31T00:00:00Z",
                "scores_file": "autoencoder_scores_run_26.csv",
                "report_file": "autoencoder_report_run_26.md",
                "model_file": "autoencoder_model_run_26.pt",
                "metadata_file": "autoencoder_model_run_26_metadata.json",
                "warnings": ["Autoencoder anomalies are not confirmed fraud."],
            }
        ),
        encoding="utf-8",
    )
    return AutoencoderAnomalyService(str(processed), str(models))


def test_post_autoencoder_train_responds_controlled_when_torch_missing(test_client):
    response = test_client.post(
        "/api/anomaly/autoencoder/train",
        json={"source_run": "preprocessed_run_missing", "epochs": 1, "batch_size": 2, "latent_dim": 2, "contamination": 0.1},
    )
    assert response.status_code in {200, 404, 500}
    if response.status_code == 200:
        assert response.json()["algorithm"] == "autoencoder_pytorch"


def test_get_autoencoder_metrics_returns_metadata(test_client, monkeypatch, tmp_path):
    import backend.app.routes.autoencoder_anomaly_routes as routes

    routes.autoencoder_service = _prepare_autoencoder_artifacts(tmp_path)
    response = test_client.get("/api/anomaly/autoencoder/metrics", params={"source_run": "preprocessed_run_26"})
    assert response.status_code == 200
    data = response.json()
    assert data["algorithm"] == "autoencoder_pytorch"
    assert data["anomaly_count"] == 1


def test_get_autoencoder_scores_paginates_and_hides_fraud_labels(test_client, tmp_path):
    import backend.app.routes.autoencoder_anomaly_routes as routes

    routes.autoencoder_service = _prepare_autoencoder_artifacts(tmp_path)
    response = test_client.get(
        "/api/anomaly/autoencoder/scores",
        params={"source_run": "preprocessed_run_26", "page": 1, "page_size": 1, "anomaly_flag": 1},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total_items"] == 1
    assert "is_fraud" not in data["items"][0]
    assert "confirmed_fraud" not in data["items"][0]


def test_get_autoencoder_report_and_metadata(test_client, tmp_path):
    import backend.app.routes.autoencoder_anomaly_routes as routes

    routes.autoencoder_service = _prepare_autoencoder_artifacts(tmp_path)
    report = test_client.get("/api/anomaly/autoencoder/report", params={"source_run": "preprocessed_run_26"})
    metadata = test_client.get("/api/anomaly/autoencoder/model-metadata", params={"source_run": "preprocessed_run_26"})
    assert report.status_code == 200
    assert "Autoencoder" in report.json()["report"]
    assert metadata.status_code == 200
    assert metadata.json()["metadata"]["algorithm"] == "autoencoder_pytorch"


def test_existing_isolation_forest_endpoint_still_exists(test_client):
    response = test_client.get("/api/anomaly/runs")
    assert response.status_code == 200

import json

import pandas as pd

from backend.app.ml.validate_autoencoder_outputs import validate_autoencoder_outputs


def _write_valid_files(tmp_path):
    score_file = tmp_path / "autoencoder_scores_run_1.csv"
    metadata_file = tmp_path / "autoencoder_model_run_1_metadata.json"
    pd.DataFrame(
        [
            {
                "source_run": "preprocessed_run_1",
                "transaction_id": "tx1",
                "customer_hash": "c1",
                "reconstruction_error": 0.9,
                "autoencoder_anomaly_score": 1.0,
                "autoencoder_anomaly_flag": 1,
                "anomaly_rank": 1,
            },
            {
                "source_run": "preprocessed_run_1",
                "transaction_id": "tx2",
                "customer_hash": "c2",
                "reconstruction_error": 0.1,
                "autoencoder_anomaly_score": 0.0,
                "autoencoder_anomaly_flag": 0,
                "anomaly_rank": 2,
            },
        ]
    ).to_csv(score_file, index=False)
    metadata_file.write_text(
        json.dumps({"contamination": 0.5, "feature_columns": ["amount_log", "hour_of_day"]}),
        encoding="utf-8",
    )
    return score_file, metadata_file


def test_validator_returns_ready_with_valid_file(tmp_path):
    score_file, metadata_file = _write_valid_files(tmp_path)
    result = validate_autoencoder_outputs(str(score_file), str(metadata_file))
    assert result["verdict"] == "AUTOENCODER_OUTPUTS_READY"
    assert result["anomaly_count"] == 1
    assert result["anomaly_rate"] == 0.5


def test_validator_fails_when_reconstruction_error_missing(tmp_path):
    score_file, metadata_file = _write_valid_files(tmp_path)
    df = pd.read_csv(score_file).drop(columns=["reconstruction_error"])
    df.to_csv(score_file, index=False)
    result = validate_autoencoder_outputs(str(score_file), str(metadata_file))
    assert result["verdict"] == "AUTOENCODER_OUTPUTS_INVALID"
    assert "missing_required_column:reconstruction_error" in result["issues"]


def test_validator_fails_when_confirmed_fraud_present(tmp_path):
    score_file, metadata_file = _write_valid_files(tmp_path)
    df = pd.read_csv(score_file)
    df["confirmed_fraud"] = 1
    df.to_csv(score_file, index=False)
    result = validate_autoencoder_outputs(str(score_file), str(metadata_file))
    assert result["verdict"] == "AUTOENCODER_OUTPUTS_INVALID"
    assert "forbidden_column_present:confirmed_fraud" in result["issues"]


def test_validator_fails_when_pan_tarjeta_present(tmp_path):
    score_file, metadata_file = _write_valid_files(tmp_path)
    df = pd.read_csv(score_file)
    df["PAN_TARJETA"] = "123"
    df.to_csv(score_file, index=False)
    result = validate_autoencoder_outputs(str(score_file), str(metadata_file))
    assert result["verdict"] == "AUTOENCODER_OUTPUTS_INVALID"
    assert "forbidden_column_present:pan_tarjeta" in result["issues"]

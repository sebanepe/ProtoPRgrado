import importlib.util

import pandas as pd
import pytest

from backend.app.ml.autoencoder_anomaly import (
    AUTOENCODER_DEPENDENCY_ERROR,
    AutoencoderDependencyError,
    AutoencoderTabular,
    prepare_autoencoder_matrix,
    save_autoencoder_artifacts,
    train_autoencoder_model,
)


def _torch_available():
    return importlib.util.find_spec("torch") is not None


def test_autoencoder_tabular_dependency_error_is_controlled_when_torch_missing():
    if _torch_available():
        pytest.skip("PyTorch is installed in this environment")
    with pytest.raises(AutoencoderDependencyError) as exc:
        AutoencoderTabular(input_dim=3, latent_dim=2)
    assert AUTOENCODER_DEPENDENCY_ERROR in str(exc.value)


@pytest.mark.skipif(not _torch_available(), reason="PyTorch not installed")
def test_autoencoder_tabular_output_dimension_matches_input():
    import torch

    model = AutoencoderTabular(input_dim=4, latent_dim=2)
    output = model(torch.zeros((2, 4), dtype=torch.float32))
    assert tuple(output.shape) == (2, 4)


def test_prepare_autoencoder_matrix_excludes_context_and_forbidden_columns():
    df = pd.DataFrame(
        [
            {
                "source_run": "preprocessed_run_1",
                "transaction_id": "tx1",
                "customer_hash": "cust1",
                "amount": 10.0,
                "amount_log": 2.3,
                "hour_of_day": 10,
                "is_fraud": 1,
                "confirmed_fraud": 1,
                "anomaly_flag": 1,
                "PAN_TARJETA": 1234,
            }
        ]
    )
    _, _, feature_columns, context_df = prepare_autoencoder_matrix(df)
    assert "amount_log" in feature_columns
    assert "hour_of_day" in feature_columns
    assert "amount" not in feature_columns
    assert "transaction_id" not in feature_columns
    assert "customer_hash" not in feature_columns
    assert "is_fraud" not in feature_columns
    assert "confirmed_fraud" not in feature_columns
    assert "anomaly_flag" not in feature_columns
    assert "PAN_TARJETA" not in feature_columns
    assert "transaction_id" in context_df.columns
    assert "customer_hash" in context_df.columns


@pytest.mark.skipif(not _torch_available(), reason="PyTorch not installed")
def test_training_with_small_dataset_generates_reconstruction_error(tmp_path):
    feature_file = tmp_path / "features.csv"
    pd.DataFrame(
        [
            {"source_run": "preprocessed_run_1", "transaction_id": "tx1", "customer_hash": "c1", "amount": 10, "amount_log": 2.3, "hour_of_day": 1},
            {"source_run": "preprocessed_run_1", "transaction_id": "tx2", "customer_hash": "c2", "amount": 20, "amount_log": 3.0, "hour_of_day": 2},
            {"source_run": "preprocessed_run_1", "transaction_id": "tx3", "customer_hash": "c3", "amount": 200, "amount_log": 5.3, "hour_of_day": 3},
        ]
    ).to_csv(feature_file, index=False)
    result = train_autoencoder_model(str(feature_file), "preprocessed_run_1", epochs=1, batch_size=2, contamination=0.34)
    output = result["score_frame"]
    assert "reconstruction_error" in output.columns
    assert "is_fraud" not in output.columns
    assert "confirmed_fraud" not in output.columns
    assert set(output["autoencoder_anomaly_flag"].unique()).issubset({0, 1})


@pytest.mark.skipif(not _torch_available(), reason="PyTorch not installed")
def test_save_autoencoder_artifacts_keeps_autoencoder_outputs(tmp_path):
    feature_file = tmp_path / "features.csv"
    pd.DataFrame(
        [
            {"source_run": "preprocessed_run_1", "transaction_id": "tx1", "customer_hash": "c1", "amount": 10, "amount_log": 2.3, "hour_of_day": 1, "is_fraud": 1},
            {"source_run": "preprocessed_run_1", "transaction_id": "tx2", "customer_hash": "c2", "amount": 200, "amount_log": 5.3, "hour_of_day": 3, "confirmed_fraud": 1},
        ]
    ).to_csv(feature_file, index=False)
    trained = train_autoencoder_model(str(feature_file), "preprocessed_run_1", epochs=1, batch_size=2, contamination=0.5)
    result = save_autoencoder_artifacts(
        source_run="preprocessed_run_1",
        trained=trained,
        epochs=1,
        batch_size=2,
        learning_rate=0.001,
        latent_dim=16,
        contamination=0.5,
        input_path=str(feature_file),
        output_dir=tmp_path,
        models_dir=tmp_path,
    )
    scores = pd.read_csv(result["paths"]["scores_file"])
    assert "reconstruction_error" in scores.columns
    assert "autoencoder_anomaly_score" in scores.columns
    assert "autoencoder_anomaly_flag" in scores.columns
    assert "is_fraud" not in scores.columns
    assert "confirmed_fraud" not in scores.columns

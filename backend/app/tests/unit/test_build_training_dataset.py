import os
import pandas as pd
import pytest
from backend.app.ml import build_training_dataset


def make_sample_csv(path):
    df = pd.DataFrame([
        {"customer_hash": "c1", "merchant_hash": "m1", "transaction_datetime": "2026-05-01T00:00:00Z", "amount": 10.5, "is_fraud": 1},
        {"customer_hash": "c2", "merchant_hash": "m2", "transaction_datetime": "2026-05-01T01:00:00Z", "amount": 5.0, "is_fraud": 0},
    ])
    df.to_csv(path, index=False)


def test_input_or_dataset_required(tmp_path):
    with pytest.raises(SystemExit):
        build_training_dataset.build_training_dataset()


def test_build_from_csv_creates_outputs(tmp_path):
    csv_path = tmp_path / "sample.csv"
    make_sample_csv(csv_path)
    os.environ["PROJECT_PROCESSED_DIR"] = str(tmp_path)
    feature_path, report_path = build_training_dataset.build_training_dataset(input_csv=str(csv_path), chunksize=1, out_name="testrun")
    assert os.path.exists(feature_path)
    assert os.path.exists(report_path)
    df_feat = pd.read_csv(feature_path)
    assert "is_fraud" in df_feat.columns
    # forbidden columns absent
    for c in ["transaction_id", "customer_hash", "merchant_hash"]:
        assert c not in df_feat.columns

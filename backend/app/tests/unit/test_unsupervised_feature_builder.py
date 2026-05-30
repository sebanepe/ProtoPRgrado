import pandas as pd

from backend.app.ml.unsupervised_feature_builder import MODEL_INPUT_COLUMNS, build_unsupervised_features


def test_build_unsupervised_features_generates_expected_columns(tmp_path):
    source = tmp_path / "preprocessed_run_1.csv"
    pd.DataFrame(
        [
            {
                "transaction_id": "tx_1",
                "amount": 10,
                "customer_hash": "cust_a",
                "merchant_hash": "merch_a",
                "merchant_rubro_proxy": "5411",
                "country_code": "BO",
                "pos_entry_mode": 7,
                "has_pinblock": 1,
                "card_presence_type": "TP",
                "transaction_datetime": "2026-05-01T10:00:00Z",
            },
            {
                "transaction_id": "tx_2",
                "amount": 2500,
                "customer_hash": "cust_a",
                "merchant_hash": "merch_b",
                "merchant_rubro_proxy": "7995",
                "country_code": "US",
                "pos_entry_mode": 81,
                "has_pinblock": 0,
                "card_presence_type": "TNP",
                "transaction_datetime": "2026-05-01T11:00:00Z",
            },
        ]
    ).to_csv(source, index=False)

    feature_file, metadata = build_unsupervised_features(str(source), "preprocessed_run_1", output_dir=tmp_path)
    feature_frame = pd.read_csv(feature_file)

    for column in ["amount_log", "hour_of_day", "day_of_week", "is_weekend", "is_international", "tx_count_customer_1h"]:
        assert column in feature_frame.columns

    for column in ["is_fraud", "confirmed_fraud", "target_is_fraud", "rule_code", "rule_name", "alert_reason"]:
        assert column not in feature_frame.columns

    assert "customer_hash" in feature_frame.columns
    assert "transaction_id" in feature_frame.columns
    assert "customer_hash" not in metadata["model_input_columns"]
    assert all(column in feature_frame.columns for column in MODEL_INPUT_COLUMNS)

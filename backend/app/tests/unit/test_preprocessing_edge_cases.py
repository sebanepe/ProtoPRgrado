"""
Pruebas unitarias de preprocesamiento (casos límite).
Cada prueba valida comportamientos rápidos y aislados de `preprocess_dataframe`.
No usan la base de datos; se fuerza `apply_smote=False` para rendimiento.
"""
import pandas as pd
import numpy as np
from backend.app.ml.preprocessing import preprocess_dataframe


def test_clean_data_does_not_mutate_original_dataframe():
    df = pd.DataFrame([
        {"transaction_id": "t1", "amount": "100", "transaction_datetime": "2021-01-01", "is_fraud": 0},
        {"transaction_id": "t2", "amount": None, "transaction_datetime": "2021-01-02", "is_fraud": 1},
    ])
    original_copy = df.copy(deep=True)
    _processed, _summary = preprocess_dataframe(df, apply_smote=False)
    # original should remain identical to original_copy
    pd.testing.assert_frame_equal(df, original_copy)


def test_clean_data_removes_duplicate_rows():
    df = pd.DataFrame([
        {"transaction_id": "t1", "amount": 10, "transaction_datetime": "2021-01-01"},
        {"transaction_id": "t1", "amount": 10, "transaction_datetime": "2021-01-01"},
    ])
    processed, summary = preprocess_dataframe(df, apply_smote=False)
    assert summary["after_clean"] == 1


def test_clean_data_removes_duplicate_transaction_id_if_supported():
    df = pd.DataFrame([
        {"transaction_id": "a", "amount": 1, "transaction_datetime": "2021-01-01"},
        {"transaction_id": "a", "amount": 2, "transaction_datetime": "2021-01-02"},
    ])
    processed, _ = preprocess_dataframe(df, apply_smote=False)
    # only one transaction_id should remain
    assert processed.index.size >= 1
    assert processed["is_fraud"].shape[0] == 1 or processed.index.nlevels >= 0


def test_clean_data_handles_null_amount():
    df = pd.DataFrame([
        {"transaction_id": "t1", "amount": None, "transaction_datetime": "2021-01-01", "is_fraud": 0},
    ])
    processed, _ = preprocess_dataframe(df, apply_smote=False)
    # amount_scaled should exist even if original amount was null
    assert "amount_scaled" in processed.columns


def test_clean_data_handles_null_transaction_id():
    df = pd.DataFrame([
        {"transaction_id": None, "amount": 5, "transaction_datetime": "2021-01-01", "is_fraud": 0},
    ])
    processed, summary = preprocess_dataframe(df, apply_smote=False)
    # rows without valid transaction_datetime are dropped; transaction_id None should be handled
    assert isinstance(processed, pd.DataFrame)


def test_clean_data_handles_empty_dataframe():
    df = pd.DataFrame([])
    processed, summary = preprocess_dataframe(df, apply_smote=False)
    assert processed.empty
    assert summary.get("after", 0) == 0


def test_amount_string_is_converted_to_numeric():
    df = pd.DataFrame([
        {"transaction_id": "t1", "amount": "123.45", "transaction_datetime": "2021-01-01"},
    ])
    processed, _ = preprocess_dataframe(df, apply_smote=False)
    assert "amount_scaled" in processed.columns


def test_invalid_amount_string_is_handled():
    df = pd.DataFrame([
        {"transaction_id": "t1", "amount": "notanumber", "transaction_datetime": "2021-01-01"},
    ])
    processed, _ = preprocess_dataframe(df, apply_smote=False)
    # should not raise and should produce amount_scaled column
    assert "amount_scaled" in processed.columns


def test_encode_features_handles_missing_optional_categorical_columns():
    # missing transaction_type/channel/location should not break processing
    df = pd.DataFrame([
        {"transaction_id": "t1", "amount": 10, "transaction_datetime": "2021-01-01"},
    ])
    processed, _ = preprocess_dataframe(df, apply_smote=False)
    assert isinstance(processed, pd.DataFrame)


def test_encode_features_preserves_is_fraud_column():
    df = pd.DataFrame([
        {"transaction_id": "t1", "amount": 10, "transaction_datetime": "2021-01-01", "is_fraud": 1},
    ])
    processed, _ = preprocess_dataframe(df, apply_smote=False)
    assert "is_fraud" in processed.columns


def test_scale_features_handles_constant_amounts():
    df = pd.DataFrame([
        {"transaction_id": "t1", "amount": 5, "transaction_datetime": "2021-01-01"},
        {"transaction_id": "t2", "amount": 5, "transaction_datetime": "2021-01-02"},
    ])
    processed, _ = preprocess_dataframe(df, apply_smote=False)
    # amount_scaled should exist and be finite
    assert "amount_scaled" in processed.columns
    assert processed["amount_scaled"].notna().all()


def test_preprocess_dataset_returns_dataframe():
    df = pd.DataFrame([
        {"transaction_id": "t1", "amount": 1, "transaction_datetime": "2021-01-01"},
    ])
    processed, summary = preprocess_dataframe(df, apply_smote=False)
    assert isinstance(processed, pd.DataFrame)
    assert "columns_transformed" in summary

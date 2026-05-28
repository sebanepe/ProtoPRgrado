import os
import pandas as pd

from backend.app.ml import preprocessing
from backend.app.ml import validate_feature_set
from backend.app.ml.build_training_dataset import build_training_dataset


def _base_df():
    return pd.DataFrame([
        {
            "PAIS": "BO",
            "POS_ENTRY_MODE": "2",
            "TIENE_PINBLOCK": "1",
            "CODIGO_ESTABLECIMIENTO": "M001",
            "CODIGO_TERMINAL": "T001",
            "ESTABLECIMIENTO": "SHOP_A",
            "TARJETA": "4XXX",
            "PAN_TARJETA": "412345******1111",
            "monto": "100.50",
            "fecha": "2026-05-20 10:00:00",
        },
        {
            "PAIS": "BOLIVIA",
            "POS_ENTRY_MODE": "10",
            "TIENE_PINBLOCK": "0",
            "CODIGO_ESTABLECIMIENTO": "M002",
            "CODIGO_TERMINAL": "T002",
            "ESTABLECIMIENTO": "SHOP_B",
            "TARJETA": "5XXX",
            "PAN_TARJETA": "512345******2222",
            "monto": "50",
            "fecha": "2026-05-20 11:00:00",
        },
    ])


def test_mapping_and_country_code_rules():
    df = _base_df()
    cleaned, _ = preprocessing.preprocess_dataframe(df)
    assert "country_code" in cleaned.columns
    assert cleaned["country_code"].notna().any()
    assert cleaned["feature_international_transaction"].sum() == 0
    assert cleaned["is_international"].sum() == 0


def test_pos_entry_mode_and_pinblock_preserved():
    df = _base_df()
    cleaned, _ = preprocessing.preprocess_dataframe(df)
    assert cleaned["pos_entry_mode"].isna().sum() < len(cleaned)
    assert set(cleaned["has_pinblock"].dropna().astype(int).unique().tolist()) == {0, 1}
    assert (cleaned["has_pinblock_source"] == "raw").all()


def test_card_presence_unknown_when_imputed():
    df = pd.DataFrame([
        {
            "PAIS": "BO",
            "monto": "10",
            "fecha": "2026-05-20 10:00:00",
        }
    ])
    cleaned, _ = preprocessing.preprocess_dataframe(df)
    assert cleaned["card_presence_type"].iloc[0] == "UNKNOWN"
    assert int(cleaned["feature_tnp_transaction"].iloc[0]) == 0


def test_merchant_hash_varies_by_merchant_code():
    df = _base_df()
    cleaned, _ = preprocessing.preprocess_dataframe(df)
    assert cleaned["merchant_hash"].nunique() >= 2


def test_unknown_merchant_does_not_trigger_many_merchants():
    df = pd.DataFrame([
        {
            "PAIS": "BO",
            "monto": "10",
            "fecha": "2026-05-20 10:00:00",
            "TARJETA": "4XXX",
        },
        {
            "PAIS": "BO",
            "monto": "20",
            "fecha": "2026-05-20 10:05:00",
            "TARJETA": "4YYY",
        },
    ])
    cleaned, _ = preprocessing.preprocess_dataframe(df)
    assert int(cleaned["feature_many_merchants_customer_day"].sum()) == 0
    assert int(cleaned["feature_many_merchants_customer_hour"].sum()) == 0


def test_card_brand_inferred_from_tarjeta():
    df = _base_df()
    cleaned, _ = preprocessing.preprocess_dataframe(df)
    assert cleaned["card_brand"].notna().any()
    assert "VISA" in cleaned["card_brand"].values


def test_feature_set_excludes_sensitive_columns(tmp_path, monkeypatch):
    df = _base_df()
    src = tmp_path / "sample.csv"
    df.to_csv(src, index=False)
    monkeypatch.setenv("PROJECT_PROCESSED_DIR", str(tmp_path))
    out_file, _ = build_training_dataset(str(src), out_name="test_safe")
    fs = pd.read_csv(out_file)
    for forbidden in ["PAN_TARJETA", "TARJETA", "pan_card", "masked_card", "response_code", "behavioral_risk_score"]:
        assert forbidden not in fs.columns


def test_validate_feature_set_flags_full_feature(tmp_path):
    df = pd.DataFrame({
        "is_fraud": [0, 1, 0, 1],
        "feature_always_one": [1, 1, 1, 1],
        "amount": [10, 20, 30, 40],
    })
    p = tmp_path / "feature_set.csv"
    df.to_csv(p, index=False)
    report = validate_feature_set.validate(str(p))
    assert report["verdict"] == "NOT_READY"

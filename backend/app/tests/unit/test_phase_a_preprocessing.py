import pandas as pd
from datetime import datetime

from backend.app.ml.preprocessing import preprocess_dataframe
from backend.app.ml.validate_cleaned_dataset import (
    READY_VERDICT,
    NOT_READY_VERDICT,
    validate,
)
from backend.app.repositories import transaction_repository
from backend.app.models.models import Transaction
from backend.app.services import preprocessing_service
from backend.app.services.preprocessing_service import _phase_a_filter_columns


def test_phase_a_filter_columns_removes_training_columns():
    df = pd.DataFrame(
        [
            {
                "transaction_id": "tx-1",
                "amount": 120.5,
                "transaction_datetime": "2026-01-01T10:00:00",
                "card_brand": "VISA",
                "customer_hash": "cust-1",
                "is_fraud": True,
                "amount_scaled": 0.42,
                "feature_tp_pem_07": 1,
                "behavioral_risk_score": 0.9,
                "_card_product_unknown": 1,
            }
        ]
    )

    filtered, dropped = _phase_a_filter_columns(df)

    assert "is_fraud" in dropped
    assert "amount_scaled" in dropped
    assert "feature_tp_pem_07" in dropped
    assert "behavioral_risk_score" in dropped
    assert "_card_product_unknown" in dropped
    assert "is_fraud" not in filtered.columns
    assert "amount_scaled" not in filtered.columns
    assert "feature_tp_pem_07" not in filtered.columns
    assert "behavioral_risk_score" not in filtered.columns
    assert "_card_product_unknown" not in filtered.columns
    assert list(filtered.columns) == ["transaction_id", "amount", "transaction_datetime", "card_brand", "customer_hash"]


def test_country_code_canonicalization_and_international_flag():
    df = pd.DataFrame(
        [
            {"transaction_id": "tx-1", "amount": 10, "transaction_datetime": "2026-01-01T00:00:00", "country_code": "USA"},
            {"transaction_id": "tx-2", "amount": 11, "transaction_datetime": "2026-01-01T01:00:00", "country_code": "GBR"},
            {"transaction_id": "tx-3", "amount": 12, "transaction_datetime": "2026-01-01T02:00:00", "country_code": "BOL"},
            {"transaction_id": "tx-4", "amount": 13, "transaction_datetime": "2026-01-01T03:00:00", "country_code": "UNKNOWN"},
            {"transaction_id": "tx-5", "amount": 14, "transaction_datetime": "2026-01-01T04:00:00", "country_code": "US"},
        ]
    )

    cleaned, _ = preprocess_dataframe(df)

    assert list(cleaned["country_code"]) == ["US", "GB", "BO", "UNKNOWN", "US"]
    assert list(cleaned["is_international"].astype(int)) == [1, 1, 0, 0, 1]


def test_merchant_rubro_proxy_from_rubro_and_mcc():
    df = pd.DataFrame(
        [
            {"transaction_id": "tx-1", "amount": 10, "transaction_datetime": "2026-01-01T00:00:00", "RUBRO": " cajeros automaticos ", "country_code": "BO"},
            {"transaction_id": "tx-2", "amount": 11, "transaction_datetime": "2026-01-01T01:00:00", "MCC": " 6011 ", "country_code": "BO"},
        ]
    )

    cleaned, _ = preprocess_dataframe(df)

    assert "merchant_rubro_proxy" in cleaned.columns
    assert cleaned.loc[0, "merchant_rubro_proxy"] == "CAJEROS AUTOMATICOS"
    assert cleaned.loc[1, "merchant_rubro_proxy"] == "6011"


def test_merchant_rubro_proxy_numeric_aliases_and_source_columns_are_dropped():
    df = pd.DataFrame(
        [
            {
                "transaction_id": "tx-1",
                "amount": 10,
                "transaction_datetime": "2026-01-01T00:00:00",
                "MCC_CODE": 6011.0,
                "DESCRIPTION": "ATM",
                "DESCRIPTION.1": "ATM-ALT",
                "country_code": "BO",
            },
            {
                "transaction_id": "tx-2",
                "amount": 11,
                "transaction_datetime": "2026-01-01T01:00:00",
                "CODIGO_MCC": " 5812 ",
                "merchant_rubro_description": "Retail",
                "country_code": "BO",
            },
            {
                "transaction_id": "tx-3",
                "amount": 12,
                "transaction_datetime": "2026-01-01T02:00:00",
                "mcc": None,
                "country_code": "BO",
            },
        ]
    )

    cleaned, summary = preprocess_dataframe(df)

    assert list(cleaned["merchant_rubro_proxy"]) == ["6011", "5812", "UNKNOWN"]
    assert "MCC_CODE" not in cleaned.columns
    assert "CODIGO_MCC" not in cleaned.columns
    assert "DESCRIPTION" not in cleaned.columns
    assert "DESCRIPTION.1" not in cleaned.columns
    assert "merchant_rubro_description" not in cleaned.columns
    assert summary.get("merchant_rubro_source_present") is True
    assert summary.get("merchant_rubro_valid_4digit_count") == 2


def test_missing_rubro_defaults_to_unknown():
    df = pd.DataFrame(
        [
            {"transaction_id": "tx-1", "amount": 10, "transaction_datetime": "2026-01-01T00:00:00", "country_code": "BO"},
        ]
    )

    cleaned, _ = preprocess_dataframe(df)

    assert "merchant_rubro_proxy" in cleaned.columns
    assert cleaned["merchant_rubro_proxy"].iloc[0] == "UNKNOWN"


def test_validate_cleaned_dataset_accepts_phase_a_output(tmp_path):
    df = pd.DataFrame(
        [
            {
                "transaction_id": "tx-1",
                "amount": 120.5,
                "transaction_datetime": "2026-01-01T10:00:00",
                "customer_hash": "cust-1",
                "merchant_hash": "merch-1",
                "country_code": "BO",
                "pos_entry_mode": "CHIP",
                "has_pinblock": False,
                "card_presence_type": "PHYSICAL",
                "card_brand": "VISA",
                "merchant_rubro_proxy": "RETAIL",
            }
        ]
    )
    path = tmp_path / "preprocessed_run_1.csv"
    df.to_csv(path, index=False)

    report = validate(path)

    assert report["verdict"] == READY_VERDICT
    assert report["missing_required_columns"] == []
    assert report["forbidden_columns_present"] == []
    assert "customer_hash" in report["preferred_columns_present"]


def test_validate_cleaned_dataset_rejects_forbidden_columns(tmp_path):
    df = pd.DataFrame(
        [
            {
                "transaction_id": "tx-1",
                "amount": 120.5,
                "transaction_datetime": "2026-01-01T10:00:00",
                "card_brand": "VISA",
                "is_fraud": True,
            }
        ]
    )
    path = tmp_path / "preprocessed_run_2.csv"
    df.to_csv(path, index=False)

    report = validate(path)

    assert report["verdict"] == NOT_READY_VERDICT
    assert "is_fraud" in report["forbidden_columns_present"]


def test_validate_cleaned_dataset_rejects_lost_mcc_when_source_has_mcc(tmp_path):
    source_df = pd.DataFrame(
        [
            {"transaction_id": "tx-1", "amount": 120.5, "transaction_datetime": "2026-01-01T10:00:00", "MCC_CODE": 6011},
            {"transaction_id": "tx-2", "amount": 80.0, "transaction_datetime": "2026-01-01T11:00:00", "MCC_CODE": 5812},
        ]
    )
    cleaned_df = pd.DataFrame(
        [
            {
                "transaction_id": "tx-1",
                "amount": 120.5,
                "transaction_datetime": "2026-01-01T10:00:00",
                "merchant_rubro_proxy": "UNKNOWN",
            },
            {
                "transaction_id": "tx-2",
                "amount": 80.0,
                "transaction_datetime": "2026-01-01T11:00:00",
                "merchant_rubro_proxy": "UNKNOWN",
            },
        ]
    )

    source_path = tmp_path / "source.csv"
    cleaned_path = tmp_path / "preprocessed_run_99.csv"
    source_df.to_csv(source_path, index=False)
    cleaned_df.to_csv(cleaned_path, index=False)

    report = validate(cleaned_path, source_path=source_path)

    assert report["verdict"] == NOT_READY_VERDICT
    assert any("MCC_CODE" in note for note in report["notes"])


def test_transaction_repository_preserves_merchant_rubro_proxy(db_session):
    inserted = transaction_repository.insert_transactions(
        db_session,
        [
            {
                "transaction_id": "tx-repo-1",
                "amount": 25.0,
                "transaction_datetime": datetime.utcnow(),
                "merchant_rubro_proxy": "6011",
                "is_fraud": False,
            }
        ],
        dataset_id=None,
    )

    row = db_session.query(Transaction).filter(Transaction.transaction_id == "tx-repo-1").first()

    assert inserted == 1
    assert row is not None
    assert row.merchant_rubro_proxy == "6011"


def test_preprocessing_service_includes_merchant_rubro_proxy_from_db(db_session, tmp_path, monkeypatch):
    db_session.add(
        Transaction(
            transaction_id="tx-db-1",
            amount=33.0,
            transaction_datetime=datetime.utcnow(),
            merchant_rubro_proxy="5812",
            country_code="BO",
            is_fraud=False,
        )
    )
    db_session.commit()

    old_dir = preprocessing_service.PROJECT_PROCESSED_DIR
    monkeypatch.setattr(preprocessing_service, "PROJECT_PROCESSED_DIR", str(tmp_path))
    try:
        summary = preprocessing_service.run_preprocessing(db_session, output_path="", apply_smote=False)
        output_path = summary.get("project_output_path") or summary.get("output_path")
        assert output_path
        out_df = pd.read_csv(output_path)
        assert "merchant_rubro_proxy" in out_df.columns
        assert str(out_df.loc[0, "merchant_rubro_proxy"]) == "5812"
        assert "is_fraud" not in out_df.columns
        assert "confirmed_fraud" not in out_df.columns
    finally:
        monkeypatch.setattr(preprocessing_service, "PROJECT_PROCESSED_DIR", old_dir)


def test_iso3_to_iso2_normalization_extended():
    """Test extended ISO-3 to ISO-2 normalization for multiple countries."""
    df = pd.DataFrame(
        [
            {"transaction_id": "tx-1", "amount": 10, "transaction_datetime": "2026-01-01T00:00:00", "country_code": "CHL"},  # CHL -> CL
            {"transaction_id": "tx-2", "amount": 11, "transaction_datetime": "2026-01-01T01:00:00", "country_code": "MEX"},  # MEX -> MX
            {"transaction_id": "tx-3", "amount": 12, "transaction_datetime": "2026-01-01T02:00:00", "country_code": "PRY"},  # PRY -> PY
            {"transaction_id": "tx-4", "amount": 13, "transaction_datetime": "2026-01-01T03:00:00", "country_code": "DEU"},  # DEU -> DE
            {"transaction_id": "tx-5", "amount": 14, "transaction_datetime": "2026-01-01T04:00:00", "country_code": "PAN"},  # PAN -> PA
            {"transaction_id": "tx-6", "amount": 15, "transaction_datetime": "2026-01-01T05:00:00", "country_code": "COL"},  # COL -> CO
        ]
    )

    cleaned, summary = preprocess_dataframe(df)

    assert list(cleaned["country_code"]) == ["CL", "MX", "PY", "DE", "PA", "CO"]
    cc_norm = summary.get("country_code_normalization", {})
    assert cc_norm.get("iso3_normalized_count", 0) >= 6


def test_bolivia_dirty_variants_normalization():
    """Test normalization of dirty Bolivia variants (0BO, ZBO, CBO, etc.)."""
    df = pd.DataFrame(
        [
            {"transaction_id": "tx-1", "amount": 10, "transaction_datetime": "2026-01-01T00:00:00", "country_code": "0BO"},
            {"transaction_id": "tx-2", "amount": 11, "transaction_datetime": "2026-01-01T01:00:00", "country_code": "ZBO"},
            {"transaction_id": "tx-3", "amount": 12, "transaction_datetime": "2026-01-01T02:00:00", "country_code": "CBO"},
            {"transaction_id": "tx-4", "amount": 13, "transaction_datetime": "2026-01-01T03:00:00", "country_code": "PBO"},
            {"transaction_id": "tx-5", "amount": 14, "transaction_datetime": "2026-01-01T04:00:00", "country_code": "BO"},  # Already clean
        ]
    )

    cleaned, summary = preprocess_dataframe(df)

    # All should be BO
    assert list(cleaned["country_code"]) == ["BO", "BO", "BO", "BO", "BO"]
    # Check that 4 dirty variants were normalized (not the clean one)
    cc_norm = summary.get("country_code_normalization", {})
    assert cc_norm.get("bolivia_dirty_normalized_count", 0) >= 4


def test_is_international_for_bo_and_unknown():
    """Test that BO and UNKNOWN have is_international = 0."""
    df = pd.DataFrame(
        [
            {"transaction_id": "tx-1", "amount": 10, "transaction_datetime": "2026-01-01T00:00:00", "country_code": "BO"},
            {"transaction_id": "tx-2", "amount": 11, "transaction_datetime": "2026-01-01T01:00:00", "country_code": "UNKNOWN"},
            {"transaction_id": "tx-3", "amount": 12, "transaction_datetime": "2026-01-01T02:00:00", "country_code": "US"},
            {"transaction_id": "tx-4", "amount": 13, "transaction_datetime": "2026-01-01T03:00:00", "country_code": "AR"},
        ]
    )

    cleaned, summary = preprocess_dataframe(df)

    # BO and UNKNOWN should have is_international = 0
    assert cleaned.loc[0, "is_international"] == 0  # BO
    assert cleaned.loc[1, "is_international"] == 0  # UNKNOWN
    assert cleaned.loc[2, "is_international"] == 1  # US
    assert cleaned.loc[3, "is_international"] == 1  # AR
    
    # Verify checks in summary
    cc_norm = summary.get("country_code_normalization", {})
    assert cc_norm.get("bo_is_international_check") == True
    assert cc_norm.get("unknown_is_international_check") == True


def test_no_is_fraud_in_phase_a_output():
    """Test that is_fraud never appears in Phase A output."""
    df = pd.DataFrame(
        [
            {"transaction_id": "tx-1", "amount": 10, "transaction_datetime": "2026-01-01T00:00:00", "country_code": "BO"},
        ]
    )

    cleaned, summary = preprocess_dataframe(df)
    filtered, dropped = _phase_a_filter_columns(cleaned)

    assert "is_fraud" not in filtered.columns
    assert "is_fraud" in dropped  # Should be in dropped columns list


def test_merchant_rubro_proxy_with_missing_data():
    """Test that merchant_rubro_proxy defaults to UNKNOWN when no rubro data."""
    df = pd.DataFrame(
        [
            {"transaction_id": "tx-1", "amount": 10, "transaction_datetime": "2026-01-01T00:00:00", "country_code": "BO"},
            {"transaction_id": "tx-2", "amount": 11, "transaction_datetime": "2026-01-01T01:00:00", "country_code": "CL"},
        ]
    )

    cleaned, _ = preprocess_dataframe(df)

    assert "merchant_rubro_proxy" in cleaned.columns
    assert cleaned.loc[0, "merchant_rubro_proxy"] == "UNKNOWN"
    assert cleaned.loc[1, "merchant_rubro_proxy"] == "UNKNOWN"


def test_country_code_distribution_in_summary():
    """Test that country code distribution is tracked in summary."""
    df = pd.DataFrame(
        [
            {"transaction_id": "tx-1", "amount": 10, "transaction_datetime": "2026-01-01T00:00:00", "country_code": "CL"},
            {"transaction_id": "tx-2", "amount": 11, "transaction_datetime": "2026-01-01T01:00:00", "country_code": "CL"},
            {"transaction_id": "tx-3", "amount": 12, "transaction_datetime": "2026-01-01T02:00:00", "country_code": "BO"},
        ]
    )

    cleaned, summary = preprocess_dataframe(df)

    cc_norm = summary.get("country_code_normalization", {})
    cc_dist = cc_norm.get("country_code_distribution", {})
    
    assert "CL" in cc_dist
    assert "BO" in cc_dist
    assert cc_dist["CL"] == 2
    assert cc_dist["BO"] == 1

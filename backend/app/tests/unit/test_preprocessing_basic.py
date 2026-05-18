import pandas as pd
from backend.app.ml.preprocessing import preprocess_dataframe

"""Unit test extracted from test_preprocessing.py to `unit` folder."""


def test_preprocess_basic_missing_and_scaling_extracted():
    df = pd.DataFrame([
        {
            "transaction_id": "t1",
            "amount": "100",
            "transaction_type": "purchase",
            "channel": None,
            "location": "A",
            "transaction_datetime": "2021-01-01",
            "is_fraud": 0,
        },
        {
            "transaction_id": "t2",
            "amount": None,
            "transaction_type": None,
            "channel": "web",
            "location": "B",
            "transaction_datetime": "2021-01-02",
            "is_fraud": 1,
        },
    ])

    processed, summary = preprocess_dataframe(df, apply_smote=False)

    assert summary["after_clean"] == 2
    assert "amount_scaled" in processed.columns
    assert "is_fraud" in processed.columns
    assert len(summary["columns_transformed"]) > 0

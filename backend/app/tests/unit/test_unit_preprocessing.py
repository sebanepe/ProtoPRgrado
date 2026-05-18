import pandas as pd
from backend.app.ml.preprocessing import preprocess_dataframe

"""Unit test for preprocessing logic that doesn't require DB."""

def test_preprocess_handles_missing_and_encoding():
    df = pd.DataFrame([
        {'transaction_id':'a','amount': 10, 'transaction_type': 't', 'channel': 'c', 'location':'l', 'transaction_datetime':'2021-01-01', 'is_fraud':0},
        {'transaction_id':'b','amount': None, 'transaction_type': None, 'channel': None, 'location':None, 'transaction_datetime':'2021-01-02', 'is_fraud':1},
    ])
    processed, summary = preprocess_dataframe(df, apply_smote=False)
    assert 'amount_scaled' in processed.columns
    assert 'is_fraud' in processed.columns
    assert summary['after_clean'] == 2

import pandas as pd
from backend.app.ml import proxy_labeling, preprocessing


def test_product_rules_disabled_when_unknown():
    df = pd.DataFrame({
        'merchant_hash': ['m1', 'm1', 'm2'],
        'transaction_datetime': [pd.Timestamp('2026-01-01T00:00:00')] * 3,
        'customer_hash': ['c1','c2','c3'],
        'amount': [10,20,30],
    })
    # ensure card_product_proxy absent -> becomes UNKNOWN
    out = proxy_labeling.generate_behavioral_risk_features(df)
    assert 'feature_same_merchant_20_cards_by_product_presence' in out.columns
    assert 'feature_tnp_50_approved_by_product' in out.columns
    # since all product proxies are UNKNOWN, features must be 0
    assert int(out['feature_same_merchant_20_cards_by_product_presence'].sum()) == 0
    assert int(out['feature_tnp_50_approved_by_product'].sum()) == 0


def test_is_international_aligned_with_feature():
    df = pd.DataFrame({
        'transaction_datetime': [pd.Timestamp('2026-01-01T00:00:00')],
        'amount': [10],
        'country_code': ['US']
    })
    cleaned, summary = preprocessing.preprocess_dataframe(df)
    assert 'feature_international_transaction' in cleaned.columns
    assert 'is_international' in cleaned.columns
    assert int(cleaned['is_international'].iloc[0]) == int(cleaned['feature_international_transaction'].iloc[0])


def test_transaction_id_deterministic():
    df = pd.DataFrame({
        'transaction_datetime': [pd.Timestamp('2026-01-01T00:00:00')],
        'amount': [12.34],
        'customer_hash': ['cust1'],
        'merchant_hash': ['merch1'],
        'reference_number': ['ref1'],
        'authorization_code': ['auth1']
    })
    df1 = preprocessing.generate_anonymized_keys(df.copy())
    df2 = preprocessing.generate_anonymized_keys(df.copy())
    assert df1['transaction_id'].iloc[0] == df2['transaction_id'].iloc[0]


def test_feature_set_excludes_leakage_columns():
    df = pd.DataFrame({
        'transaction_id': ['t1'],
        'customer_hash': ['c1'],
        'merchant_hash': ['m1'],
        'amount': [10],
        'is_fraud': [0],
        'response_code': ['07'],
        'behavioral_risk_score': [0.8],
        'fraud_label_reason': ['reason']
    })
    X, y = preprocessing.get_training_columns(df)
    # excluded columns should not be in X
    for col in ['response_code', 'behavioral_risk_score', 'fraud_label_reason', 'transaction_id', 'customer_hash', 'merchant_hash']:
        assert col not in X.columns
    assert 'amount' in X.columns
    assert y.iloc[0] == 0

import pandas as pd
import pytest
from backend.app.ml.proxy_labeling import normalize_response_code, generate_proxy_fraud_label, calculate_behavioral_risk_score, calculate_independent_rule_groups


def test_normalize_response_code():
    assert normalize_response_code(' 07 ') == '07'
    assert normalize_response_code(7.0) == '07'
    assert normalize_response_code(41) == '41'
    assert normalize_response_code(None) == ''


def test_response_code_high_risk_labels():
    df = pd.DataFrame({'response_code': ['59', '43', '00']})
    df = generate_proxy_fraud_label(df)
    assert df.iloc[0]['is_fraud'] == 1  # 59
    assert df.iloc[1]['is_fraud'] == 1  # 43
    assert df.iloc[2]['is_fraud'] == 0  # 00


def test_behavioral_weak_label_combination():
    # create row that should activate multiple rule groups: high amount 1h, night, international, many tx day
    rows = []
    # create 6 transactions for same customer same day including high amounts to trigger counts
    base_dt = pd.Timestamp('2026-05-20 01:30')
    for i in range(6):
        rows.append({
            'transaction_datetime': base_dt + pd.Timedelta(minutes=i),
            'amount': 4000 if i==0 else 2000,
            'customer_hash': 'cust_1',
            'country_code': 'US',
            'merchant_hash': 'm1',
            'card_brand': 'VISA',
            'card_product_proxy': 'P1',
            'pos_entry_mode': 10, # TNP
            'has_pinblock': 0,
        })
    # add several other customers for same merchant within same hour to trigger merchant-distinct-customer features
    for j in range(2, 7):
        rows.append({
            'transaction_datetime': base_dt + pd.Timedelta(minutes=6 + j),
            'amount': 50,
            'customer_hash': f'cust_{j}',
            'country_code': 'US',
            'merchant_hash': 'm1',
            'card_brand': 'VISA',
            'card_product_proxy': 'P1',
            'pos_entry_mode': 10,
            'has_pinblock': 0,
        })
    df = pd.DataFrame(rows)
    df = generate_proxy_fraud_label(df)
    # behavioral_risk_score within [0,1]
    assert df['behavioral_risk_score'].between(0,1).all()
    # independent groups should be >=3 for first row
    assert df.iloc[0]['independent_rule_groups'] >= 3
    # label should be flagged and fraud_label_reason should contain activated rules
    assert df.iloc[0]['is_fraud'] == 1
    assert isinstance(df.iloc[0]['fraud_label_reason'], str) and len(df.iloc[0]['fraud_label_reason']) > 0


def test_independent_rule_groups_counting():
    df = pd.DataFrame([
        {'feature_high_amount':1, 'feature_night_transaction':0, 'feature_international_transaction':0, 'feature_many_customer_transactions_day':0, 'feature_many_merchants_customer_day':0, 'feature_tp_pem_07':0},
        {'feature_high_amount':0, 'feature_night_transaction':1, 'feature_international_transaction':1, 'feature_many_customer_transactions_day':1, 'feature_many_merchants_customer_day':0, 'feature_tp_pem_07':0},
    ])
    counts = calculate_independent_rule_groups(df)
    assert counts.iloc[0] >= 1
    assert counts.iloc[1] >= 3

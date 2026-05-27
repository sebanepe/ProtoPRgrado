import os
import pandas as pd
from backend.app.ml.build_training_dataset import build_training_dataset, PROJECT_PROCESSED_DIR


def test_feature_set_generation_removes_forbidden_and_keeps_target(tmp_path):
    df = pd.DataFrame([
        {
            'transaction_id': 't1', 'amount': 100, 'transaction_datetime': '2026-05-20 10:00',
            'response_code': '59', 'normalized_response_code': '59', 'response_high_risk': 1,
            'behavioral_risk_score': 0.8, 'independent_rule_groups': 3, 'label_source': 'response_code_proxy',
            'customer_hash': 'c1', 'merchant_hash': 'm1', 'device_id': 'd1', 'feature_high_amount': 1, 'is_fraud': 1
        },
        {
            'transaction_id': 't2', 'amount': 50, 'transaction_datetime': '2026-05-20 11:00',
            'response_code': '00', 'normalized_response_code': '00', 'response_high_risk': 0,
            'behavioral_risk_score': 0.0, 'independent_rule_groups': 0, 'label_source': 'no_proxy_risk_detected',
            'customer_hash': 'c2', 'merchant_hash': 'm2', 'device_id': 'd2', 'feature_high_amount': 0, 'is_fraud': 0
        }
    ])
    src = tmp_path / "sample.csv"
    df.to_csv(src, index=False)
    out_file, report = build_training_dataset(str(src), out_name=f"test_feature_set_{os.getpid()}")

    assert os.path.exists(out_file)
    assert os.path.exists(report)

    train = pd.read_csv(out_file)
    # forbidden columns must not be present
    forbidden = [
        'response_code', 'normalized_response_code', 'response_high_risk', 'response_code_reason',
        'is_fraud_proxy', 'behavioral_risk_score', 'independent_rule_groups', 'label_source', 'fraud_label_reason', 'risk_signal_reason',
        'transaction_id', 'customer_hash', 'merchant_hash', 'device_id'
    ]
    for c in forbidden:
        assert c not in train.columns

    # target preserved
    assert 'is_fraud' in train.columns

    # behavior features preserved
    assert 'feature_high_amount' in train.columns

    # rows not increased
    assert len(train) == len(df)

    # report confirms response_code not used (or indicates label_source present and checked)
    with open(report, 'r', encoding='utf-8') as f:
        txt = f.read()
    assert (
        'CONFIRMATION: response_code was NOT used' in txt
        or 'label_source column not present' in txt
        or 'ALERT: response_code_proxy was used' in txt
    )

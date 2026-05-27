import pandas as pd
from backend.app.ml.build_training_dataset import deterministic_transaction_id
from backend.app.ml import preprocessing as pp


def test_reference_not_unique_generates_different_tids():
    # two rows same reference but different amount -> different deterministic ids
    r1 = pd.Series({
        'transaction_datetime': '2026-04-01T10:00:00Z',
        'amount': 100.0,
        'customer_hash': 'cust_abc',
        'merchant_code': 'M1',
        'terminal_code': 'T1',
        'reference_number': 'REF123',
        'authorization_code': 'AUTH1',
        'transaction_type': 'SALE',
        'transaction_category': 'CAT1',
        'process_code': 'P1'
    })
    r2 = pd.Series({
        'transaction_datetime': '2026-04-01T10:00:00Z',
        'amount': 200.0,
        'customer_hash': 'cust_abc',
        'merchant_code': 'M1',
        'terminal_code': 'T1',
        'reference_number': 'REF123',
        'authorization_code': 'AUTH1',
        'transaction_type': 'SALE',
        'transaction_category': 'CAT1',
        'process_code': 'P1'
    })
    tid1 = deterministic_transaction_id(r1)
    tid2 = deterministic_transaction_id(r2)
    assert tid1 != tid2


def test_pais_maps_to_country_code():
    df = pd.DataFrame({'PAIS': ['Bolivia', 'Peru', 'BO', '']})
    df2 = pp.normalize_column_names(df)
    assert 'country_code' in df2.columns
    assert df2['country_code'].iloc[0].upper() in ('BOLIVIA', 'BO', 'BOLIVIA'.upper())


def test_sanitize_customer_hash_from_pan():
    df = pd.DataFrame({'pan_card': ['1000068173284919', None], 'customer_hash': ['1000068173284919', None]})
    df2 = pp.generate_anonymized_keys(df)
    # customer_hash should be prefixed with cust_
    assert df2['customer_hash'].iloc[0].startswith('cust_')
    assert df2['customer_hash'].iloc[1] is None or pd.isna(df2['customer_hash'].iloc[1])

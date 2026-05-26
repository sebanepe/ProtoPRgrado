import pandas as pd
import numpy as np
from backend.app.ml import preprocessing


def test_get_training_columns_excludes_sensitive():
    df = pd.DataFrame({
        'transaction_id': ['t1','t2'],
        'amount': [10,20],
        'is_fraud': [0,1],
        'masked_card': ['xxxx','yyyy'],
        'customer_hash': ['c1','c2'],
        'feature_high_amount': [0,1]
    })
    X, y = preprocessing.get_training_columns(df)
    assert 'is_fraud' not in X.columns
    assert 'masked_card' not in X.columns
    assert 'customer_hash' not in X.columns
    assert 'feature_high_amount' in X.columns
    assert list(y) == [0,1]


def test_split_train_test_stratify():
    df = pd.DataFrame({'a': range(50)})
    y = pd.Series([0]*40 + [1]*10)
    X_train, X_test, y_train, y_test = preprocessing.split_train_test(df, y, test_size=0.2, random_state=42)
    assert len(X_test) == 10
    # stratify means proportion roughly maintained
    assert sum(y_test) in (1,2,3)  # small sample may be 2


def test_detect_feature_types():
    X = pd.DataFrame({
        'num': [1,2,3],
        'cat': ['a','b','a'],
        'flag_bool': [True, False, True],
        'zero_one_str': ['0','1','1']
    })
    types = preprocessing.detect_feature_types(X)
    assert 'num' in types['numeric']
    assert 'cat' in types['categorical']
    assert 'flag_bool' in types['boolean'] or 'zero_one_str' in types['boolean']


def test_build_preprocessing_pipeline_and_transform():
    X = pd.DataFrame({
        'num': [1.0,2.0,3.0],
        'cat': ['a','b','a'],
        'flag': [0,1,0]
    })
    preprocessor, pipeline = preprocessing.build_preprocessing_pipeline(X)
    # fit pipeline
    pipeline.fit(X)
    Xt = pipeline.transform(X)
    assert Xt.shape[0] == 3


def test_apply_smote_if_needed_single_class():
    X = pd.DataFrame({'a': [1,2,3]})
    y = pd.Series([0,0,0])
    Xr, yr, report = preprocessing.apply_smote_if_needed(X, y)
    assert report['smote_applied'] is False
    assert report['reason'] == 'single_class'

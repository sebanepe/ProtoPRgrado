import pandas as pd
import numpy as np
import os
import json
from backend.app.ml import preprocessing
from backend.app.services import preprocessing_service
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.app.database import Base
from backend.app.models import models


def test_apply_smote_if_needed_applies_and_reports():
    # create imbalanced dataset with small minority but enough for SMOTE
    X = pd.DataFrame({'f1': list(range(10)), 'f2': list(range(10,20))})
    y = pd.Series([0]*8 + [1]*2)
    Xr, yr, report = preprocessing.apply_smote_if_needed(X, y)
    assert report['smote_applied'] is True
    assert 'before_distribution' in report
    assert 'after_distribution' in report
    assert sum(report['after_distribution'].values()) == len(yr)


def test_smote_not_applied_too_few_minority():
    X = pd.DataFrame({'f1': [1,2,3,4], 'f2':[5,6,7,8]})
    y = pd.Series([0,0,0,1])
    # minority_count =1 -> too_few_minority
    Xr, yr, report = preprocessing.apply_smote_if_needed(X, y)
    assert report['smote_applied'] is False
    assert report['reason'] == 'too_few_minority'


def test_pipeline_persisted_and_feature_set_saved(tmp_path):
    # create a small csv to act as training dataset
    df = pd.DataFrame({'amount':[1,2,3,4,5,6,7,8],'is_fraud':[0,0,0,0,0,1,1,1],'feature_a':[1,2,3,4,5,6,7,8]})
    csv_path = tmp_path / "training.csv"
    df.to_csv(csv_path, index=False)

    # prepare a sqlite in-memory DB and create tables
    engine = create_engine('sqlite:///:memory:')
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    db = Session()

    # insert a dummy PreprocessingRun so run_id exists
    pr = models.PreprocessingRun(status='COMPLETED')
    db.add(pr)
    db.commit()
    db.refresh(pr)

    # copy csv into project processed dir (use tmp dir)
    from backend.app.services import preprocessing_service as svc
    old_dir = svc.PROJECT_PROCESSED_DIR
    svc.PROJECT_PROCESSED_DIR = str(tmp_path)

    try:
        report = svc.run_preprocessing_for_training(db, run_id=pr.id, training_dataset_path=str(csv_path), apply_smote=True)
        assert 'feature_set_id' in report
        assert 'smote_report' in report
        assert report['pipeline_path'] is not None
        # check file exists
        assert os.path.exists(report['pipeline_path'])
        # load FeatureSet and check dedicated columns contain pipeline_path and smote_report
        from backend.app.models.models import FeatureSet
        fs = db.query(FeatureSet).filter(FeatureSet.id == report['feature_set_id']).first()
        assert fs.pipeline_path is not None
        assert fs.smote_report_json is not None
    finally:
        svc.PROJECT_PROCESSED_DIR = old_dir


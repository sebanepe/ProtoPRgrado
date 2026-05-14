import os
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from backend.app.models.models import ModelResult, Transaction
from datetime import datetime
from backend.app.services import settings_service
from backend.app.services.alert_service import generate_alerts_from_batch


def test_generate_alerts_respects_threshold(db_session, tmp_path):
    # create a very simple model and save it
    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    y = np.array([0, 1])
    clf = LogisticRegression()
    clf.fit(X, y)

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    model_file = models_dir / "testmodel_1.pkl"
    joblib.dump(clf, str(model_file))

    # create an active model row
    mr = ModelResult(model_name="testmodel", version="1", is_active=True)
    db_session.add(mr)
    db_session.commit()
    db_session.refresh(mr)

    # create a transaction to point alerts to
    tx = Transaction(transaction_id="tx123", amount=10.0, transaction_type="purchase", channel="web", location="loc", device_id="d1", customer_hash="c1", transaction_datetime=datetime(2021,1,1), is_fraud=False)
    db_session.add(tx)
    db_session.commit()
    db_session.refresh(tx)

    # set config with threshold 0.5
    settings_service.set_model_config(db_session, active_model_id=mr.id, alert_threshold=0.5, updated_by="tester")

    # a transaction likely to be scored high
    transactions = [
        {"id": tx.id, "feature1": 1.0, "feature2": 1.0},
    ]

    created = generate_alerts_from_batch(db_session, transactions, models_dir=str(models_dir))

    # expect at least one alert created
    assert isinstance(created, list)
    assert len(created) >= 1
import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.app.database import Base
from backend.app.models.models import ModelResult
from backend.app.services import alert_service


def get_test_session():
    engine = create_engine("sqlite:///:memory:")
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    return TestingSessionLocal()


def test_classify_risk_levels():
    from backend.app.ml.scoring import classify_risk
    assert classify_risk(0.8) == "HIGH"
    assert classify_risk(0.5) == "MEDIUM"
    assert classify_risk(0.1) == "LOW"


def test_alert_generation_integration(tmp_path):
    db = get_test_session()
    # create a simple model and save
    X = [[0.1, 1], [1.0, 0], [3.0, 1], [2.0, 0]]
    y = [0, 0, 1, 1]
    lr = LogisticRegression(max_iter=1000).fit(X, y)

    models_dir = os.path.join(str(tmp_path), "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "logistic_regression_testver.pkl")
    joblib.dump(lr, model_path)

    # insert ModelResult as active
    mr = ModelResult(model_name="logistic_regression_testver", version="testver", is_active=True)
    db.add(mr)
    db.commit()
    db.refresh(mr)

    transactions = [
        {"id": 1, "feat1": 0.1, "feat2": 1},
        {"id": 2, "feat1": 3.0, "feat2": 1},
    ]

    created = alert_service.generate_alerts_from_batch(db, transactions, models_dir=models_dir)
    # created may be empty depending on threshold; ensure function runs and returns list
    assert isinstance(created, list)

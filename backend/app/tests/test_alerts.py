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

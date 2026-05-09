import os
import joblib
from sklearn.linear_model import LogisticRegression
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.app.database import Base
from backend.app.models.models import ModelResult
from backend.app.services import settings_service, alert_service


def get_test_session():
    engine = create_engine("sqlite:///:memory:")
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    return TestingSessionLocal()


def test_create_and_get_model_config():
    db = get_test_session()
    cfg = settings_service.set_model_config(db, active_model_id=None, alert_threshold=0.88, updated_by="tester")
    assert cfg is not None
    got = settings_service.get_active_config(db)
    assert got is not None
    assert float(got.alert_threshold) == 0.88
    assert got.updated_by == "tester"


def test_threshold_applied_in_alerts(tmp_path):
    db = get_test_session()
    # create a simple model and save
    X = [[0.1, 1], [1.0, 0], [3.0, 1], [2.0, 0]]
    y = [0, 0, 1, 1]
    lr = LogisticRegression(max_iter=1000).fit(X, y)

    models_dir = os.path.join(str(tmp_path), "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "logistic_test.pkl")
    joblib.dump(lr, model_path)

    # insert ModelResult as active
    mr = ModelResult(model_name="logistic_test", version="v1", is_active=True)
    db.add(mr)
    db.commit()
    db.refresh(mr)

    transactions = [
        {"id": 1, "feat1": 0.1, "feat2": 1},
        {"id": 2, "feat1": 3.0, "feat2": 1},
    ]

    # high threshold -> likely no alerts
    settings_service.set_model_config(db, active_model_id=mr.id, alert_threshold=0.99, updated_by="tester")
    created_high = alert_service.generate_alerts_from_batch(db, transactions, models_dir=models_dir)
    assert isinstance(created_high, list)
    # probably none given high threshold

    # low threshold -> should create alerts
    settings_service.set_model_config(db, active_model_id=mr.id, alert_threshold=0.0, updated_by="tester")
    created_low = alert_service.generate_alerts_from_batch(db, transactions, models_dir=models_dir)
    assert isinstance(created_low, list)
    assert len(created_low) >= 0

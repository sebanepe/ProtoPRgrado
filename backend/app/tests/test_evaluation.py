import os
import pandas as pd
import tempfile
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.app.database import Base
from backend.app.services import evaluation_service


def get_test_session():
    engine = create_engine("sqlite:///:memory:")
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    return TestingSessionLocal()


def make_processed_csv(tmpdir):
    df = pd.DataFrame(
        {
            "feat1": [0.1, 0.2, 3.0, 2.5, 0.0, 1.1, -0.5, 0.4],
            "feat2": [1, 0, 1, 0, 1, 0, 1, 0],
            "is_fraud": [0, 0, 1, 1, 0, 0, 0, 1],
        }
    )
    path = os.path.join(tmpdir, "processed.csv")
    df.to_csv(path, index=False)
    return path, df


def test_evaluation_and_export(tmp_path):
    db = get_test_session()
    processed_path, df = make_processed_csv(str(tmp_path))
    models_dir = os.path.join(str(tmp_path), "models")
    os.makedirs(models_dir, exist_ok=True)

    # train two simple models and save
    X = df[["feat1", "feat2"]]
    y = df["is_fraud"]
    lr = LogisticRegression(max_iter=1000).fit(X, y)
    rf = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)
    lr_path = os.path.join(models_dir, "logistic_regression_test.pkl")
    rf_path = os.path.join(models_dir, "random_forest_test.pkl")
    joblib.dump(lr, lr_path)
    joblib.dump(rf, rf_path)

    results = evaluation_service.compare_models(db, input_path=processed_path, models_dir=models_dir, export_path=os.path.join(str(tmp_path), "comp.csv"))
    assert isinstance(results, list)
    assert len(results) >= 1
    # check export file
    assert os.path.exists(os.path.join(str(tmp_path), "comp.csv"))

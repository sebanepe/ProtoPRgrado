import os
import pandas as pd
import tempfile
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.app.database import Base
from backend.app.services import model_service


def get_test_session():
    engine = create_engine("sqlite:///:memory:")
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    return TestingSessionLocal()


def make_processed_csv(tmpdir):
    # create a small processed dataframe with numeric features and label
    df = pd.DataFrame(
        {
            "amount_scaled": [0.1, 0.2, 3.0, 2.5, 0.0, 1.1, -0.5, 0.4],
            "transaction_type_payment": [1, 0, 1, 0, 1, 0, 1, 0],
            "channel_web": [1, 1, 0, 0, 1, 1, 0, 0],
            "is_fraud": [0, 0, 1, 1, 0, 0, 0, 1],
        }
    )
    path = os.path.join(tmpdir, "processed.csv")
    df.to_csv(path, index=False)
    return path


def test_training_creates_models_and_metrics(tmp_path):
    db = get_test_session()
    processed_path = make_processed_csv(str(tmp_path))
    save_dir = os.path.join(str(tmp_path), "models")
    results = model_service.train_and_record(db, input_path=processed_path, save_dir=save_dir)
    # Expect three models
    assert len(results) == 3
    # files created
    files = os.listdir(save_dir)
    assert any(f.endswith(".pkl") for f in files)
    # metrics present
    for r in results:
        assert "metrics" in r
        assert "precision" in r["metrics"]

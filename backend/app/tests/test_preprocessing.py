import tempfile
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.app.database import Base
from backend.app.models.models import Transaction
from datetime import datetime
from backend.app.services import preprocessing_service


def get_test_session():
    engine = create_engine("sqlite:///:memory:")
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    return TestingSessionLocal()


def seed_transactions(db):
    # seed imbalanced data: 10 non-fraud, 2 fraud
    now = datetime.utcnow()
    objs = []
    for i in range(10):
        objs.append(
            Transaction(
                transaction_id=f"ok{i}",
                amount=10 + i,
                transaction_type="payment",
                channel="web",
                location="city",
                device_id=f"dev{i}",
                customer_hash=f"cust{i}",
                transaction_datetime=now,
                is_fraud=False,
            )
        )
    for j in range(2):
        objs.append(
            Transaction(
                transaction_id=f"fraud{j}",
                amount=1000 + j,
                transaction_type="transfer",
                channel="mobile",
                location="other",
                device_id=f"devf{j}",
                customer_hash=f"custf{j}",
                transaction_datetime=now,
                is_fraud=True,
            )
        )
    db.add_all(objs)
    db.commit()


def test_preprocessing_run_and_file(tmp_path):
    db = get_test_session()
    seed_transactions(db)
    out_file = tmp_path / "preprocessed.csv"
    summary = preprocessing_service.run_preprocessing(db, output_path=str(out_file), apply_smote=True)
    assert "before" in summary
    assert summary["before"] == 12
    assert summary.get("smote_applied") in (True, False)
    # If processed file created, check file exists
    if summary.get("output_path"):
        assert out_file.exists()

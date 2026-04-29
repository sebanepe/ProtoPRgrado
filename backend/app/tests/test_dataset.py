import io
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.app.database import Base
from backend.app.services import dataset_service


def get_test_session():
    engine = create_engine("sqlite:///:memory:")
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    return TestingSessionLocal()


def make_csv(rows: list, columns: list):
    df = pd.DataFrame(rows, columns=columns)
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf


def test_import_success():
    db = get_test_session()
    columns = [
        "transaction_id",
        "amount",
        "transaction_type",
        "channel",
        "location",
        "device_id",
        "customer_hash",
        "transaction_datetime",
        "is_fraud",
    ]
    rows = [
        ["tx1", 100.0, "payment", "web", "city", "dev1", "cust1", "2023-01-01T10:00:00", False],
        ["tx2", 50.5, "withdrawal", "atm", "city", "dev2", "cust2", "2023-01-02T11:00:00", False],
    ]
    buf = make_csv(rows, columns)
    res = dataset_service.import_dataset(db, buf, name="test", file_name="test.csv")
    assert res["total"] == 2
    assert res["valid"] == 2
    assert res["invalid"] == 0


def test_import_missing_columns():
    db = get_test_session()
    columns = ["transaction_id", "amount"]
    rows = [["tx1", 100.0]]
    buf = make_csv(rows, columns)
    try:
        dataset_service.import_dataset(db, buf, name="test", file_name="bad.csv")
        assert False, "Expected ValueError for missing columns"
    except ValueError as e:
        assert "Missing required columns" in str(e)


def test_import_empty_file():
    db = get_test_session()
    buf = io.BytesIO(b"")
    try:
        dataset_service.import_dataset(db, buf, name="empty", file_name="empty.csv")
        assert False, "Expected ValueError for empty file"
    except ValueError as e:
        assert "Uploaded file is empty" in str(e)

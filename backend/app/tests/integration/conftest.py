import io
import pytest
import pandas as pd
from datetime import datetime

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from sqlalchemy.orm import sessionmaker

from backend.app.database import Base, get_db
from backend.app.main import app


TEST_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(scope="session", autouse=True)
def prepare_database():
    # ensure all models are imported so Base metadata is populated
    import backend.app.models.models  # noqa: F401
    # create all tables in test DB
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


def _override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture()
def test_client():
    # override dependency
    app.dependency_overrides[get_db] = _override_get_db
    # create a default admin user in test DB and set header for requests
    db = TestingSessionLocal()
    try:
        from backend.app.services.auth_service import hash_password
        from backend.app.repositories import user_repository

        admin_email = "sebanpb@gmail.com"
        if not user_repository.get_user_by_email(db, admin_email):
            admin_pw = hash_password("Mariokart8$")
            user_repository.create_user(db, full_name="Test Admin", email=admin_email, password_hash=admin_pw, role="ADMIN")
    finally:
        db.close()

    with TestClient(app) as c:
        c.headers.update({"X-User-Email": "sebanpb@gmail.com"})
        yield c
    app.dependency_overrides.clear()


@pytest.fixture()
def db_session():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture()
def sample_user():
    return {
        "full_name": "Test User",
        "email": "test.user@example.com",
        "password": "s3cret",
        "role": "FRAUD_ANALYST",
    }


@pytest.fixture()
def sample_transactions():
    import uuid

    now = datetime.utcnow().isoformat()
    return [
        {
            "transaction_id": f"t1-{uuid.uuid4().hex[:8]}",
            "amount": "10.50",
            "transaction_type": "purchase",
            "channel": "web",
            "location": "loc1",
            "device_id": "dev1",
            "customer_hash": "c1",
            "transaction_datetime": now,
            "is_fraud": False,
        },
        {
            "transaction_id": f"t2-{uuid.uuid4().hex[:8]}",
            "amount": "99.99",
            "transaction_type": "transfer",
            "channel": "mobile",
            "location": "loc2",
            "device_id": "dev2",
            "customer_hash": "c2",
            "transaction_datetime": now,
            "is_fraud": True,
        },
    ]


@pytest.fixture()
def sample_dataset_dataframe():
    now = datetime.utcnow()
    df = pd.DataFrame(
        [
            {
                "transaction_id": "t1",
                "amount": 10.5,
                "transaction_type": "purchase",
                "channel": "web",
                "location": "loc1",
                "device_id": "dev1",
                "customer_hash": "c1",
                "transaction_datetime": now,
                "is_fraud": False,
            },
            {
                "transaction_id": "t2",
                "amount": 5.0,
                "transaction_type": "transfer",
                "channel": "mobile",
                "location": "loc2",
                "device_id": "dev2",
                "customer_hash": "c2",
                "transaction_datetime": now,
                "is_fraud": True,
            },
        ]
    )
    return df

import pytest
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
    # ensure models are imported so Base metadata is populated
    import backend.app.models.models  # noqa: F401
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
def client():
    app.dependency_overrides[get_db] = _override_get_db
    # create default admin in test DB and set header
    db = TestingSessionLocal()
    try:
        from backend.app.services.auth_service import hash_password
        from backend.app.repositories import user_repository

        admin_email = "reg.admin@example.com"
        if not user_repository.get_user_by_email(db, admin_email):
            admin_pw = hash_password("Admin123!")
            user_repository.create_user(db, full_name="Regression Admin", email=admin_email, password_hash=admin_pw, role="ADMIN")
    finally:
        db.close()

    with TestClient(app) as c:
        c.headers.update({"X-User-Email": "reg.admin@example.com"})
        yield c
    app.dependency_overrides.clear()


@pytest.fixture()
def sample_user():
    return {
        "full_name": "Regression User",
        "email": "reg.user@example.com",
        "password": "s3cret",
        "role": "FRAUD_ANALYST",
    }


@pytest.fixture()
def sample_transactions():
    from datetime import datetime
    import uuid

    now = datetime.utcnow().isoformat()
    return [
        {
            "transaction_id": f"rt1-{uuid.uuid4().hex[:8]}",
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
            "transaction_id": f"rt2-{uuid.uuid4().hex[:8]}",
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

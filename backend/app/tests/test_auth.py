from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.app.database import Base
from backend.app.repositories import user_repository
from backend.app.services import auth_service
from backend.app.schemas.auth import UserCreate


def get_test_session():
    engine = create_engine("sqlite:///:memory:")
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    return TestingSessionLocal()


def test_password_hash_and_verify():
    plain = "SuperSecret123!"
    hashed = auth_service.hash_password(plain)
    assert auth_service.verify_password(plain, hashed)


def test_register_and_login():
    db = get_test_session()
    user_in = UserCreate(full_name="Test User", email="test@example.com", password="secret", role="DATA_SCIENTIST")
    user = auth_service.register_user(db, user_in)
    assert user.email == "test@example.com"

    authenticated = auth_service.authenticate_user(db, "test@example.com", "secret")
    assert authenticated is not None
    assert authenticated.email == "test@example.com"

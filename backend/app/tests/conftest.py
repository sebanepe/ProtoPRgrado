import pytest
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app.database import Base


@pytest.fixture()
def db_engine(tmp_path):
    # Use an in-memory SQLite database for tests
    db_url = "sqlite:///:memory:"
    engine = create_engine(db_url)
    Base.metadata.create_all(bind=engine)
    yield engine
    engine.dispose()


@pytest.fixture()
def db_session(db_engine):
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)
    sess = SessionLocal()
    try:
        yield sess
    finally:
        sess.close()

import pytest
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app.database import Base


@pytest.fixture(scope="session", autouse=True)
def _isolate_data_dirs(tmp_path_factory):
    """Redirect all test file I/O away from real data/processed and data/uploads directories."""
    tmp = tmp_path_factory.mktemp("test_data")
    processed = tmp / "processed"
    uploads = tmp / "uploads"
    processed.mkdir()
    uploads.mkdir()

    import backend.app.services.preprocessing_service as ps
    import backend.app.ml.build_training_dataset as bdt

    orig_ps = ps.PROJECT_PROCESSED_DIR
    orig_bdt = bdt.PROJECT_PROCESSED_DIR
    orig_ds = os.environ.get("DATASET_STORAGE")

    ps.PROJECT_PROCESSED_DIR = str(processed)
    bdt.PROJECT_PROCESSED_DIR = str(processed)
    os.environ["DATASET_STORAGE"] = str(uploads)

    yield

    ps.PROJECT_PROCESSED_DIR = orig_ps
    bdt.PROJECT_PROCESSED_DIR = orig_bdt
    if orig_ds is None:
        os.environ.pop("DATASET_STORAGE", None)
    else:
        os.environ["DATASET_STORAGE"] = orig_ds


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

import os
import json
import pandas as pd
import tempfile
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from sqlalchemy.orm import sessionmaker
from backend.app.main import app
from backend.app.database import Base
from backend.app.database import get_db
from backend.app.models.models import User, FeatureSet, SystemLog


def override_get_db(engine):
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    def _get_db():
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()
    return _get_db


def test_feature_set_endpoints_with_auth(tmp_path):
    # setup in-memory sqlite and override dependency
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(bind=engine)
    app.dependency_overrides[get_db] = override_get_db(engine)

    # create test DB session
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()

    # create a test user
    user = User(full_name='Test User', email='tester@example.com', password_hash='x', role='ADMIN', is_active=True)
    db.add(user)
    db.commit()
    db.refresh(user)

    # create a CSV file
    df = pd.DataFrame([{"a":1, "b":2}, {"a":3, "b":4}])
    fp = tmp_path / 'fs.csv'
    df.to_csv(fp, index=False)

    # create FeatureSet
    fs = FeatureSet(dataset_id=None, preprocessing_run_id=None, name='fs_integ', file_path=str(fp), row_count=2, feature_columns_json=json.dumps({"features":["a","b"]}))
    db.add(fs)
    db.commit()
    db.refresh(fs)

    client = TestClient(app)
    headers = {"X-User-Email": user.email}

    # preview
    r = client.get(f"/feature_sets/{fs.id}/preview", headers=headers)
    assert r.status_code == 200
    body = r.json()
    assert body['id'] == fs.id
    assert len(body['rows']) == 2

    # download
    r2 = client.get(f"/feature_sets/{fs.id}/download", headers=headers)
    assert r2.status_code == 200
    assert 'text/csv' in r2.headers.get('content-type', '')

    # delete
    r3 = client.delete(f"/feature_sets/{fs.id}", headers=headers)
    assert r3.status_code == 200
    assert r3.json().get('status') == 'ok'

    # verify deleted and log created with user_id
    fs_check = db.query(FeatureSet).filter(FeatureSet.id == fs.id).first()
    assert fs_check is None
    log = db.query(SystemLog).filter(SystemLog.action == 'delete_feature_set').first()
    assert log is not None
    assert log.user_id == user.id

    db.close()
 
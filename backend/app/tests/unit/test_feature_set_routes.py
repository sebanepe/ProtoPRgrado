import os
import json
import pandas as pd
from fastapi.responses import JSONResponse, FileResponse
from backend.app.routes import feature_set_routes as routes
from backend.app.models.models import FeatureSet, SystemLog


def test_feature_set_preview_and_download_and_delete(db_session, tmp_path):
    # create a small CSV file
    df = pd.DataFrame([{"a": 1, "b": 2}, {"a": 3, "b": 4}])
    file_path = tmp_path / "fs_test.csv"
    df.to_csv(file_path, index=False)

    # create FeatureSet row
    fs = FeatureSet(dataset_id=None, preprocessing_run_id=None, name="fs_test", file_path=str(file_path), row_count=2, feature_columns_json=json.dumps({"features":["a","b"]}))
    db_session.add(fs)
    db_session.commit()
    db_session.refresh(fs)

    # preview
    resp = routes.preview_feature_set(fs.id, rows=2, db=db_session, _auth=True)
    assert isinstance(resp, JSONResponse)
    body = json.loads(resp.body)
    assert body["id"] == fs.id
    assert len(body["rows"]) == 2

    # download
    resp2 = routes.download_feature_set(fs.id, db=db_session, _auth=True)
    assert isinstance(resp2, FileResponse)
    assert resp2.media_type == 'text/csv'

    # delete
    resp3 = routes.delete_feature_set(fs.id, db=db_session, _auth=True)
    assert isinstance(resp3, JSONResponse)
    content = json.loads(resp3.body)
    assert content.get("status") == "ok"

    # ensure feature set removed
    fs_check = db_session.query(FeatureSet).filter(FeatureSet.id == fs.id).first()
    assert fs_check is None

    # ensure a SystemLog entry was created (audit)
    log = db_session.query(SystemLog).filter(SystemLog.action == 'delete_feature_set').first()
    assert log is not None
    
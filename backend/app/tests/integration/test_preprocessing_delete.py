import os
from datetime import datetime, timezone

from backend.app.models.models import PreprocessingRun


def test_delete_preprocessing_run_endpoint_and_service(test_client, db_session, tmp_path):
    # create a temporary output file to simulate processed CSV
    out_file = tmp_path / "cleaned_run_test.csv"
    out_file.write_text("col1,col2\n1,2\n")

    # insert a PreprocessingRun row that references the file
    run = PreprocessingRun(
        status="COMPLETED",
        started_at=datetime.now(timezone.utc),
        finished_at=datetime.now(timezone.utc),
        output_file_path=str(out_file),
        total_records=2,
        processed_records=2,
    )
    db_session.add(run)
    db_session.commit()
    db_session.refresh(run)

    assert os.path.exists(str(out_file))

    # call DELETE endpoint
    resp = test_client.delete(f"/preprocessing/runs/{run.id}")
    assert resp.status_code == 200
    assert resp.json().get("status") == "ok"

    # run should be removed from DB
    rem = db_session.query(PreprocessingRun).filter(PreprocessingRun.id == run.id).first()
    assert rem is None

    # file should be removed
    assert not os.path.exists(str(out_file))


def test_delete_nonexistent_run_returns_404(test_client):
    resp = test_client.delete("/preprocessing/runs/999999")
    assert resp.status_code == 404

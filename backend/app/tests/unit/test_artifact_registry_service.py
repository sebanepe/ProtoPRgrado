from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from backend.app.services import artifact_registry_service, model_registry_service, rule_run_service


def _write_run_artifacts(base: Path, models: Path, token: str = "26") -> None:
    base.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"transaction_id": "tx1"}, {"transaction_id": "tx2"}]).to_csv(base / f"preprocessed_run_{token}.csv", index=False)
    (base / f"preprocessing_report_run_{token}.md").write_text("# preprocessing\n", encoding="utf-8")
    pd.DataFrame([{"alert_id": "a1"}, {"alert_id": "a2"}, {"alert_id": "a3"}]).to_csv(base / f"alerts_run_{token}.csv", index=False)
    pd.DataFrame([{"summary_alert_id": "s1"}, {"summary_alert_id": "s2"}]).to_csv(base / f"alerts_summary_run_{token}.csv", index=False)
    (base / f"rules_report_run_{token}.md").write_text("# rules\n", encoding="utf-8")
    pd.DataFrame([{"transaction_id": "tx1", "anomaly_flag": 0}]).to_csv(base / f"anomaly_scores_run_{token}.csv", index=False)
    pd.DataFrame([{"transaction_id": "tx1", "feature": 1}]).to_csv(base / f"unsupervised_feature_set_run_{token}.csv", index=False)
    (base / f"anomaly_report_run_{token}.md").write_text("# anomaly\n", encoding="utf-8")
    (models / f"isolation_forest_run_{token}.pkl").write_bytes(b"not-a-real-model")
    (models / f"isolation_forest_run_{token}_metadata.json").write_text(
        json.dumps({"algorithm": "isolation_forest", "source_run": f"preprocessed_run_{token}"}),
        encoding="utf-8",
    )


def test_register_artifact_creates_checksum_and_row_count(db_session, tmp_path):
    csv_path = tmp_path / "sample.csv"
    pd.DataFrame([{"a": 1}, {"a": 2}]).to_csv(csv_path, index=False)

    artifact = artifact_registry_service.register_artifact(
        db_session,
        artifact_type="PREPROCESSED_CSV",
        phase="PHASE_A",
        source_run="preprocessed_run_26",
        file_path=csv_path,
    )

    assert artifact.file_path == str(csv_path)
    assert artifact.file_name == "sample.csv"
    assert artifact.row_count == 2
    assert artifact.checksum
    assert artifact.file_size_bytes == csv_path.stat().st_size
    assert artifact.status == "AVAILABLE"


def test_register_or_update_artifact_does_not_duplicate(db_session, tmp_path):
    csv_path = tmp_path / "sample.csv"
    pd.DataFrame([{"a": 1}]).to_csv(csv_path, index=False)

    first = artifact_registry_service.register_or_update_artifact(
        db_session,
        artifact_type="PREPROCESSED_CSV",
        phase="PHASE_A",
        source_run="preprocessed_run_26",
        file_path=csv_path,
    )
    second = artifact_registry_service.register_or_update_artifact(
        db_session,
        artifact_type="PREPROCESSED_CSV",
        phase="PHASE_A",
        source_run="preprocessed_run_26",
        file_path=csv_path,
    )

    assert first.id == second.id
    assert len(artifact_registry_service.list_artifacts(db_session, source_run="preprocessed_run_26")) == 1


def test_scan_existing_artifacts_detects_expected_run_files(db_session, tmp_path):
    processed = tmp_path / "processed"
    models = tmp_path / "models"
    _write_run_artifacts(processed, models)

    result = artifact_registry_service.scan_existing_artifacts(
        db_session,
        "preprocessed_run_26",
        processed_dir=processed,
        models_dir=models,
    )
    artifact_types = {item["artifact_type"] for item in result["registered"]}

    assert "PREPROCESSED_CSV" in artifact_types
    assert "RULE_ALERTS_CSV" in artifact_types
    assert "RULE_SUMMARY_CSV" in artifact_types
    assert "ANOMALY_SCORES_CSV" in artifact_types
    assert "MODEL_PICKLE" in artifact_types
    assert result["registered_count"] >= 10


def test_register_rule_run_from_artifacts_creates_counts_without_regenerating(db_session, tmp_path):
    processed = tmp_path / "processed"
    models = tmp_path / "models"
    _write_run_artifacts(processed, models)
    alerts_path = processed / "alerts_run_26.csv"
    summary_path = processed / "alerts_summary_run_26.csv"
    before = (alerts_path.stat().st_mtime_ns, summary_path.stat().st_mtime_ns)
    artifact_registry_service.scan_existing_artifacts(db_session, "preprocessed_run_26", processed_dir=processed, models_dir=models)

    rule_run = rule_run_service.register_rule_run_from_artifacts(db_session, "preprocessed_run_26")

    assert rule_run.detailed_alert_count == 3
    assert rule_run.grouped_alert_count == 2
    assert rule_run.status == "AVAILABLE"
    assert (alerts_path.stat().st_mtime_ns, summary_path.stat().st_mtime_ns) == before


def test_register_unsupervised_model_from_artifacts_creates_registry_without_retraining(db_session, tmp_path):
    processed = tmp_path / "processed"
    models = tmp_path / "models"
    _write_run_artifacts(processed, models)
    model_path = models / "isolation_forest_run_26.pkl"
    before = model_path.stat().st_mtime_ns
    artifact_registry_service.scan_existing_artifacts(db_session, "preprocessed_run_26", processed_dir=processed, models_dir=models)

    model = model_registry_service.register_unsupervised_model_from_artifacts(db_session, "preprocessed_run_26")

    assert model.model_family == "UNSUPERVISED"
    assert model.algorithm == "isolation_forest"
    assert model.is_active is True
    assert model.status == "AVAILABLE"
    assert model_path.stat().st_mtime_ns == before

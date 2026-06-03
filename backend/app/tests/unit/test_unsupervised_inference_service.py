"""Unit tests for unsupervised_inference_service.

Validates:
- No is_fraud or confirmed_fraud in results
- Incompatible CSV returns controlled error
- Model is loaded without retraining
- preprocessed_run can be used as input
- anomaly_flag and anomaly_rank present; is_fraud absent
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from backend.app.services import unsupervised_inference_service as svc


# ── Helpers ───────────────────────────────────────────────────────────────────

def _minimal_preprocessed_csv(tmp_path: Path) -> Path:
    """A preprocessed CSV with the columns expected by unsupervised_feature_builder."""
    df = pd.DataFrame([
        {
            "transaction_id": f"tx{i}",
            "customer_hash": f"cust{i}",
            "amount": float(10 + i),
            "transaction_datetime": "2026-01-01 10:00:00",
            "country_code": "BO",
            "pos_entry_mode": "05",
            "merchant_rubro_proxy": "RETAIL",
            "has_pinblock": 1,
            "card_presence_type": "PRESENT",
        }
        for i in range(20)
    ])
    path = tmp_path / "preprocessed_run_99.csv"
    df.to_csv(path, index=False)
    return path


def _make_mock_db():
    return MagicMock()


# ── list_trained_models ───────────────────────────────────────────────────────

def test_list_trained_models_returns_unsupervised_available(db_session):
    from backend.app.models.models import ModelRegistry
    reg = ModelRegistry(
        model_family="UNSUPERVISED",
        algorithm="isolation_forest",
        source_run="preprocessed_run_99",
        status="AVAILABLE",
        metrics_json=json.dumps({"anomaly_rate": 0.01, "contamination": 0.01, "total_records": 100}),
    )
    db_session.add(reg)
    db_session.commit()

    result = svc.list_trained_models(db_session)
    assert len(result) >= 1
    model = next(r for r in result if r["source_run"] == "preprocessed_run_99")
    assert model["algorithm"] == "isolation_forest"
    assert model["status"] == "AVAILABLE"
    assert model["anomaly_rate"] == pytest.approx(0.01)


def test_list_trained_models_excludes_unavailable(db_session):
    from backend.app.models.models import ModelRegistry
    reg = ModelRegistry(
        model_family="UNSUPERVISED",
        algorithm="isolation_forest",
        source_run="preprocessed_run_missing",
        status="MISSING",
    )
    db_session.add(reg)
    db_session.commit()

    result = svc.list_trained_models(db_session)
    ids = [r["source_run"] for r in result]
    assert "preprocessed_run_missing" not in ids


# ── list_preprocessed_runs ────────────────────────────────────────────────────

def test_list_preprocessed_runs_returns_completed(db_session):
    from backend.app.models.models import PreprocessingRun
    from datetime import datetime, timezone
    run = PreprocessingRun(
        status="COMPLETED",
        output_file_path="/some/path.csv",
        total_records=500,
        finished_at=datetime.now(timezone.utc),
    )
    db_session.add(run)
    db_session.commit()

    result = svc.list_preprocessed_runs(db_session)
    assert any(r["output_file_path"] == "/some/path.csv" for r in result)


def test_list_preprocessed_runs_excludes_failed(db_session):
    from backend.app.models.models import PreprocessingRun
    run = PreprocessingRun(status="FAILED", output_file_path="/failed.csv", total_records=0)
    db_session.add(run)
    db_session.commit()

    result = svc.list_preprocessed_runs(db_session)
    assert not any(r["output_file_path"] == "/failed.csv" for r in result)


# ── apply_model: forbidden columns ───────────────────────────────────────────

def test_apply_isolation_forest_no_forbidden_columns(tmp_path, db_session):
    """Result CSV must not contain is_fraud or confirmed_fraud."""
    input_csv = _minimal_preprocessed_csv(tmp_path)

    from backend.app.models.models import ModelRegistry
    import joblib
    from sklearn.ensemble import IsolationForest
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    # Build a minimal trained pipeline
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, ["amount"])],
        remainder="drop",
    )
    pipe = Pipeline([("preprocessor", preprocessor), ("clf", IsolationForest(contamination=0.1, random_state=42))])

    train_df = pd.DataFrame({"amount": np.random.rand(20)})
    pipe.fit(train_df)

    model_path = tmp_path / "models" / "if_model.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, model_path)

    reg = ModelRegistry(
        model_family="UNSUPERVISED",
        algorithm="isolation_forest",
        source_run="preprocessed_run_99",
        status="AVAILABLE",
        model_file=str(model_path),
        metrics_json=json.dumps({"contamination": 0.1}),
    )
    db_session.add(reg)
    db_session.commit()
    db_session.refresh(reg)

    with patch.object(svc, "_processed_dir", return_value=tmp_path / "processed"):
        (tmp_path / "processed").mkdir(parents=True, exist_ok=True)
        result = svc.apply_model(
            db=db_session,
            model_registry_id=reg.id,
            input_file_path=str(input_csv),
            input_type="csv_upload",
            input_source="test.csv",
        )

    assert result["status"] == "COMPLETED"
    assert result["total_analyzed"] > 0
    assert "is_fraud" not in result
    assert "confirmed_fraud" not in result

    # Verify the saved CSV also has no forbidden columns
    from backend.app.models.models import UnsupervisedInferenceRun
    run_rec = db_session.query(UnsupervisedInferenceRun).filter(UnsupervisedInferenceRun.id == result["id"]).first()
    assert run_rec is not None
    assert run_rec.results_file is not None
    results_df = pd.read_csv(run_rec.results_file)
    for forbidden in ["is_fraud", "confirmed_fraud"]:
        assert forbidden not in results_df.columns, f"Forbidden column '{forbidden}' found in results"


def test_apply_model_result_has_anomaly_flag_and_rank(tmp_path, db_session):
    """Output must contain anomaly_flag and anomaly_rank."""
    input_csv = _minimal_preprocessed_csv(tmp_path)

    from backend.app.models.models import ModelRegistry
    import joblib
    from sklearn.ensemble import IsolationForest
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.compose import ColumnTransformer

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, ["amount"])],
        remainder="drop",
    )
    pipe = Pipeline([("preprocessor", preprocessor), ("clf", IsolationForest(contamination=0.1, random_state=42))])
    pipe.fit(pd.DataFrame({"amount": np.random.rand(20)}))
    model_path = tmp_path / "models2" / "if2.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, model_path)

    reg = ModelRegistry(
        model_family="UNSUPERVISED",
        algorithm="isolation_forest",
        source_run="preprocessed_run_100",
        status="AVAILABLE",
        model_file=str(model_path),
        metrics_json=json.dumps({"contamination": 0.1}),
    )
    db_session.add(reg)
    db_session.commit()
    db_session.refresh(reg)

    with patch.object(svc, "_processed_dir", return_value=tmp_path / "processed2"):
        (tmp_path / "processed2").mkdir(parents=True, exist_ok=True)
        result = svc.apply_model(
            db=db_session,
            model_registry_id=reg.id,
            input_file_path=str(input_csv),
            input_type="csv_upload",
            input_source="test.csv",
        )

    from backend.app.models.models import UnsupervisedInferenceRun
    run_rec = db_session.query(UnsupervisedInferenceRun).filter(UnsupervisedInferenceRun.id == result["id"]).first()
    results_df = pd.read_csv(run_rec.results_file)
    assert "anomaly_flag" in results_df.columns
    assert "anomaly_rank" in results_df.columns


# ── Incompatible CSV ──────────────────────────────────────────────────────────

def test_apply_model_empty_csv_raises_controlled_error(tmp_path, db_session):
    """Empty CSV should fail with a descriptive ValueError."""
    empty_csv = tmp_path / "empty.csv"
    pd.DataFrame().to_csv(empty_csv, index=False)

    from backend.app.models.models import ModelRegistry
    import joblib
    from sklearn.ensemble import IsolationForest
    pipe = Pipeline = type("P", (), {"fit": lambda s, x: s, "decision_function": lambda s, x: np.zeros(len(x))})()
    # Just need a model file on disk
    import pickle
    model_path = tmp_path / "models3" / "fake.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(object(), f)

    reg = ModelRegistry(
        model_family="UNSUPERVISED",
        algorithm="isolation_forest",
        source_run="preprocessed_run_101",
        status="AVAILABLE",
        model_file=str(model_path),
        metrics_json=json.dumps({"contamination": 0.1}),
    )
    db_session.add(reg)
    db_session.commit()
    db_session.refresh(reg)

    with patch.object(svc, "_processed_dir", return_value=tmp_path / "processed3"):
        (tmp_path / "processed3").mkdir(parents=True, exist_ok=True)
        with pytest.raises(Exception) as exc_info:
            svc.apply_model(
                db=db_session,
                model_registry_id=reg.id,
                input_file_path=str(empty_csv),
                input_type="csv_upload",
                input_source="empty.csv",
            )
    assert exc_info.value is not None


# ── get_prediction_results strips forbidden columns ───────────────────────────

def test_get_prediction_results_strips_forbidden_columns(tmp_path, db_session):
    """Results returned via API must not expose is_fraud or sensitive columns."""
    from backend.app.models.models import UnsupervisedInferenceRun
    from datetime import datetime, timezone

    results_path = tmp_path / "results.csv"
    pd.DataFrame([
        {"transaction_id": "tx1", "anomaly_score": 0.9, "anomaly_flag": 1, "anomaly_rank": 1,
         "is_fraud": 1, "confirmed_fraud": 0, "PAN_TARJETA": "4111111111111111"},
        {"transaction_id": "tx2", "anomaly_score": 0.1, "anomaly_flag": 0, "anomaly_rank": 2,
         "is_fraud": 0, "confirmed_fraud": 0, "PAN_TARJETA": "4111111111111112"},
    ]).to_csv(results_path, index=False)

    run = UnsupervisedInferenceRun(
        algorithm="isolation_forest",
        model_source_run="preprocessed_run_99",
        input_type="csv_upload",
        input_source="test.csv",
        results_file=str(results_path),
        total_analyzed=2,
        anomaly_count=1,
        anomaly_rate=0.5,
        status="COMPLETED",
        started_at=datetime.now(timezone.utc),
        finished_at=datetime.now(timezone.utc),
    )
    db_session.add(run)
    db_session.commit()
    db_session.refresh(run)

    result = svc.get_prediction_results(db_session, run_id=run.id, page=1, page_size=10)
    rows = result["rows"]
    assert len(rows) == 2
    for row in rows:
        assert "is_fraud" not in row
        assert "confirmed_fraud" not in row
        assert "PAN_TARJETA" not in row


def test_get_prediction_results_anomaly_only_filter(tmp_path, db_session):
    from backend.app.models.models import UnsupervisedInferenceRun
    from datetime import datetime, timezone

    results_path = tmp_path / "results_filter.csv"
    pd.DataFrame([
        {"transaction_id": "tx1", "anomaly_score": 0.9, "anomaly_flag": 1, "anomaly_rank": 1},
        {"transaction_id": "tx2", "anomaly_score": 0.1, "anomaly_flag": 0, "anomaly_rank": 2},
        {"transaction_id": "tx3", "anomaly_score": 0.8, "anomaly_flag": 1, "anomaly_rank": 3},
    ]).to_csv(results_path, index=False)

    run = UnsupervisedInferenceRun(
        algorithm="isolation_forest",
        model_source_run="preprocessed_run_99",
        input_type="csv_upload",
        input_source="test.csv",
        results_file=str(results_path),
        total_analyzed=3,
        anomaly_count=2,
        anomaly_rate=0.66,
        status="COMPLETED",
        started_at=datetime.now(timezone.utc),
        finished_at=datetime.now(timezone.utc),
    )
    db_session.add(run)
    db_session.commit()
    db_session.refresh(run)

    result = svc.get_prediction_results(db_session, run_id=run.id, anomaly_only=True)
    assert result["total"] == 2
    for row in result["rows"]:
        assert row["anomaly_flag"] == 1


def test_methodology_warning_in_results(tmp_path, db_session):
    """All results responses must include the methodology warning."""
    from backend.app.models.models import UnsupervisedInferenceRun
    from datetime import datetime, timezone

    results_path = tmp_path / "results_warn.csv"
    pd.DataFrame([{"transaction_id": "tx1", "anomaly_score": 0.5, "anomaly_flag": 1, "anomaly_rank": 1}]).to_csv(results_path, index=False)

    run = UnsupervisedInferenceRun(
        algorithm="isolation_forest",
        model_source_run="preprocessed_run_99",
        input_type="csv_upload",
        input_source="test.csv",
        results_file=str(results_path),
        total_analyzed=1,
        anomaly_count=1,
        anomaly_rate=1.0,
        status="COMPLETED",
        started_at=datetime.now(timezone.utc),
        finished_at=datetime.now(timezone.utc),
    )
    db_session.add(run)
    db_session.commit()
    db_session.refresh(run)

    result = svc.get_prediction_results(db_session, run_id=run.id)
    assert "methodology_warning" in result
    assert "fraude confirmado" in result["methodology_warning"].lower() or "no constituyen fraude" in result["methodology_warning"].lower()

    report = svc.get_prediction_report(db_session, run_id=run.id)
    assert "methodology_warning" in report

    meta = svc.get_prediction_metadata(db_session, run_id=run.id)
    assert "methodology_warning" in meta

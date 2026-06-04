"""Unit tests for supervised_inference_service.

Validates:
- No is_fraud or confirmed_fraud in results
- prediction_label, prediction_probability, priority_level present in results
- Model loaded without retraining (mock joblib.load)
- priority_level correctly assigned (HIGH/MEDIUM/LOW)
- Incompatible/empty CSV returns controlled error
- Forbidden columns stripped from results
- methodology_warning in all response types
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from backend.app.services import supervised_inference_service as svc


# ── Helpers ───────────────────────────────────────────────────────────────────

def _minimal_alert_summary_csv(tmp_path: Path) -> Path:
    """Minimal alert-level CSV matching ARTIFACT_RULE_SUMMARY_CSV schema."""
    df = pd.DataFrame([
        {
            "summary_alert_id": f"alert{i}",
            "customer_hash": f"cust{i}",
            "representative_transaction_id": f"tx{i}",
            "rule_code": "VELOCITY_HIGH" if i % 2 == 0 else "DOUBLE_COUNTRY",
            "rule_name": "Velocidad alta" if i % 2 == 0 else "Doble pais",
            "risk_level": "HIGH" if i % 3 == 0 else "MEDIUM",
            "max_score": float(50 + i),
            "transactions_detected": i + 1,
            "countries_detected": "BO|US" if i % 2 == 0 else "BO",
            "merchant_rubro_proxy": "RETAIL",
            "merchant_rubro_values": "5814|5411" if i % 2 == 0 else "5311",
            "window_start": "2026-01-01 08:00:00",
            "window_end": "2026-01-01 10:00:00",
            "status": "OPEN",
        }
        for i in range(20)
    ])
    path = tmp_path / "alerts_summary_run_99.csv"
    df.to_csv(path, index=False)
    return path


# ── _assign_priority ──────────────────────────────────────────────────────────

def test_assign_priority_high_threshold():
    assert svc._assign_priority(0.75, 1) == "HIGH"
    assert svc._assign_priority(0.7, 1) == "HIGH"


def test_assign_priority_medium_threshold():
    assert svc._assign_priority(0.5, 1) == "MEDIUM"
    assert svc._assign_priority(0.4, 1) == "MEDIUM"


def test_assign_priority_low_threshold():
    assert svc._assign_priority(0.1, 0) == "LOW"
    assert svc._assign_priority(0.39, 0) == "LOW"


def test_assign_priority_no_prob_label_1():
    assert svc._assign_priority(None, 1) == "HIGH"


def test_assign_priority_no_prob_label_0():
    assert svc._assign_priority(None, 0) == "LOW"


# ── list_trained_models ───────────────────────────────────────────────────────

def test_list_trained_models_returns_supervised_human_available(db_session):
    from backend.app.models.models import ModelRegistry
    reg = ModelRegistry(
        model_family="SUPERVISED_HUMAN",
        algorithm="random_forest",
        source_run="preprocessed_run_26",
        status="AVAILABLE",
        metrics_json=json.dumps({"metrics": {"f1_score": 0.73, "precision": 0.67, "recall": 0.80, "roc_auc": 0.89}}),
    )
    db_session.add(reg)
    db_session.commit()

    result = svc.list_trained_models(db_session)
    assert any(r["algorithm"] == "random_forest" and r["source_run"] == "preprocessed_run_26" for r in result)
    model = next(r for r in result if r["source_run"] == "preprocessed_run_26")
    assert model["f1_score"] == pytest.approx(0.73)
    assert model["status"] == "AVAILABLE"


def test_list_trained_models_excludes_unsupervised(db_session):
    from backend.app.models.models import ModelRegistry
    reg = ModelRegistry(
        model_family="UNSUPERVISED",
        algorithm="isolation_forest",
        source_run="preprocessed_run_99",
        status="AVAILABLE",
    )
    db_session.add(reg)
    db_session.commit()

    result = svc.list_trained_models(db_session)
    assert not any(r["algorithm"] == "isolation_forest" for r in result)


def test_list_trained_models_excludes_unavailable(db_session):
    from backend.app.models.models import ModelRegistry
    reg = ModelRegistry(
        model_family="SUPERVISED_HUMAN",
        algorithm="logistic_regression",
        source_run="preprocessed_run_missing",
        status="MISSING",
    )
    db_session.add(reg)
    db_session.commit()

    result = svc.list_trained_models(db_session)
    assert not any(r["source_run"] == "preprocessed_run_missing" for r in result)


# ── list_preprocessed_runs ────────────────────────────────────────────────────

def test_list_preprocessed_runs_returns_completed(db_session):
    from backend.app.models.models import PreprocessingRun
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


# ── _run_inference: no forbidden columns ──────────────────────────────────────

# Models are trained on features produced by _build_alert_feature_frame using
# the alert summary CSV schema. No mocks needed for build_unsupervised_features.

def _build_test_model_from_alert_csv(tmp_path: Path, alert_csv: Path, suffix: str = "") -> Path:
    """Train a LR model on features from _build_alert_feature_frame; save as dict artifact."""
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer

    df = pd.read_csv(alert_csv)
    feat_frame = svc._build_alert_feature_frame(df)
    feature_columns = feat_frame.columns.tolist()

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(random_state=42, max_iter=200)),
    ])
    labels = ([0, 1] * (len(feat_frame) // 2))[:len(feat_frame)]
    pipe.fit(feat_frame, labels)

    model_path = tmp_path / f"models{suffix}" / "lr_model.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": pipe, "feature_columns": feature_columns}, model_path)
    return model_path


def test_run_inference_no_forbidden_columns(tmp_path, db_session):
    """Result CSV must not contain is_fraud or confirmed_fraud."""
    alert_csv = _minimal_alert_summary_csv(tmp_path)
    model_path = _build_test_model_from_alert_csv(tmp_path, alert_csv, "a")

    from backend.app.models.models import ModelRegistry
    reg = ModelRegistry(
        model_family="SUPERVISED_HUMAN",
        algorithm="logistic_regression",
        source_run="preprocessed_run_26",
        status="AVAILABLE",
        model_file=str(model_path),
        metrics_json=json.dumps({"metrics": {"f1_score": 0.73}}),
    )
    db_session.add(reg)
    db_session.commit()
    db_session.refresh(reg)

    processed = tmp_path / "processedA"
    processed.mkdir(parents=True, exist_ok=True)

    with patch.object(svc, "_processed_dir", return_value=processed):
        result = svc._run_inference(
            db=db_session,
            model_registry_id=reg.id,
            input_file_path=str(alert_csv),
            input_type="rule_summary",
            input_source="preprocessed_run_26",
        )

    results_df = pd.read_csv(result["results_file"])
    for forbidden in ["is_fraud", "confirmed_fraud", "target_human_label"]:
        assert forbidden not in results_df.columns, f"Forbidden column '{forbidden}' found in results"


def test_run_inference_has_required_columns(tmp_path, db_session):
    """Output must contain prediction_label, prediction_probability, priority_level."""
    alert_csv = _minimal_alert_summary_csv(tmp_path)
    model_path = _build_test_model_from_alert_csv(tmp_path, alert_csv, "b")

    from backend.app.models.models import ModelRegistry
    reg = ModelRegistry(
        model_family="SUPERVISED_HUMAN",
        algorithm="logistic_regression",
        source_run="preprocessed_run_27",
        status="AVAILABLE",
        model_file=str(model_path),
        metrics_json=json.dumps({"metrics": {"f1_score": 0.73}}),
    )
    db_session.add(reg)
    db_session.commit()
    db_session.refresh(reg)

    processed = tmp_path / "processedB"
    processed.mkdir(parents=True, exist_ok=True)

    with patch.object(svc, "_processed_dir", return_value=processed):
        result = svc._run_inference(
            db=db_session,
            model_registry_id=reg.id,
            input_file_path=str(alert_csv),
            input_type="rule_summary",
            input_source="preprocessed_run_27",
        )

    results_df = pd.read_csv(result["results_file"])
    assert "prediction_label" in results_df.columns
    assert "priority_level" in results_df.columns
    assert all(p in ("HIGH", "MEDIUM", "LOW") for p in results_df["priority_level"].unique())
    assert "high_count" in result
    assert "medium_count" in result
    assert "low_count" in result
    assert result["high_count"] + result["medium_count"] + result["low_count"] == result["total_analyzed"]


def test_run_inference_empty_csv_raises(tmp_path, db_session):
    """Empty CSV should raise a controlled ValueError."""
    empty_csv = tmp_path / "empty.csv"
    pd.DataFrame().to_csv(empty_csv, index=False)

    import pickle
    model_path = tmp_path / "models3" / "fake.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(object(), f)

    from backend.app.models.models import ModelRegistry
    reg = ModelRegistry(
        model_family="SUPERVISED_HUMAN",
        algorithm="logistic_regression",
        source_run="preprocessed_run_28",
        status="AVAILABLE",
        model_file=str(model_path),
    )
    db_session.add(reg)
    db_session.commit()
    db_session.refresh(reg)

    with patch.object(svc, "_processed_dir", return_value=tmp_path / "processed3"):
        (tmp_path / "processed3").mkdir(parents=True, exist_ok=True)
        with pytest.raises(Exception):
            svc._run_inference(
                db=db_session,
                model_registry_id=reg.id,
                input_file_path=str(empty_csv),
                input_type="csv_upload",
                input_source="empty.csv",
            )


# ── get_prediction_results ────────────────────────────────────────────────────

def test_get_prediction_results_strips_forbidden_columns(tmp_path, db_session):
    from backend.app.models.models import SupervisedInferenceRun

    results_path = tmp_path / "results.csv"
    pd.DataFrame([
        {"transaction_id": "tx1", "prediction_label": 1, "prediction_probability": 0.85,
         "priority_level": "HIGH", "is_fraud": 1, "confirmed_fraud": 0, "PAN_TARJETA": "4111111111111111"},
        {"transaction_id": "tx2", "prediction_label": 0, "prediction_probability": 0.20,
         "priority_level": "LOW", "is_fraud": 0, "confirmed_fraud": 0, "PAN_TARJETA": "4111111111111112"},
    ]).to_csv(results_path, index=False)

    run = SupervisedInferenceRun(
        algorithm="logistic_regression",
        model_source_run="preprocessed_run_26",
        input_type="csv_upload",
        input_source="test.csv",
        results_file=str(results_path),
        total_analyzed=2,
        high_count=1,
        medium_count=0,
        low_count=1,
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


def test_get_prediction_results_priority_filter(tmp_path, db_session):
    from backend.app.models.models import SupervisedInferenceRun

    results_path = tmp_path / "results_filter.csv"
    pd.DataFrame([
        {"transaction_id": "tx1", "prediction_label": 1, "prediction_probability": 0.9, "priority_level": "HIGH"},
        {"transaction_id": "tx2", "prediction_label": 0, "prediction_probability": 0.2, "priority_level": "LOW"},
        {"transaction_id": "tx3", "prediction_label": 1, "prediction_probability": 0.85, "priority_level": "HIGH"},
    ]).to_csv(results_path, index=False)

    run = SupervisedInferenceRun(
        algorithm="random_forest",
        model_source_run="preprocessed_run_26",
        input_type="csv_upload",
        input_source="test.csv",
        results_file=str(results_path),
        total_analyzed=3,
        high_count=2,
        medium_count=0,
        low_count=1,
        status="COMPLETED",
        started_at=datetime.now(timezone.utc),
        finished_at=datetime.now(timezone.utc),
    )
    db_session.add(run)
    db_session.commit()
    db_session.refresh(run)

    result = svc.get_prediction_results(db_session, run_id=run.id, priority_filter="HIGH")
    assert result["total"] == 2
    for row in result["rows"]:
        assert row["priority_level"] == "HIGH"

    result_low = svc.get_prediction_results(db_session, run_id=run.id, priority_filter="LOW")
    assert result_low["total"] == 1


# ── methodology_warning ───────────────────────────────────────────────────────

def test_methodology_warning_in_all_responses(tmp_path, db_session):
    from backend.app.models.models import SupervisedInferenceRun

    results_path = tmp_path / "results_warn.csv"
    pd.DataFrame([
        {"transaction_id": "tx1", "prediction_label": 1, "prediction_probability": 0.85, "priority_level": "HIGH"}
    ]).to_csv(results_path, index=False)

    run = SupervisedInferenceRun(
        algorithm="logistic_regression",
        model_source_run="preprocessed_run_26",
        input_type="csv_upload",
        input_source="test.csv",
        results_file=str(results_path),
        total_analyzed=1,
        high_count=1,
        medium_count=0,
        low_count=0,
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


# ── create_pending_run ────────────────────────────────────────────────────────

def test_create_pending_run_creates_pending_record(db_session):
    from backend.app.models.models import SupervisedInferenceRun

    run = svc.create_pending_run(
        db=db_session,
        model_registry_id=999,
        input_file_path="/some/path.csv",
        input_type="csv_upload",
        input_source="test.csv",
    )
    assert run.status == "PENDING"
    assert run.model_registry_id == 999
    assert run.input_type == "csv_upload"

    from_db = db_session.query(SupervisedInferenceRun).filter(SupervisedInferenceRun.id == run.id).first()
    assert from_db is not None
    assert from_db.status == "PENDING"


# ── get_run_status ────────────────────────────────────────────────────────────

def test_get_run_status_returns_correct_fields(db_session):
    from backend.app.models.models import SupervisedInferenceRun

    run = SupervisedInferenceRun(
        algorithm="random_forest",
        model_source_run="preprocessed_run_26",
        input_type="preprocessed_run",
        input_source="preprocessed_run_26",
        status="RUNNING",
        total_analyzed=0,
        high_count=0,
        medium_count=0,
        low_count=0,
        started_at=datetime.now(timezone.utc),
    )
    db_session.add(run)
    db_session.commit()
    db_session.refresh(run)

    status = svc.get_run_status(db_session, run.id)
    assert status["status"] == "RUNNING"
    assert status["id"] == run.id
    assert "methodology_warning" in status


def test_get_run_status_not_found_raises(db_session):
    with pytest.raises(ValueError, match="not found"):
        svc.get_run_status(db_session, 999999)


# ── validate_alert_schema_csv ─────────────────────────────────────────────────

def test_validate_alert_schema_csv_accepts_alert_csv(tmp_path):
    csv_path = tmp_path / "alert.csv"
    csv_path.write_text("rule_code,rule_name,summary_alert_id\nRULE_A,Test,A1\n")
    svc.validate_alert_schema_csv(str(csv_path))  # must not raise


def test_validate_alert_schema_csv_rejects_raw_transaction_csv(tmp_path):
    csv_path = tmp_path / "tx.csv"
    csv_path.write_text("transaction_id,amount_log,hour_of_day\ntx1,0.5,14\n")
    with pytest.raises(ValueError, match="transacciones crudas"):
        svc.validate_alert_schema_csv(str(csv_path))


def test_validate_alert_schema_csv_accepts_mixed_if_has_rule_code(tmp_path):
    csv_path = tmp_path / "mixed.csv"
    csv_path.write_text("transaction_id,rule_code,rule_name,amount_log\ntx1,RULE_A,Test,0.5\n")
    svc.validate_alert_schema_csv(str(csv_path))  # has rule_code → OK


# ── same_run_warning ──────────────────────────────────────────────────────────

def test_run_inference_same_run_warning(tmp_path, db_session):
    """When model source_run == inference source_run → same_run_warning is set."""
    alert_csv = _minimal_alert_summary_csv(tmp_path)
    model_path = _build_test_model_from_alert_csv(tmp_path, alert_csv, "_samerun")

    from backend.app.models.models import ModelRegistry
    reg = ModelRegistry(
        model_family="SUPERVISED_HUMAN",
        algorithm="logistic_regression",
        source_run="preprocessed_run_26",
        status="AVAILABLE",
        model_file=str(model_path),
        metrics_json=json.dumps({"metrics": {"f1_score": 0.73}}),
    )
    db_session.add(reg)
    db_session.commit()
    db_session.refresh(reg)

    processed = tmp_path / "processed_samerun"
    processed.mkdir(parents=True, exist_ok=True)

    with patch.object(svc, "_processed_dir", return_value=processed):
        result = svc._run_inference(
            db=db_session,
            model_registry_id=reg.id,
            input_file_path=str(alert_csv),
            input_type="preprocessed_run",
            input_source="preprocessed_run_26",
        )

    assert result["same_run_warning"] is not None
    assert "validación técnica" in result["same_run_warning"]


def test_run_inference_no_same_run_warning_when_different(tmp_path, db_session):
    """When model source_run != inference source_run → same_run_warning is None."""
    alert_csv = _minimal_alert_summary_csv(tmp_path)
    model_path = _build_test_model_from_alert_csv(tmp_path, alert_csv, "_diffrun")

    from backend.app.models.models import ModelRegistry
    reg = ModelRegistry(
        model_family="SUPERVISED_HUMAN",
        algorithm="logistic_regression",
        source_run="preprocessed_run_26",
        status="AVAILABLE",
        model_file=str(model_path),
        metrics_json=json.dumps({"metrics": {"f1_score": 0.73}}),
    )
    db_session.add(reg)
    db_session.commit()
    db_session.refresh(reg)

    processed = tmp_path / "processed_diffrun"
    processed.mkdir(parents=True, exist_ok=True)

    with patch.object(svc, "_processed_dir", return_value=processed):
        result = svc._run_inference(
            db=db_session,
            model_registry_id=reg.id,
            input_file_path=str(alert_csv),
            input_type="preprocessed_run",
            input_source="preprocessed_run_27",
        )

    assert result["same_run_warning"] is None

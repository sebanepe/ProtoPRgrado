"""Tests unitarios para el servicio de scoring por lotes D1."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from backend.app.ml.batch_scoring_service import (
    FORBIDDEN_OUTPUT_COLUMNS,
    LOW_MEDIUM_THRESHOLD,
    MEDIUM_HIGH_THRESHOLD,
    _build_results_df,
    _feature_frame_for_scoring,
    run_batch_scoring,
)
from backend.app.ml.validate_scoring_outputs import (
    VERDICT_INVALID,
    VERDICT_READY,
    validate_scoring_outputs,
)
from backend.app.models.models import BatchScoringRun, ModelRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dataset_csv(path: Path, token: str) -> Path:
    """CSV mínimo compatible con el dataset supervisado humano."""
    csv_path = path / f"supervised_human_alert_dataset_run_{token}.csv"
    df = pd.DataFrame(
        [
            {
                "source_run": f"preprocessed_run_{token}",
                "summary_alert_id": f"alert_{i}",
                "representative_transaction_id": f"tx_{i}",
                "customer_hash": f"hash_{i}",
                "rule_code": "RULE_VELOCITY_CARD_DAY",
                "rule_name": "Card Velocity Day",
                "risk_level": "HIGH" if i % 2 == 0 else "MEDIUM",
                "max_score": 0.9,
                "transactions_detected": 5,
                "countries_detected": "BO",
                "merchant_rubro_proxy": "5814",
                "merchant_rubro_values": "5814",
                "window_start": "2024-01-01T00:00:00",
                "window_end": "2024-01-01T01:00:00",
                "duration_minutes": 60,
                "status": "NEW",
                "countries_count": 1,
                "has_multiple_countries": False,
                "is_high_risk_rule": True,
                "is_velocity_rule": True,
                "is_double_country_rule": False,
                "is_mcc_risk_rule": False,
                "is_card_present_rule": True,
                "is_card_absent_rule": False,
                "is_internet_related": False,
                "is_atm_or_cash_related": False,
                "human_review_status": "REVIEWED",
                "reviewed_at": "2024-01-01T02:00:00",
                "reviewed_by": "analyst_1",
                "human_review_comment": "ok",
                "target_human_label": i % 2,
                "target_label_source": "HUMAN_REVIEW",
                "target_label_meaning": "0=DISMISSED,1=CONFIRMED_FRAUD",
            }
            for i in range(4)
        ]
    )
    df.to_csv(csv_path, index=False)
    return csv_path


def _make_mock_pkl(path: Path, token: str, algorithm: str) -> Path:
    """Guarda un pkl mínimo con la misma estructura que el training real."""
    feature_columns = [
        "max_score",
        "transactions_detected",
        "duration_minutes",
        "countries_count",
        "has_multiple_countries",
        "is_high_risk_rule",
        "is_velocity_rule",
        "is_double_country_rule",
        "is_mcc_risk_rule",
        "is_card_present_rule",
        "is_card_absent_rule",
        "is_internet_related",
        "is_atm_or_cash_related",
        "risk_level_HIGH",
        "risk_level_MEDIUM",
        "risk_level_nan",
    ]
    X_dummy, y_dummy = make_classification(n_samples=20, n_features=len(feature_columns), random_state=42)
    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    clf.fit(X_dummy, y_dummy)

    pkl_path = path / f"supervised_human_{algorithm}_run_{token}.pkl"
    joblib.dump({"model": clf, "feature_columns": feature_columns}, pkl_path)
    return pkl_path


def _register_model(db_session, source_run: str, algorithm: str, pkl_path: Path) -> ModelRegistry:
    row = ModelRegistry(
        model_family="SUPERVISED_HUMAN",
        algorithm=algorithm,
        source_run=source_run,
        run_token=source_run.split("_")[-1],
        model_file=str(pkl_path),
        status="AVAILABLE",
    )
    db_session.add(row)
    db_session.commit()
    db_session.refresh(row)
    return row


# ---------------------------------------------------------------------------
# _feature_frame_for_scoring
# ---------------------------------------------------------------------------

class TestFeatureFrameForScoring:
    def test_aligns_to_feature_columns(self):
        feature_columns = ["col_a", "col_b", "col_c"]
        df = pd.DataFrame({"col_a": [1, 2], "col_b": [3, 4], "extra_col": [5, 6]})
        result = _feature_frame_for_scoring(df, feature_columns)
        assert list(result.columns) == feature_columns
        assert result.shape == (2, 3)

    def test_fills_missing_columns_with_zero(self):
        feature_columns = ["col_a", "col_b", "col_missing"]
        df = pd.DataFrame({"col_a": [1, 2], "col_b": [3, 4]})
        result = _feature_frame_for_scoring(df, feature_columns)
        assert (result["col_missing"] == 0).all()

    def test_drops_forbidden_columns(self):
        feature_columns = ["max_score"]
        df = pd.DataFrame(
            {
                "max_score": [0.9, 0.5],
                "is_fraud": [1, 0],
                "confirmed_fraud": [1, 0],
                "PAN_TARJETA": ["1234", "5678"],
            }
        )
        result = _feature_frame_for_scoring(df, feature_columns)
        assert "is_fraud" not in result.columns
        assert "confirmed_fraud" not in result.columns
        assert "PAN_TARJETA" not in result.columns

    def test_converts_bool_to_int(self):
        feature_columns = ["flag_a", "flag_b"]
        df = pd.DataFrame({"flag_a": [True, False], "flag_b": [False, True]})
        result = _feature_frame_for_scoring(df, feature_columns)
        assert result["flag_a"].dtype in (int, np.int64, np.int32)

    def test_drops_training_identity_columns(self):
        feature_columns = ["max_score"]
        df = pd.DataFrame(
            {
                "max_score": [0.9],
                "source_run": ["preprocessed_run_26"],
                "summary_alert_id": ["alert_1"],
                "customer_hash": ["abc"],
                "target_human_label": [1],
                "human_review_comment": ["text"],
            }
        )
        result = _feature_frame_for_scoring(df, feature_columns)
        for dropped in ("source_run", "summary_alert_id", "customer_hash", "target_human_label"):
            assert dropped not in result.columns


# ---------------------------------------------------------------------------
# _build_results_df
# ---------------------------------------------------------------------------

class TestBuildResultsDf:
    def _make_mock_model_and_features(self):
        feature_columns = ["max_score", "transactions_detected", "countries_count"]
        X_dummy = np.array([[0.9, 5, 1], [0.3, 2, 1], [0.8, 10, 2], [0.2, 1, 1]])
        y_dummy = np.array([1, 0, 1, 0])
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_dummy, y_dummy)
        return clf, feature_columns

    def _make_df(self):
        return pd.DataFrame(
            {
                "source_run": ["preprocessed_run_99"] * 4,
                "summary_alert_id": ["a1", "a2", "a3", "a4"],
                "representative_transaction_id": ["t1", "t2", "t3", "t4"],
                "customer_hash": ["h1", "h2", "h3", "h4"],
                "rule_code": ["RULE_V"] * 4,
                "rule_name": ["Velocity"] * 4,
                "risk_level": ["HIGH", "MEDIUM", "HIGH", "LOW"],
                "max_score": [0.9, 0.3, 0.8, 0.2],
                "transactions_detected": [5, 2, 10, 1],
                "countries_detected": ["BO"] * 4,
                "merchant_rubro_proxy": ["5814"] * 4,
                "window_start": ["2024-01-01"] * 4,
                "window_end": ["2024-01-01"] * 4,
                "duration_minutes": [60] * 4,
                "countries_count": [1, 1, 2, 1],
                "target_human_label": [1, 0, 1, 0],
                "is_fraud": [1, 0, 1, 0],
                "confirmed_fraud": [1, 0, 1, 0],
            }
        )

    def test_no_forbidden_output_columns(self):
        clf, feature_columns = self._make_mock_model_and_features()
        df = self._make_df()
        result = _build_results_df(df, clf, feature_columns, "preprocessed_run_99", "logistic_regression")
        for col in FORBIDDEN_OUTPUT_COLUMNS:
            assert col not in result.columns, f"Columna prohibida presente: {col}"

    def test_contains_required_output_columns(self):
        clf, feature_columns = self._make_mock_model_and_features()
        df = self._make_df()
        result = _build_results_df(df, clf, feature_columns, "preprocessed_run_99", "logistic_regression")
        for col in ("ml_risk_score", "ml_risk_level", "algorithm", "scored_at", "summary_alert_id"):
            assert col in result.columns

    def test_risk_level_thresholds(self):
        """Verifica que los thresholds D1 (0.5/0.75) se aplican correctamente."""
        feature_columns = ["max_score"]

        class MockModel:
            def predict_proba(self, X):
                return np.column_stack([1 - X[:, 0], X[:, 0]])

        model = MockModel()
        df = pd.DataFrame(
            {
                "max_score": [0.3, 0.6, 0.8],
                "source_run": ["preprocessed_run_99"] * 3,
                "summary_alert_id": ["a1", "a2", "a3"],
                "representative_transaction_id": ["t1", "t2", "t3"],
                "customer_hash": ["h1", "h2", "h3"],
                "rule_code": ["R"] * 3,
                "rule_name": ["Rule"] * 3,
                "risk_level": ["LOW", "MEDIUM", "HIGH"],
                "transactions_detected": [1, 2, 3],
                "countries_detected": ["BO"] * 3,
                "merchant_rubro_proxy": ["5814"] * 3,
                "window_start": ["2024-01-01"] * 3,
                "window_end": ["2024-01-01"] * 3,
                "duration_minutes": [60] * 3,
                "countries_count": [1, 1, 2],
            }
        )
        result = _build_results_df(df, model, feature_columns, "preprocessed_run_99", "logistic_regression")
        levels = result["ml_risk_level"].tolist()
        assert levels[0] == "LOW", f"Expected LOW for 0.3, got {levels[0]}"
        assert levels[1] == "MEDIUM", f"Expected MEDIUM for 0.6, got {levels[1]}"
        assert levels[2] == "HIGH", f"Expected HIGH for 0.8, got {levels[2]}"


# ---------------------------------------------------------------------------
# run_batch_scoring (funcional con DB)
# ---------------------------------------------------------------------------

class TestRunBatchScoring:
    def test_blocked_dataset_not_found(self, db_session, tmp_path, monkeypatch):
        monkeypatch.setenv("PROJECT_PROCESSED_DIR", str(tmp_path))
        result = run_batch_scoring(
            "preprocessed_run_999",
            "logistic_regression",
            db=db_session,
        )
        assert result["status"] == "BLOCKED"
        assert "DATASET_NOT_FOUND" in result["verdict"]

    def test_blocked_model_not_found(self, db_session, tmp_path, monkeypatch):
        monkeypatch.setenv("PROJECT_PROCESSED_DIR", str(tmp_path))
        token = "888"
        dataset_path = _make_dataset_csv(tmp_path, token)
        result = run_batch_scoring(
            f"preprocessed_run_{token}",
            "random_forest",
            db=db_session,
            input_dataset_path=str(dataset_path),
        )
        assert result["status"] in ("BLOCKED", "FAILED")
        assert "MODEL_NOT_FOUND" in result.get("verdict", "") or "FAILED" in result.get("status", "")

    def test_completed_with_mock_pkl(self, db_session, tmp_path, monkeypatch):
        monkeypatch.setenv("PROJECT_PROCESSED_DIR", str(tmp_path))
        token = "42"
        source_run = f"preprocessed_run_{token}"
        dataset_path = _make_dataset_csv(tmp_path, token)
        pkl_path = _make_mock_pkl(tmp_path, token, "logistic_regression")
        _register_model(db_session, source_run, "logistic_regression", pkl_path)

        result = run_batch_scoring(
            source_run,
            "logistic_regression",
            db=db_session,
            input_dataset_path=str(dataset_path),
        )

        assert result["status"] == "COMPLETED", f"Esperado COMPLETED, got {result}"
        assert result["total_scored"] == 4
        assert result["high_count"] + result["medium_count"] + result["low_count"] == 4
        assert METHODOLOGY_WARNING_FRAGMENT in result["warnings"][0]

        db_run = (
            db_session.query(BatchScoringRun)
            .filter(BatchScoringRun.source_run == source_run)
            .first()
        )
        assert db_run is not None
        assert db_run.status == "COMPLETED"

        results_path = Path(result["results_file"])
        assert results_path.exists()
        df = pd.read_csv(results_path)
        for col in FORBIDDEN_OUTPUT_COLUMNS:
            assert col not in df.columns, f"Columna prohibida en resultados: {col}"
        assert "ml_risk_level" in df.columns
        assert set(df["ml_risk_level"].unique()).issubset({"HIGH", "MEDIUM", "LOW"})

    def test_metadata_file_created(self, db_session, tmp_path, monkeypatch):
        monkeypatch.setenv("PROJECT_PROCESSED_DIR", str(tmp_path))
        token = "43"
        source_run = f"preprocessed_run_{token}"
        dataset_path = _make_dataset_csv(tmp_path, token)
        pkl_path = _make_mock_pkl(tmp_path, token, "random_forest")
        _register_model(db_session, source_run, "random_forest", pkl_path)

        result = run_batch_scoring(
            source_run,
            "random_forest",
            db=db_session,
            input_dataset_path=str(dataset_path),
        )
        assert result["status"] == "COMPLETED"
        meta_path = Path(result["metadata_file"])
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["algorithm"] == "random_forest"
        assert meta["total_scored"] == 4
        assert meta["source_run"] == source_run

    def test_invalid_algorithm_raises(self, db_session):
        with pytest.raises(ValueError, match="algorithm"):
            run_batch_scoring("preprocessed_run_26", "neural_net", db=db_session)

    def test_results_no_is_fraud(self, db_session, tmp_path, monkeypatch):
        monkeypatch.setenv("PROJECT_PROCESSED_DIR", str(tmp_path))
        token = "44"
        source_run = f"preprocessed_run_{token}"
        dataset_path = _make_dataset_csv(tmp_path, token)
        pkl_path = _make_mock_pkl(tmp_path, token, "gradient_boosting")
        _register_model(db_session, source_run, "gradient_boosting", pkl_path)

        result = run_batch_scoring(
            source_run,
            "gradient_boosting",
            db=db_session,
            input_dataset_path=str(dataset_path),
        )
        assert result["status"] == "COMPLETED"
        df = pd.read_csv(result["results_file"])
        assert "is_fraud" not in df.columns
        assert "confirmed_fraud" not in df.columns

    def test_artifact_registry_populated(self, db_session, tmp_path, monkeypatch):
        from backend.app.models.models import ArtifactRegistry
        from backend.app.services.artifact_registry_service import ARTIFACT_SCORING_RESULTS

        monkeypatch.setenv("PROJECT_PROCESSED_DIR", str(tmp_path))
        token = "45"
        source_run = f"preprocessed_run_{token}"
        dataset_path = _make_dataset_csv(tmp_path, token)
        pkl_path = _make_mock_pkl(tmp_path, token, "logistic_regression")
        _register_model(db_session, source_run, "logistic_regression", pkl_path)

        result = run_batch_scoring(
            source_run,
            "logistic_regression",
            db=db_session,
            input_dataset_path=str(dataset_path),
        )
        assert result["status"] == "COMPLETED"

        artifact = (
            db_session.query(ArtifactRegistry)
            .filter(
                ArtifactRegistry.source_run == source_run,
                ArtifactRegistry.artifact_type == ARTIFACT_SCORING_RESULTS,
            )
            .first()
        )
        assert artifact is not None
        assert artifact.phase == "PHASE_D1"

    def test_scoring_run_db_record(self, db_session, tmp_path, monkeypatch):
        monkeypatch.setenv("PROJECT_PROCESSED_DIR", str(tmp_path))
        token = "46"
        source_run = f"preprocessed_run_{token}"
        dataset_path = _make_dataset_csv(tmp_path, token)
        pkl_path = _make_mock_pkl(tmp_path, token, "logistic_regression")
        _register_model(db_session, source_run, "logistic_regression", pkl_path)

        result = run_batch_scoring(
            source_run,
            "logistic_regression",
            db=db_session,
            input_dataset_path=str(dataset_path),
        )
        assert result["status"] == "COMPLETED"

        run_id = result["batch_scoring_run_id"]
        db_run = db_session.query(BatchScoringRun).filter(BatchScoringRun.id == run_id).first()
        assert db_run is not None
        assert db_run.status == "COMPLETED"
        assert db_run.total_scored == 4
        assert db_run.finished_at is not None


METHODOLOGY_WARNING_FRAGMENT = "No confirma fraude automáticamente"


# ---------------------------------------------------------------------------
# validate_scoring_outputs
# ---------------------------------------------------------------------------

class TestValidateScoringOutputs:
    def _make_valid_csv(self, path: Path) -> Path:
        csv_path = path / "scoring_results_test.csv"
        df = pd.DataFrame(
            [
                {
                    "source_run": "preprocessed_run_26",
                    "summary_alert_id": f"alert_{i}",
                    "customer_hash": f"hash_{i}",
                    "ml_risk_score": 0.3 + i * 0.2,
                    "ml_risk_level": ["LOW", "MEDIUM", "HIGH"][i % 3],
                    "algorithm": "logistic_regression",
                    "scored_at": "2024-01-01T00:00:00",
                }
                for i in range(3)
            ]
        )
        df.to_csv(csv_path, index=False)
        return csv_path

    def test_valid_outputs_ready(self, tmp_path):
        csv = self._make_valid_csv(tmp_path)
        result = validate_scoring_outputs(csv)
        assert result["verdict"] == VERDICT_READY
        assert not result["errors"]

    def test_missing_file(self, tmp_path):
        result = validate_scoring_outputs(tmp_path / "nonexistent.csv")
        assert result["verdict"] == VERDICT_INVALID

    def test_forbidden_column_is_fraud(self, tmp_path):
        csv_path = tmp_path / "bad.csv"
        df = pd.DataFrame(
            {
                "source_run": ["preprocessed_run_26"],
                "summary_alert_id": ["a1"],
                "customer_hash": ["h1"],
                "ml_risk_score": [0.5],
                "ml_risk_level": ["MEDIUM"],
                "algorithm": ["logistic_regression"],
                "scored_at": ["2024-01-01"],
                "is_fraud": [1],
            }
        )
        df.to_csv(csv_path, index=False)
        result = validate_scoring_outputs(csv_path)
        assert result["verdict"] == VERDICT_INVALID
        assert any("is_fraud" in e for e in result["errors"])

    def test_forbidden_column_confirmed_fraud(self, tmp_path):
        csv_path = tmp_path / "bad2.csv"
        df = pd.DataFrame(
            {
                "source_run": ["preprocessed_run_26"],
                "summary_alert_id": ["a1"],
                "customer_hash": ["h1"],
                "ml_risk_score": [0.5],
                "ml_risk_level": ["MEDIUM"],
                "algorithm": ["logistic_regression"],
                "scored_at": ["2024-01-01"],
                "confirmed_fraud": [0],
            }
        )
        df.to_csv(csv_path, index=False)
        result = validate_scoring_outputs(csv_path)
        assert result["verdict"] == VERDICT_INVALID
        assert any("confirmed_fraud" in e for e in result["errors"])

    def test_missing_required_column(self, tmp_path):
        csv_path = tmp_path / "missing_col.csv"
        df = pd.DataFrame(
            {
                "source_run": ["preprocessed_run_26"],
                "summary_alert_id": ["a1"],
                "customer_hash": ["h1"],
                "ml_risk_level": ["LOW"],
                "algorithm": ["logistic_regression"],
                "scored_at": ["2024-01-01"],
            }
        )
        df.to_csv(csv_path, index=False)
        result = validate_scoring_outputs(csv_path)
        assert result["verdict"] == VERDICT_INVALID
        assert any("ml_risk_score" in e for e in result["errors"])

    def test_invalid_risk_level(self, tmp_path):
        csv_path = tmp_path / "bad_level.csv"
        df = pd.DataFrame(
            {
                "source_run": ["preprocessed_run_26"],
                "summary_alert_id": ["a1"],
                "customer_hash": ["h1"],
                "ml_risk_score": [0.5],
                "ml_risk_level": ["CRITICAL"],
                "algorithm": ["logistic_regression"],
                "scored_at": ["2024-01-01"],
            }
        )
        df.to_csv(csv_path, index=False)
        result = validate_scoring_outputs(csv_path)
        assert result["verdict"] == VERDICT_INVALID
        assert any("CRITICAL" in e for e in result["errors"])

    def test_score_out_of_range(self, tmp_path):
        csv_path = tmp_path / "bad_score.csv"
        df = pd.DataFrame(
            {
                "source_run": ["preprocessed_run_26"],
                "summary_alert_id": ["a1"],
                "customer_hash": ["h1"],
                "ml_risk_score": [1.5],
                "ml_risk_level": ["HIGH"],
                "algorithm": ["logistic_regression"],
                "scored_at": ["2024-01-01"],
            }
        )
        df.to_csv(csv_path, index=False)
        result = validate_scoring_outputs(csv_path)
        assert result["verdict"] == VERDICT_INVALID

    def test_metadata_mismatch_total_scored(self, tmp_path):
        csv = self._make_valid_csv(tmp_path)
        meta_path = tmp_path / "meta.json"
        meta_path.write_text(
            json.dumps(
                {
                    "algorithm": "logistic_regression",
                    "source_run": "preprocessed_run_26",
                    "total_scored": 999,
                }
            )
        )
        result = validate_scoring_outputs(csv, meta_path)
        assert result["verdict"] == VERDICT_INVALID
        assert any("total_scored" in e for e in result["errors"])

    def test_valid_with_metadata(self, tmp_path):
        csv = self._make_valid_csv(tmp_path)
        meta_path = tmp_path / "meta.json"
        meta_path.write_text(
            json.dumps(
                {
                    "algorithm": "logistic_regression",
                    "source_run": "preprocessed_run_26",
                    "total_scored": 3,
                }
            )
        )
        result = validate_scoring_outputs(csv, meta_path)
        assert result["verdict"] == VERDICT_READY

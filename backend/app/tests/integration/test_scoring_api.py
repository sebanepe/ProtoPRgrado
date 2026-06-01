"""Tests de integración para los endpoints de scoring por lotes D1."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from backend.app.models.models import BatchScoringRun, ModelRegistry


# ---------------------------------------------------------------------------
# Helpers (duplicados de unit tests para independencia de fixture)
# ---------------------------------------------------------------------------

def _write_dataset(tmp_path: Path, token: str) -> Path:
    csv_path = tmp_path / f"supervised_human_alert_dataset_run_{token}.csv"
    df = pd.DataFrame(
        [
            {
                "source_run": f"preprocessed_run_{token}",
                "summary_alert_id": f"alert_{i}",
                "representative_transaction_id": f"tx_{i}",
                "customer_hash": f"hash_{i}",
                "rule_code": "RULE_VELOCITY_CARD_DAY",
                "rule_name": "Card Velocity Day",
                "risk_level": "HIGH",
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


def _write_pkl(tmp_path: Path, token: str, algorithm: str) -> Path:
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
    clf = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000, random_state=42))])
    clf.fit(X_dummy, y_dummy)
    pkl_path = tmp_path / f"supervised_human_{algorithm}_run_{token}.pkl"
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


def _insert_completed_run(db_session, source_run: str, algorithm: str, results_file: str, report_file: str) -> BatchScoringRun:
    run = BatchScoringRun(
        source_run=source_run,
        run_token=source_run.split("_")[-1],
        algorithm=algorithm,
        model_family="SUPERVISED_HUMAN",
        status="COMPLETED",
        total_scored=4,
        high_count=1,
        medium_count=2,
        low_count=1,
        results_file=results_file,
        report_file=report_file,
    )
    db_session.add(run)
    db_session.commit()
    db_session.refresh(run)
    return run


# ---------------------------------------------------------------------------
# POST /api/scoring/batch-run
# ---------------------------------------------------------------------------

class TestBatchRunEndpoint:
    def test_blocked_no_dataset(self, test_client, tmp_path, monkeypatch):
        monkeypatch.setenv("PROJECT_PROCESSED_DIR", str(tmp_path))
        r = test_client.post(
            "/api/scoring/batch-run",
            json={"source_run": "preprocessed_run_9999", "algorithm": "logistic_regression"},
        )
        assert r.status_code == 409
        detail = r.json()["detail"]
        assert detail.get("status") == "BLOCKED"

    def test_invalid_algorithm_returns_422(self, test_client):
        r = test_client.post(
            "/api/scoring/batch-run",
            json={"source_run": "preprocessed_run_26", "algorithm": "neural_network"},
        )
        assert r.status_code == 422

    def test_completed_with_mock_scoring(self, test_client, db_session, tmp_path, monkeypatch):
        monkeypatch.setenv("PROJECT_PROCESSED_DIR", str(tmp_path))
        token = "200"
        source_run = f"preprocessed_run_{token}"
        dataset_path = _write_dataset(tmp_path, token)
        pkl_path = _write_pkl(tmp_path, token, "logistic_regression")
        _register_model(db_session, source_run, "logistic_regression", pkl_path)

        r = test_client.post(
            "/api/scoring/batch-run",
            json={
                "source_run": source_run,
                "algorithm": "logistic_regression",
                "input_dataset_path": str(dataset_path),
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "COMPLETED"
        assert body["total_scored"] == 4
        assert body["high_count"] + body["medium_count"] + body["low_count"] == 4
        assert "warnings" in body


# ---------------------------------------------------------------------------
# GET /api/scoring/runs
# ---------------------------------------------------------------------------

class TestListRunsEndpoint:
    def test_empty_list(self, test_client):
        r = test_client.get("/api/scoring/runs?source_run=preprocessed_run_NOPE")
        assert r.status_code == 200
        body = r.json()
        assert body["count"] == 0
        assert body["items"] == []

    def test_list_with_data(self, test_client, db_session, tmp_path):
        source_run = "preprocessed_run_300"
        run = BatchScoringRun(
            source_run=source_run,
            run_token="300",
            algorithm="random_forest",
            model_family="SUPERVISED_HUMAN",
            status="COMPLETED",
            total_scored=10,
        )
        db_session.add(run)
        db_session.commit()

        r = test_client.get(f"/api/scoring/runs?source_run={source_run}")
        assert r.status_code == 200
        body = r.json()
        assert body["count"] >= 1
        item = body["items"][0]
        assert "id" in item
        assert item["source_run"] == source_run
        assert item["algorithm"] == "random_forest"

    def test_filter_by_algorithm(self, test_client, db_session):
        source_run = "preprocessed_run_301"
        for algo in ("logistic_regression", "random_forest"):
            db_session.add(
                BatchScoringRun(
                    source_run=source_run,
                    run_token="301",
                    algorithm=algo,
                    model_family="SUPERVISED_HUMAN",
                    status="COMPLETED",
                )
            )
        db_session.commit()

        r = test_client.get(f"/api/scoring/runs?source_run={source_run}&algorithm=logistic_regression")
        assert r.status_code == 200
        body = r.json()
        assert all(i["algorithm"] == "logistic_regression" for i in body["items"])


# ---------------------------------------------------------------------------
# GET /api/scoring/runs/{run_id}
# ---------------------------------------------------------------------------

class TestGetRunEndpoint:
    def test_not_found(self, test_client):
        r = test_client.get("/api/scoring/runs/999999")
        assert r.status_code == 404

    def test_found(self, test_client, db_session):
        run = BatchScoringRun(
            source_run="preprocessed_run_400",
            run_token="400",
            algorithm="gradient_boosting",
            model_family="SUPERVISED_HUMAN",
            status="COMPLETED",
            total_scored=5,
        )
        db_session.add(run)
        db_session.commit()
        db_session.refresh(run)

        r = test_client.get(f"/api/scoring/runs/{run.id}")
        assert r.status_code == 200
        body = r.json()
        assert body["id"] == run.id
        assert body["algorithm"] == "gradient_boosting"


# ---------------------------------------------------------------------------
# GET /api/scoring/results
# ---------------------------------------------------------------------------

class TestGetResultsEndpoint:
    def test_not_found(self, test_client):
        r = test_client.get("/api/scoring/results?source_run=preprocessed_run_NOPE&algorithm=logistic_regression")
        assert r.status_code == 404

    def test_results_success(self, test_client, db_session, tmp_path):
        source_run = "preprocessed_run_500"
        results_path = tmp_path / "scoring_results.csv"
        df = pd.DataFrame(
            [
                {
                    "source_run": source_run,
                    "summary_alert_id": f"a{i}",
                    "customer_hash": f"h{i}",
                    "ml_risk_score": 0.3 + i * 0.2,
                    "ml_risk_level": ["LOW", "MEDIUM", "HIGH"][i % 3],
                    "algorithm": "logistic_regression",
                    "scored_at": "2024-01-01T00:00:00",
                }
                for i in range(3)
            ]
        )
        df.to_csv(results_path, index=False)

        run = BatchScoringRun(
            source_run=source_run,
            run_token="500",
            algorithm="logistic_regression",
            model_family="SUPERVISED_HUMAN",
            status="COMPLETED",
            total_scored=3,
            results_file=str(results_path),
        )
        db_session.add(run)
        db_session.commit()

        r = test_client.get(
            f"/api/scoring/results?source_run={source_run}&algorithm=logistic_regression&page=1&page_size=10"
        )
        assert r.status_code == 200
        body = r.json()
        assert body["total"] == 3
        assert len(body["rows"]) == 3
        for row in body["rows"]:
            assert "is_fraud" not in row
            assert "confirmed_fraud" not in row

    def test_results_pagination(self, test_client, db_session, tmp_path):
        source_run = "preprocessed_run_501"
        results_path = tmp_path / "scoring_results_501.csv"
        df = pd.DataFrame(
            [
                {
                    "source_run": source_run,
                    "summary_alert_id": f"a{i}",
                    "customer_hash": f"h{i}",
                    "ml_risk_score": 0.5,
                    "ml_risk_level": "MEDIUM",
                    "algorithm": "random_forest",
                    "scored_at": "2024-01-01T00:00:00",
                }
                for i in range(10)
            ]
        )
        df.to_csv(results_path, index=False)
        db_session.add(
            BatchScoringRun(
                source_run=source_run,
                run_token="501",
                algorithm="random_forest",
                model_family="SUPERVISED_HUMAN",
                status="COMPLETED",
                total_scored=10,
                results_file=str(results_path),
            )
        )
        db_session.commit()

        r = test_client.get(
            f"/api/scoring/results?source_run={source_run}&algorithm=random_forest&page=1&page_size=3"
        )
        assert r.status_code == 200
        body = r.json()
        assert body["total"] == 10
        assert len(body["rows"]) == 3
        assert body["total_pages"] == 4

    def test_results_risk_level_filter(self, test_client, db_session, tmp_path):
        source_run = "preprocessed_run_502"
        results_path = tmp_path / "scoring_results_502.csv"
        df = pd.DataFrame(
            [
                {
                    "source_run": source_run,
                    "summary_alert_id": f"a{i}",
                    "customer_hash": f"h{i}",
                    "ml_risk_score": 0.3 + i * 0.25,
                    "ml_risk_level": "HIGH" if i >= 2 else "LOW",
                    "algorithm": "gradient_boosting",
                    "scored_at": "2024-01-01T00:00:00",
                }
                for i in range(4)
            ]
        )
        df.to_csv(results_path, index=False)
        db_session.add(
            BatchScoringRun(
                source_run=source_run,
                run_token="502",
                algorithm="gradient_boosting",
                model_family="SUPERVISED_HUMAN",
                status="COMPLETED",
                total_scored=4,
                results_file=str(results_path),
            )
        )
        db_session.commit()

        r = test_client.get(
            f"/api/scoring/results?source_run={source_run}&algorithm=gradient_boosting&ml_risk_level=HIGH"
        )
        assert r.status_code == 200
        body = r.json()
        assert all(row["ml_risk_level"] == "HIGH" for row in body["rows"])

    def test_results_no_forbidden_columns_in_response(self, test_client, db_session, tmp_path):
        source_run = "preprocessed_run_503"
        results_path = tmp_path / "scoring_results_503.csv"
        df = pd.DataFrame(
            [
                {
                    "source_run": source_run,
                    "summary_alert_id": "a1",
                    "customer_hash": "h1",
                    "ml_risk_score": 0.5,
                    "ml_risk_level": "MEDIUM",
                    "algorithm": "logistic_regression",
                    "scored_at": "2024-01-01",
                    "is_fraud": 1,
                    "confirmed_fraud": 0,
                }
            ]
        )
        df.to_csv(results_path, index=False)
        db_session.add(
            BatchScoringRun(
                source_run=source_run,
                run_token="503",
                algorithm="logistic_regression",
                model_family="SUPERVISED_HUMAN",
                status="COMPLETED",
                total_scored=1,
                results_file=str(results_path),
            )
        )
        db_session.commit()

        r = test_client.get(
            f"/api/scoring/results?source_run={source_run}&algorithm=logistic_regression"
        )
        assert r.status_code == 200
        body = r.json()
        assert len(body["rows"]) == 1
        row = body["rows"][0]
        assert "is_fraud" not in row
        assert "confirmed_fraud" not in row


# ---------------------------------------------------------------------------
# GET /api/scoring/report
# ---------------------------------------------------------------------------

class TestGetReportEndpoint:
    def test_not_found(self, test_client):
        r = test_client.get("/api/scoring/report?source_run=preprocessed_run_NOPE&algorithm=random_forest")
        assert r.status_code == 404

    def test_report_success(self, test_client, db_session, tmp_path):
        source_run = "preprocessed_run_600"
        report_path = tmp_path / "scoring_report_600.md"
        report_path.write_text("# Reporte D1\n\nContenido de prueba.\n", encoding="utf-8")

        db_session.add(
            BatchScoringRun(
                source_run=source_run,
                run_token="600",
                algorithm="random_forest",
                model_family="SUPERVISED_HUMAN",
                status="COMPLETED",
                total_scored=5,
                report_file=str(report_path),
            )
        )
        db_session.commit()

        r = test_client.get(f"/api/scoring/report?source_run={source_run}&algorithm=random_forest")
        assert r.status_code == 200
        body = r.json()
        assert "markdown" in body
        assert "Reporte" in body["markdown"]

import json

import pandas as pd

from backend.app.ml.model_evaluation_service import build_model_evaluation_comparison


def _seed_inputs(base, include_autoencoder=True):
    token = "26"
    pd.DataFrame(
        [
            {
                "summary_alert_id": "26-S-1",
                "source_run": "26",
                "customer_hash": "c1",
                "rule_code": "R1",
                "rule_name": "rule",
                "risk_level": "HIGH",
                "max_risk_score": 80,
                "count_transactions": 2,
                "countries_detected": "PY|CL",
                "merchant_rubro_proxy": "5814",
                "representative_transaction_id": "tx1",
                "child_transaction_ids": "tx1|tx2",
            }
        ]
    ).to_csv(base / f"alerts_summary_run_{token}.csv", index=False)
    pd.DataFrame(
        [{"transaction_id": "tx1", "customer_hash": "c1", "risk_score": 80, "rule_code": "R1", "merchant_rubro_proxy": "5814"}]
    ).to_csv(base / f"alerts_run_{token}.csv", index=False)
    pd.DataFrame(
        [{"transaction_id": "tx1", "anomaly_score": 0.9, "anomaly_flag": 1, "anomaly_rank": 1}]
    ).to_csv(base / f"anomaly_scores_run_{token}.csv", index=False)
    if include_autoencoder:
        pd.DataFrame(
            [{"transaction_id": "tx2", "reconstruction_error": 12.0, "autoencoder_anomaly_score": 1.0, "autoencoder_anomaly_flag": 1, "anomaly_rank": 1}]
        ).to_csv(base / f"autoencoder_scores_run_{token}.csv", index=False)
    for model in ("logistic_regression", "random_forest", "gradient_boosting"):
        pd.DataFrame(
            [{"source_run": "preprocessed_run_26", "summary_alert_id": "26-S-1", "y_true": 1, "y_pred": 1, "y_proba": 0.99, "evaluation_result": "TRUE_POSITIVE"}]
        ).to_csv(base / f"supervised_human_{model}_predictions_run_{token}.csv", index=False)


def test_build_model_evaluation_generates_outputs(db_session, tmp_path, monkeypatch):
    monkeypatch.setenv("PROJECT_PROCESSED_DIR", str(tmp_path))
    monkeypatch.setenv("PROJECT_MODELS_DIR", str(tmp_path))
    _seed_inputs(tmp_path, include_autoencoder=True)
    result = build_model_evaluation_comparison(db_session, "preprocessed_run_26")
    assert (tmp_path / "model_comparison_alert_level_run_26.csv").exists()
    assert (tmp_path / "model_comparison_transaction_level_run_26.csv").exists()
    assert (tmp_path / "model_evaluation_comparison_report_run_26.md").exists()
    assert "autoencoder" in result["available_methods"]
    df = pd.read_csv(tmp_path / "model_comparison_alert_level_run_26.csv")
    assert "comparison_priority_score" in df.columns
    assert "is_fraud" not in df.columns


def test_build_model_evaluation_handles_missing_autoencoder(db_session, tmp_path, monkeypatch):
    monkeypatch.setenv("PROJECT_PROCESSED_DIR", str(tmp_path))
    monkeypatch.setenv("PROJECT_MODELS_DIR", str(tmp_path))
    _seed_inputs(tmp_path, include_autoencoder=False)
    result = build_model_evaluation_comparison(db_session, "preprocessed_run_26")
    assert "autoencoder" in result["missing_methods"]
    payload = json.loads((tmp_path / "model_evaluation_comparison_metadata_run_26.json").read_text(encoding="utf-8"))
    assert payload["missing_methods"]

import pandas as pd


def _seed_inputs(base):
    token = "26"

    pd.DataFrame(
        [
            {
                "summary_alert_id": "26-S-1",
                "source_run": "preprocessed_run_26",
                "customer_hash": "c1",
                "rule_code": "R1",
                "rule_name": "Regla de prueba",
                "risk_level": "HIGH",
                "max_score": 80,
                "transactions_detected": 2,
                "countries_detected": "BO|US",
                "merchant_rubro_proxy": "5411",
                "representative_transaction_id": "tx1",
                "child_transaction_ids": "tx1|tx2",
            }
        ]
    ).to_csv(base / f"alerts_summary_run_{token}.csv", index=False)

    pd.DataFrame(
        [
            {
                "source_run": "preprocessed_run_26",
                "summary_alert_id": "26-S-1",
                "transaction_id": "tx1",
                "customer_hash": "c1",
                "risk_score": 80,
                "rule_code": "R1",
            }
        ]
    ).to_csv(base / f"alerts_run_{token}.csv", index=False)

    pd.DataFrame(
        [
            {
                "source_run": "preprocessed_run_26",
                "transaction_id": "tx1",
                "customer_hash": "c1",
                "anomaly_score": 0.9,
                "anomaly_flag": 1,
                "anomaly_rank": 1,
            }
        ]
    ).to_csv(base / f"anomaly_scores_run_{token}.csv", index=False)

    pd.DataFrame(
        [
            {
                "source_run": "preprocessed_run_26",
                "summary_alert_id": "26-S-1",
                "y_true": 1,
                "y_pred": 1,
                "y_proba": 0.99,
                "prediction_label": "PREDICTED_CONFIRMED_FRAUD",
                "evaluation_result": "TRUE_POSITIVE",
            }
        ]
    ).to_csv(
        base / "supervised_human_logistic_regression_predictions_run_26.csv",
        index=False,
    )


def test_model_evaluation_routes(test_client, tmp_path, monkeypatch):
    monkeypatch.setenv("PROJECT_PROCESSED_DIR", str(tmp_path))
    monkeypatch.setenv("PROJECT_MODELS_DIR", str(tmp_path))

    _seed_inputs(tmp_path)

    build = test_client.post(
        "/api/model-evaluation/build-comparison",
        json={"source_run": "preprocessed_run_26"},
    )
    assert build.status_code == 200, build.text

    summary = test_client.get(
        "/api/model-evaluation/summary",
        params={"source_run": "preprocessed_run_26"},
    )
    assert summary.status_code == 200, summary.text

    top = test_client.get(
        "/api/model-evaluation/top-cases",
        params={"source_run": "preprocessed_run_26", "limit": 20},
    )
    assert top.status_code == 200, top.text
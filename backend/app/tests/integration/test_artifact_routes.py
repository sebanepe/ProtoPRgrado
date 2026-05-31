from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _write_route_artifacts(base: Path, models: Path, token: str = "26") -> dict[str, int]:
    base.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"transaction_id": "tx1"}, {"transaction_id": "tx2"}]).to_csv(base / f"preprocessed_run_{token}.csv", index=False)
    (base / f"preprocessing_report_run_{token}.md").write_text("# preprocessing\n", encoding="utf-8")
    pd.DataFrame([{"alert_id": "a1"}, {"alert_id": "a2"}]).to_csv(base / f"alerts_run_{token}.csv", index=False)
    pd.DataFrame([{"summary_alert_id": "s1"}]).to_csv(base / f"alerts_summary_run_{token}.csv", index=False)
    (base / f"rules_report_run_{token}.md").write_text("# rules\n", encoding="utf-8")
    pd.DataFrame([{"transaction_id": "tx1", "anomaly_flag": 1}]).to_csv(base / f"anomaly_scores_run_{token}.csv", index=False)
    pd.DataFrame([{"transaction_id": "tx1", "feature": 1}]).to_csv(base / f"unsupervised_feature_set_run_{token}.csv", index=False)
    (base / f"anomaly_report_run_{token}.md").write_text("# anomaly\n", encoding="utf-8")
    (models / f"isolation_forest_run_{token}.pkl").write_bytes(b"not-a-real-model")
    (models / f"isolation_forest_run_{token}_metadata.json").write_text(
        json.dumps({"algorithm": "isolation_forest", "source_run": f"preprocessed_run_{token}"}),
        encoding="utf-8",
    )
    return {
        "alerts": (base / f"alerts_run_{token}.csv").stat().st_mtime_ns,
        "summary": (base / f"alerts_summary_run_{token}.csv").stat().st_mtime_ns,
        "scores": (base / f"anomaly_scores_run_{token}.csv").stat().st_mtime_ns,
    }


def test_artifact_routes_register_and_traceability(test_client, tmp_path, monkeypatch):
    processed = tmp_path / "processed"
    models = tmp_path / "models"
    before = _write_route_artifacts(processed, models)
    monkeypatch.setenv("PROJECT_PROCESSED_DIR", str(processed))
    monkeypatch.setenv("PROJECT_MODELS_DIR", str(models))

    response = test_client.post("/api/artifacts/register-existing", json={"source_run": "preprocessed_run_26"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["registered_count"] >= 10
    registered_types = {item["artifact_type"] for item in payload["registered"]}
    assert "RULE_ALERTS_CSV" in registered_types
    assert "RULE_SUMMARY_CSV" in registered_types
    assert "ANOMALY_SCORES_CSV" in registered_types
    assert payload["rule_run"]["detailed_alert_count"] == 2
    assert payload["rule_run"]["grouped_alert_count"] == 1
    assert payload["model_registry"]["model_family"] == "UNSUPERVISED"

    list_response = test_client.get("/api/artifacts", params={"source_run": "preprocessed_run_26"})
    assert list_response.status_code == 200
    assert list_response.json()["count"] >= 10

    rule_response = test_client.get("/api/artifacts/rule-runs", params={"source_run": "preprocessed_run_26"})
    assert rule_response.status_code == 200
    assert rule_response.json()["items"][0]["summary_file"].endswith("alerts_summary_run_26.csv")

    model_response = test_client.get("/api/artifacts/model-registry", params={"source_run": "preprocessed_run_26"})
    assert model_response.status_code == 200
    assert model_response.json()["items"][0]["algorithm"] == "isolation_forest"

    trace_response = test_client.get("/api/artifacts/traceability", params={"source_run": "preprocessed_run_26"})
    assert trace_response.status_code == 200
    trace = trace_response.json()
    assert trace["phase_a"]["preprocessed_csv"]["file_name"] == "preprocessed_run_26.csv"
    assert trace["phase_b"]["alerts_csv"]["file_name"] == "alerts_run_26.csv"
    assert trace["phase_b"]["summary_csv"]["file_name"] == "alerts_summary_run_26.csv"
    assert trace["phase_c3"]["scores"]["file_name"] == "anomaly_scores_run_26.csv"
    assert trace["phase_c4"]["supervised_dataset_runs"] == []
    assert "is_fraud" not in json.dumps(trace)
    assert "confirmed_fraud" not in json.dumps(trace).lower()

    assert (processed / "alerts_run_26.csv").stat().st_mtime_ns == before["alerts"]
    assert (processed / "alerts_summary_run_26.csv").stat().st_mtime_ns == before["summary"]
    assert (processed / "anomaly_scores_run_26.csv").stat().st_mtime_ns == before["scores"]


def test_get_artifacts_empty_response_is_controlled(test_client):
    response = test_client.get("/api/artifacts", params={"source_run": "preprocessed_run_missing"})

    assert response.status_code == 200
    assert response.json() == {"count": 0, "items": []}

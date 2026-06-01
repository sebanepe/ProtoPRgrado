import json

import pandas as pd

from backend.app.ml.validate_model_evaluation_comparison import (
    PARTIAL,
    READY,
    validate_model_evaluation_comparison,
)


def test_validator_ready(tmp_path):
    alert = tmp_path / "a.csv"
    tx = tmp_path / "t.csv"
    meta = tmp_path / "m.json"
    pd.DataFrame([{"source_run": "preprocessed_run_26", "summary_alert_id": "26-S-1"}]).to_csv(alert, index=False)
    pd.DataFrame([{"transaction_id": "tx1"}]).to_csv(tx, index=False)
    meta.write_text(json.dumps({"available_methods": ["rules"], "missing_methods": [], "warnings": []}), encoding="utf-8")
    result = validate_model_evaluation_comparison(alert, tx, meta)
    assert result["verdict"] == READY


def test_validator_partial_when_autoencoder_missing(tmp_path):
    alert = tmp_path / "a.csv"
    tx = tmp_path / "t.csv"
    meta = tmp_path / "m.json"
    pd.DataFrame([{"source_run": "preprocessed_run_26", "summary_alert_id": "26-S-1"}]).to_csv(alert, index=False)
    pd.DataFrame([{"transaction_id": "tx1"}]).to_csv(tx, index=False)
    meta.write_text(json.dumps({"available_methods": ["rules"], "missing_methods": ["autoencoder"], "warnings": []}), encoding="utf-8")
    result = validate_model_evaluation_comparison(alert, tx, meta)
    assert result["verdict"] == PARTIAL

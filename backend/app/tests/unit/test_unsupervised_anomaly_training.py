import json

import pandas as pd

from backend.app.ml.train_unsupervised_anomaly import train_unsupervised_anomaly
from backend.app.ml.validate_anomaly_outputs import validate_anomaly_outputs


def test_train_unsupervised_anomaly_creates_artifacts_and_validates(tmp_path):
    source = tmp_path / "preprocessed_run_2.csv"
    rows = []
    for index in range(20):
        rows.append(
            {
                "transaction_id": f"tx_{index}",
                "amount": 25 + index,
                "customer_hash": "cust_a" if index < 10 else "cust_b",
                "merchant_hash": f"merch_{index % 3}",
                "merchant_rubro_proxy": "5411" if index < 19 else "7995",
                "country_code": "BO" if index % 2 == 0 else "US",
                "pos_entry_mode": 7 if index < 18 else 81,
                "has_pinblock": 1 if index % 3 else 0,
                "card_presence_type": "TP" if index < 18 else "TNP",
                "transaction_datetime": f"2026-05-01T{index:02d}:00:00Z",
            }
        )
    rows[-1]["amount"] = 100000
    pd.DataFrame(rows).to_csv(source, index=False)

    result = train_unsupervised_anomaly(
        input_path=str(source),
        source_run="preprocessed_run_2",
        model="isolation_forest",
        contamination=0.05,
        output_dir=tmp_path,
        models_dir=tmp_path / "models",
    )

    score_frame = pd.read_csv(result["score_file"])
    assert not score_frame.empty
    assert set(score_frame["anomaly_flag"].unique().tolist()).issubset({0, 1})
    assert "is_fraud" not in score_frame.columns
    assert "confirmed_fraud" not in score_frame.columns
    assert score_frame["transaction_id"].is_unique
    assert score_frame["anomaly_score"].notna().all()
    assert score_frame.iloc[0]["anomaly_rank"] == 1
    assert score_frame.iloc[0]["anomaly_flag"] == 1

    with open(result["report_file"], "r", encoding="utf-8") as handle:
        report_text = handle.read()
    assert "Las anomalías detectadas no representan fraude confirmado." in report_text
    assert "No se generó is_fraud. No se generó confirmed_fraud. No se usaron reglas como etiquetas." in report_text

    validation = validate_anomaly_outputs(
        score_file=result["score_file"],
        feature_file=result["feature_file"],
        metadata_file=result["metadata_file"],
        contamination=0.05,
    )
    assert validation["verdict"] == "ANOMALY_OUTPUTS_READY"

    with open(result["metadata_file"], "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    assert metadata["model_type"] == "unsupervised_anomaly_detection"
    assert metadata["anomaly_count"] >= 1

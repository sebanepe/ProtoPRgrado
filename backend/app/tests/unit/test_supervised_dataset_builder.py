from __future__ import annotations

from pathlib import Path

import pandas as pd

from backend.app.ml.supervised_dataset_builder import build_human_supervised_alert_dataset
from backend.app.ml.validate_human_supervised_dataset import validate_human_supervised_dataset
from backend.app.models.models import ArtifactRegistry, RuleAlertReview, RuleRun, SupervisedDatasetRun
from backend.app.services import artifact_registry_service as artifacts


def _summary(path: Path, token: str = "901") -> Path:
    path.mkdir(parents=True, exist_ok=True)
    summary = path / f"alerts_summary_run_{token}.csv"
    pd.DataFrame(
        [
            {
                "summary_alert_id": f"{token}-S-positive",
                "source_run": token,
                "customer_hash": "cust1",
                "rule_code": "RULE_DOUBLE_COUNTRY_CARD_PRESENT_SAME_DAY",
                "rule_name": "Double Country Card Present Same Day",
                "risk_level": "HIGH",
                "max_risk_score": 85,
                "count_transactions": 2,
                "countries_detected": "BO|PE",
                "merchant_rubro_proxy": "5814",
                "merchant_rubro_values": "5814|7011",
                "window_start": "2026-01-01T00:00:00Z",
                "window_end": "2026-01-01T01:00:00Z",
                "representative_transaction_id": "tx1",
                "status": "NEW",
                "PAN_TARJETA": "should-not-leak",
            },
            {
                "summary_alert_id": f"{token}-S-negative",
                "source_run": token,
                "customer_hash": "cust2",
                "rule_code": "RULE_MCC_RISK",
                "rule_name": "MCC Risk",
                "risk_level": "MEDIUM",
                "max_risk_score": 60,
                "count_transactions": 1,
                "countries_detected": "BO",
                "merchant_rubro_proxy": "7995",
                "merchant_rubro_values": "7995",
                "window_start": "2026-01-02T00:00:00Z",
                "window_end": "2026-01-02T00:10:00Z",
                "representative_transaction_id": "tx2",
                "status": "NEW",
            },
        ]
    ).to_csv(summary, index=False)
    pd.DataFrame([{"alert_id": "a1"}]).to_csv(path / f"alerts_run_{token}.csv", index=False)
    pd.DataFrame([{"transaction_id": "tx"}]).to_csv(path / f"preprocessed_run_{token}.csv", index=False)
    return summary


def _review(db_session, token: str, summary_id: str, status: str) -> None:
    db_session.add(
        RuleAlertReview(
            source_run=f"preprocessed_run_{token}",
            summary_alert_id=summary_id,
            rule_code="RULE_TEST",
            new_status=status,
            analyst_notes=f"note {status}",
        )
    )
    db_session.commit()


def test_no_usable_labels_returns_controlled_status(db_session, tmp_path):
    _summary(tmp_path, "901")
    _review(db_session, "901", "901-S-positive", "FALSE_POSITIVE")
    _review(db_session, "901", "901-S-negative", "IN_REVIEW")

    result = build_human_supervised_alert_dataset("preprocessed_run_901", db=db_session, output_dir=tmp_path)

    assert result["verdict"] == "DATASET_NOT_CREATED_INSUFFICIENT_HUMAN_LABELS"
    assert result["usable_total_labels"] == 0
    assert not (tmp_path / "supervised_human_alert_dataset_run_901.csv").exists()
    run = db_session.query(SupervisedDatasetRun).filter_by(source_run="preprocessed_run_901").first()
    assert run.status == "NOT_CREATED_INSUFFICIENT_HUMAN_LABELS"


def test_builder_maps_human_labels_and_excludes_unusable_statuses(db_session, tmp_path):
    _summary(tmp_path, "902")
    _review(db_session, "902", "902-S-positive", "CONFIRMED_FRAUD")
    _review(db_session, "902", "902-S-negative", "DISMISSED")
    _review(db_session, "902", "902-S-new", "NEW")
    _review(db_session, "902", "902-S-false-positive", "FALSE_POSITIVE")

    result = build_human_supervised_alert_dataset("preprocessed_run_902", db=db_session, output_dir=tmp_path)
    df = pd.read_csv(tmp_path / "supervised_human_alert_dataset_run_902.csv")

    assert result["verdict"] == "HUMAN_SUPERVISED_DATASET_CREATED"
    assert dict(zip(df["summary_alert_id"], df["target_human_label"])) == {"902-S-positive": 1, "902-S-negative": 0}
    assert set(df["human_review_status"]) == {"CONFIRMED_FRAUD", "DISMISSED"}
    assert "is_fraud" not in df.columns
    assert "confirmed_fraud" not in df.columns
    assert "PAN_TARJETA" not in df.columns
    assert "TARJETA" not in df.columns
    assert "rule_code" in df.columns
    assert "target_label_source" in df.columns
    assert set(df["source_run"]) == {"preprocessed_run_902"}
    assert validate_human_supervised_dataset(tmp_path / "supervised_human_alert_dataset_run_902.csv")["verdict"] == (
        "HUMAN_SUPERVISED_DATASET_READY"
    )


def test_builder_uses_artifact_registry_and_registers_c4_outputs(db_session, tmp_path):
    summary = _summary(tmp_path, "903")
    alerts = tmp_path / "alerts_run_903.csv"
    _review(db_session, "903", "903-S-positive", "CONFIRMED_FRAUD")
    _review(db_session, "903", "903-S-negative", "DISMISSED")
    artifacts.register_or_update_artifact(
        db_session,
        artifact_type=artifacts.ARTIFACT_RULE_SUMMARY_CSV,
        phase=artifacts.PHASE_B,
        source_run="preprocessed_run_903",
        file_path=summary,
    )
    artifacts.register_or_update_artifact(
        db_session,
        artifact_type=artifacts.ARTIFACT_RULE_ALERTS_CSV,
        phase=artifacts.PHASE_B,
        source_run="preprocessed_run_903",
        file_path=alerts,
    )

    result = build_human_supervised_alert_dataset("preprocessed_run_903", db=db_session, output_dir=tmp_path)

    assert result["warnings"] == []
    assert db_session.query(ArtifactRegistry).filter_by(artifact_type=artifacts.ARTIFACT_SUPERVISED_DATASET).count() == 1
    assert db_session.query(ArtifactRegistry).filter_by(artifact_type=artifacts.ARTIFACT_SUPERVISED_REPORT).count() == 1
    assert db_session.query(SupervisedDatasetRun).filter_by(source_run="preprocessed_run_903").count() == 1


def test_builder_fallback_does_not_modify_source_files(db_session, tmp_path):
    summary = _summary(tmp_path, "904")
    alerts = tmp_path / "alerts_run_904.csv"
    preprocessed = tmp_path / "preprocessed_run_904.csv"
    before = (summary.stat().st_mtime_ns, alerts.stat().st_mtime_ns, preprocessed.stat().st_mtime_ns)
    _review(db_session, "904", "904-S-positive", "CONFIRMED_FRAUD")
    _review(db_session, "904", "904-S-negative", "DISMISSED")

    result = build_human_supervised_alert_dataset("preprocessed_run_904", db=db_session, output_dir=tmp_path)

    assert "fallback" in " ".join(result["warnings"]).lower()
    assert (summary.stat().st_mtime_ns, alerts.stat().st_mtime_ns, preprocessed.stat().st_mtime_ns) == before


def test_builder_uses_rule_runs_when_artifact_registry_is_empty(db_session, tmp_path):
    summary = _summary(tmp_path, "905")
    alerts = tmp_path / "alerts_run_905.csv"
    db_session.add(
        RuleRun(
            source_run="preprocessed_run_905",
            run_token="905",
            summary_file=str(summary),
            alerts_file=str(alerts),
            status="AVAILABLE",
        )
    )
    db_session.commit()
    _review(db_session, "905", "905-S-positive", "CONFIRMED_FRAUD")
    _review(db_session, "905", "905-S-negative", "DISMISSED")

    result = build_human_supervised_alert_dataset("preprocessed_run_905", db=db_session, output_dir=tmp_path)

    assert result["warnings"] == []
    assert result["verdict"] == "HUMAN_SUPERVISED_DATASET_CREATED"


def test_validator_reports_empty_or_insufficient_classes(tmp_path):
    assert validate_human_supervised_dataset(tmp_path / "missing.csv")["verdict"] == "FILE_NOT_FOUND"
    data = tmp_path / "one_class.csv"
    pd.DataFrame(
        [
            {
                "source_run": "preprocessed_run_1",
                "summary_alert_id": "s1",
                "human_review_status": "CONFIRMED_FRAUD",
                "target_human_label": 1,
                "target_label_source": "HUMAN_REVIEW",
                "target_label_meaning": "CONFIRMED_FRAUD",
            }
        ]
    ).to_csv(data, index=False)
    assert validate_human_supervised_dataset(data)["verdict"] == "HUMAN_SUPERVISED_DATASET_INSUFFICIENT_CLASSES"

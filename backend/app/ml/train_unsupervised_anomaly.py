from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import OneClassSVM

from backend.app.ml.unsupervised_feature_builder import (
    CATEGORICAL_FEATURE_COLUMNS,
    MODEL_INPUT_COLUMNS,
    NUMERIC_FEATURE_COLUMNS,
    build_unsupervised_features,
    default_models_dir,
    default_processed_dir,
    normalize_run_token,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _build_preprocessor(feature_frame: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    numeric_features = [column for column in NUMERIC_FEATURE_COLUMNS if column in feature_frame.columns]
    categorical_features = [column for column in CATEGORICAL_FEATURE_COLUMNS if column in feature_frame.columns]
    transformers = []
    if numeric_features:
        transformers.append(
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                numeric_features,
            )
        )
    if categorical_features:
        transformers.append(
            (
                "categorical",
                Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", _make_encoder())]),
                categorical_features,
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop"), numeric_features, categorical_features


def _build_model(model_name: str, contamination: float, n_rows: int):
    key = model_name.lower().strip()
    if key == "isolation_forest":
        return IsolationForest(n_estimators=200, contamination=contamination, random_state=42, n_jobs=-1), key
    if key == "local_outlier_factor":
        n_neighbors = min(20, max(2, n_rows - 1))
        return LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True, contamination=contamination), key
    if key == "one_class_svm":
        return OneClassSVM(kernel="rbf", gamma="scale", nu=max(min(contamination, 0.5), 0.001)), key
    raise ValueError(f"Unsupported anomaly model: {model_name}")


def _score_model(pipeline: Pipeline, feature_frame: pd.DataFrame) -> np.ndarray:
    try:
        return -pipeline.decision_function(feature_frame)
    except Exception:
        try:
            return -pipeline.score_samples(feature_frame)
        except Exception:
            predictions = pipeline.predict(feature_frame)
            return np.where(predictions == -1, 1.0, 0.0).astype(float)


def _comparison_section(score_frame: pd.DataFrame, source_run: str, output_dir: Path) -> Dict[str, Any]:
    run_token = normalize_run_token(source_run)
    alerts_path = output_dir / f"alerts_run_{run_token}.csv"
    alerts_summary_path = output_dir / f"alerts_summary_run_{run_token}.csv"
    anomalies = set(score_frame.loc[score_frame["anomaly_flag"] == 1, "transaction_id"].dropna().astype(str).tolist())
    result: Dict[str, Any] = {
        "alerts_file": str(alerts_path) if alerts_path.exists() else None,
        "alerts_summary_file": str(alerts_summary_path) if alerts_summary_path.exists() else None,
        "anomalies_with_rule_alert": 0,
        "anomalies_without_rule_alert": len(anomalies),
        "rule_alerts_without_anomaly": 0,
        "top_rules_among_anomalies": [],
    }
    if not alerts_path.exists():
        return result
    alerts_df = pd.read_csv(alerts_path)
    if "transaction_id" not in alerts_df.columns:
        return result
    alerts_df["transaction_id"] = alerts_df["transaction_id"].astype(str)
    alert_transactions = set(alerts_df["transaction_id"].dropna().tolist())
    shared = anomalies & alert_transactions
    result["anomalies_with_rule_alert"] = len(shared)
    result["anomalies_without_rule_alert"] = len(anomalies - alert_transactions)
    result["rule_alerts_without_anomaly"] = len(alert_transactions - anomalies)
    if shared and "rule_code" in alerts_df.columns:
        counts = alerts_df.loc[alerts_df["transaction_id"].isin(shared), "rule_code"].fillna("UNKNOWN").astype(str).value_counts().head(10)
        result["top_rules_among_anomalies"] = [{"rule_code": rule, "count": int(count)} for rule, count in counts.items()]
    return result


def _write_report(report_path: Path, feature_metadata: Dict[str, Any], score_frame: pd.DataFrame, model_name: str, contamination: float, comparison: Dict[str, Any]) -> None:
    def _simple_markdown_table(frame: pd.DataFrame) -> str:
        columns = list(frame.columns)
        header = "| " + " | ".join(columns) + " |\n"
        separator = "| " + " | ".join(["---"] * len(columns)) + " |\n"
        rows = []
        for _, row in frame.iterrows():
            values = [str(row[column]) for column in columns]
            rows.append("| " + " | ".join(values) + " |\n")
        return header + separator + "".join(rows)

    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("# Anomaly Detection Report\n\n")
        handle.write(f"- source_run: {feature_metadata.get('source_run')}\n")
        handle.write(f"- source_run_token: {feature_metadata.get('source_run_token')}\n")
        handle.write(f"- total_transactions: {len(score_frame)}\n")
        handle.write(f"- model: {model_name}\n")
        handle.write(f"- contamination: {contamination}\n")
        handle.write(f"- anomaly_count: {int(score_frame['anomaly_flag'].sum())}\n")
        handle.write(f"- anomaly_rate: {float(score_frame['anomaly_flag'].mean()) if len(score_frame) else 0.0:.6f}\n\n")
        handle.write("## Warnings\n")
        handle.write("Las anomalías detectadas no representan fraude confirmado.\n")
        handle.write("No se generó is_fraud. No se generó confirmed_fraud. No se usaron reglas como etiquetas.\n\n")
        handle.write("## Top 20 by anomaly score\n")
        handle.write(_simple_markdown_table(score_frame.head(20)[["transaction_id", "customer_hash", "anomaly_score", "anomaly_rank", "anomaly_flag"]]))
        handle.write("\n\n")
        handle.write("## Distributions among top anomalies\n")
        for column in ["country_code", "pos_entry_mode", "merchant_rubro_proxy"]:
            if column in score_frame.columns:
                handle.write(f"- {column}: {score_frame.head(20)[column].fillna('UNKNOWN').astype(str).value_counts().to_dict()}\n")
        handle.write(f"- hour_of_day: {score_frame.head(20)['transaction_datetime'].astype(str).str[11:13].replace({'': 'UNKNOWN'}).value_counts().to_dict()}\n")
        handle.write("\n## Comparison with Phase B\n")
        handle.write(f"- anomalies_with_rule_alert: {comparison.get('anomalies_with_rule_alert', 0)}\n")
        handle.write(f"- anomalies_without_rule_alert: {comparison.get('anomalies_without_rule_alert', 0)}\n")
        handle.write(f"- rule_alerts_without_anomaly: {comparison.get('rule_alerts_without_anomaly', 0)}\n")
        handle.write(f"- top_rules_among_anomalies: {comparison.get('top_rules_among_anomalies', [])}\n")


def train_unsupervised_anomaly(
    input_path: str,
    source_run: str,
    model: str = "isolation_forest",
    contamination: float = 0.01,
    output_dir: str | os.PathLike[str] | None = None,
    models_dir: str | os.PathLike[str] | None = None,
) -> Dict[str, Any]:
    if contamination not in {0.005, 0.01, 0.02, 0.05}:
        raise ValueError("contamination must be one of 0.005, 0.01, 0.02, 0.05")

    output_directory = Path(output_dir) if output_dir else default_processed_dir()
    models_directory = Path(models_dir) if models_dir else default_models_dir()
    output_directory.mkdir(parents=True, exist_ok=True)
    models_directory.mkdir(parents=True, exist_ok=True)

    feature_file, feature_metadata = build_unsupervised_features(input_path, source_run, output_directory)
    feature_frame = pd.read_csv(feature_file)
    if feature_frame.empty:
        raise ValueError("No rows available for unsupervised anomaly training")

    for column in MODEL_INPUT_COLUMNS:
        if column not in feature_frame.columns:
            feature_frame[column] = "UNKNOWN" if column in CATEGORICAL_FEATURE_COLUMNS else 0.0

    preprocessor, numeric_features, categorical_features = _build_preprocessor(feature_frame)
    base_model, model_name = _build_model(model, contamination, len(feature_frame))
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", base_model)])
    pipeline.fit(feature_frame[MODEL_INPUT_COLUMNS])

    scores = _score_model(pipeline, feature_frame[MODEL_INPUT_COLUMNS])
    score_series = pd.Series(scores, index=feature_frame.index, dtype=float)
    score_rank = score_series.rank(method="first", ascending=False).astype(int)
    n_rows = len(feature_frame)
    anomaly_limit = max(1, int(round(n_rows * contamination)))
    anomaly_flag = pd.Series(0, index=feature_frame.index, dtype=int)
    anomaly_flag.loc[score_series.sort_values(ascending=False).head(anomaly_limit).index] = 1
    anomaly_percentile = (1.0 - ((score_rank - 1) / max(1, n_rows - 1))) * 100.0

    run_token = normalize_run_token(source_run)
    run_value = int(run_token) if run_token.isdigit() else run_token
    anomaly_run_id = f"anomaly_run_{run_token}"
    created_at = _utc_now()

    score_frame = feature_frame.copy()
    score_frame["anomaly_run_id"] = anomaly_run_id
    score_frame["source_run"] = run_value
    score_frame["anomaly_model_name"] = model_name
    score_frame["anomaly_score"] = score_series.astype(float)
    score_frame["anomaly_rank"] = score_rank
    score_frame["anomaly_flag"] = anomaly_flag
    score_frame["anomaly_percentile"] = anomaly_percentile.astype(float)
    score_frame["created_at"] = created_at
    score_frame = score_frame.sort_values(["anomaly_score", "anomaly_rank"], ascending=[False, True]).reset_index(drop=True)
    score_frame = score_frame[[
        "anomaly_run_id",
        "source_run",
        "transaction_id",
        "customer_hash",
        "transaction_datetime",
        "amount",
        "country_code",
        "pos_entry_mode",
        "has_pinblock",
        "merchant_rubro_proxy",
        "anomaly_model_name",
        "anomaly_score",
        "anomaly_rank",
        "anomaly_flag",
        "anomaly_percentile",
        "created_at",
    ]]

    score_file = output_directory / f"anomaly_scores_run_{run_token}.csv"
    score_frame.to_csv(score_file, index=False)

    report_file = output_directory / f"anomaly_report_run_{run_token}.md"
    comparison = _comparison_section(score_frame, source_run, output_directory)
    _write_report(report_file, feature_metadata, score_frame, model_name, contamination, comparison)

    model_path = models_directory / f"{model_name}_run_{run_token}.pkl"
    joblib.dump(pipeline, model_path)

    metadata_path = models_directory / f"{model_name}_run_{run_token}_metadata.json"
    metadata = {
        **feature_metadata,
        "source_run": str(source_run),
        "source_run_token": run_value,
        "model_name": model_name,
        "model_type": "unsupervised_anomaly_detection",
        "algorithm": model_name,
        "contamination": contamination,
        "features_used": MODEL_INPUT_COLUMNS,
        "categorical_features": categorical_features,
        "numeric_features": numeric_features,
        "created_at": created_at,
        "total_rows": int(len(score_frame)),
        "anomaly_count": int(score_frame["anomaly_flag"].sum()),
        "anomaly_rate": float(score_frame["anomaly_flag"].mean()) if len(score_frame) else 0.0,
        "model_path": str(model_path),
        "score_file": str(score_file),
        "feature_file": str(feature_file),
        "report_file": str(report_file),
        "anomaly_run_id": anomaly_run_id,
        "comparison_with_phase_b": comparison,
    }
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)

    return {
        "feature_file": str(feature_file),
        "score_file": str(score_file),
        "report_file": str(report_file),
        "model_path": str(model_path),
        "metadata_file": str(metadata_path),
        "metadata": metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an unsupervised anomaly detection model")
    parser.add_argument("--input", required=True)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--model", default="isolation_forest", choices=["isolation_forest", "local_outlier_factor", "one_class_svm"])
    parser.add_argument("--contamination", type=float, default=0.01, choices=[0.005, 0.01, 0.02, 0.05])
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--models-dir", default=None)
    args = parser.parse_args()

    result = train_unsupervised_anomaly(
        input_path=args.input,
        source_run=args.source_run,
        model=args.model,
        contamination=args.contamination,
        output_dir=args.output_dir,
        models_dir=args.models_dir,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

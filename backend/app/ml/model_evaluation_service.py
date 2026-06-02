from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy.orm import Session

from backend.app.ml.batch_scoring_service import (
    _feature_frame_for_scoring,
    _load_model,
    _predict,
)
from backend.app.models.models import ModelRegistry, RuleAlertReview
from backend.app.services import artifact_registry_service as artifacts

PHASE_C5 = "PHASE_C5"
ARTIFACT_MODEL_EVALUATION_DATASET = "MODEL_EVALUATION_DATASET"
ARTIFACT_MODEL_EVALUATION_REPORT = "MODEL_EVALUATION_REPORT"
ARTIFACT_MODEL_EVALUATION_METADATA = "MODEL_EVALUATION_METADATA"
FORBIDDEN_COLUMNS = {"is_fraud", "confirmed_fraud", "PAN_TARJETA", "TARJETA", "pan_card", "raw_card"}
SUPERVISED_MODELS = ("logistic_regression", "random_forest", "gradient_boosting")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _drop_forbidden(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if c not in FORBIDDEN_COLUMNS]
    return df[cols].copy()


def _join_unique_values(values: Any) -> str:
    if isinstance(values, pd.Series):
        iterable = values.tolist()
    elif isinstance(values, (list, tuple, set)):
        iterable = list(values)
    else:
        iterable = [values]
    return "|".join(sorted({str(v) for v in iterable if pd.notna(v)}))


def _build_transaction_level_from_alerts(alerts_detail: pd.DataFrame) -> pd.DataFrame:
    if alerts_detail.empty or "transaction_id" not in alerts_detail.columns:
        return pd.DataFrame(columns=["transaction_id"])
    agg_spec: dict[str, tuple[str, str | Any]] = {}
    optional_first = ("customer_hash", "transaction_datetime", "amount", "country_code", "merchant_rubro_proxy")
    for col in optional_first:
        if col in alerts_detail.columns:
            agg_spec[col] = (col, "first")
    if "rule_code" in alerts_detail.columns:
        agg_spec["rule_codes"] = ("rule_code", _join_unique_values)
        agg_spec["rule_count"] = ("rule_code", "count")
    if "risk_score" in alerts_detail.columns:
        agg_spec["max_rule_score"] = ("risk_score", "max")
    tx_df = alerts_detail.groupby("transaction_id", as_index=False).agg(**agg_spec) if agg_spec else alerts_detail[["transaction_id"]].drop_duplicates()
    tx_df["has_rule_alert"] = True
    return tx_df


def _series_or_default(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce")
    return pd.Series([default] * len(df), index=df.index, dtype=float)


def _string_series_or_default(df: pd.DataFrame, column: str, default: str = "") -> pd.Series:
    if column in df.columns:
        return df[column].astype(str)
    return pd.Series([default] * len(df), index=df.index, dtype=str)


def _frame_with_columns(df: pd.DataFrame, columns: dict[str, Any]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for column, default in columns.items():
        out[column] = df[column] if column in df.columns else default
    return out


def _sanitize_records(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    safe: list[dict[str, Any]] = []
    for row in items:
        clean: dict[str, Any] = {}
        for k, v in row.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                clean[k] = None
            else:
                clean[k] = v
        safe.append(clean)
    return safe


def _latest_reviews_by_summary(db: Session, source_run: str) -> pd.DataFrame:
    rows = db.query(RuleAlertReview).filter(RuleAlertReview.source_run == source_run).all()
    if not rows:
        return pd.DataFrame(columns=["summary_alert_id", "human_review_status", "target_human_label"])
    payload: list[dict[str, Any]] = []
    for row in rows:
        if not row.summary_alert_id:
            continue
        payload.append(
            {
                "summary_alert_id": row.summary_alert_id,
                "human_review_status": row.new_status,
                "target_human_label": 1 if str(row.new_status).upper() == "CONFIRMED_FRAUD" else 0,
                "reviewed_at": row.reviewed_at.isoformat() if row.reviewed_at else "",
            }
        )
    if not payload:
        return pd.DataFrame(columns=["summary_alert_id", "human_review_status", "target_human_label"])
    df = pd.DataFrame(payload).sort_values(by=["summary_alert_id", "reviewed_at"])
    return df.groupby("summary_alert_id", as_index=False).tail(1)[["summary_alert_id", "human_review_status", "target_human_label"]]


def _resolve_paths(db: Session, source_run: str) -> dict[str, Any]:
    normalized = artifacts.normalize_source_run(source_run)
    token = artifacts.normalize_run_token(normalized)
    processed = artifacts.default_processed_dir()
    models = artifacts.default_models_dir()
    warnings: list[str] = []

    def _pick(artifact_type: str, fallback: Path) -> Path:
        item = artifacts.get_artifact_by_type(db, normalized, artifact_type)
        if item and item.status == "AVAILABLE" and item.file_path:
            return Path(item.file_path)
        return fallback

    def _pick_iso(artifact_type: str, fallback: Path, required_col: str) -> Path:
        # Use registry path only if the file actually contains the expected column;
        # otherwise the service silently reads the wrong artifact and returns 0 anomalies.
        candidate = _pick(artifact_type, fallback)
        if candidate != fallback and candidate.exists():
            try:
                sample = pd.read_csv(candidate, nrows=1)
                if required_col not in sample.columns:
                    return fallback
            except Exception:
                return fallback
        return candidate

    paths: dict[str, Any] = {
        "source_run": normalized,
        "run_token": token,
        "alerts_summary": _pick(artifacts.ARTIFACT_RULE_SUMMARY_CSV, processed / f"alerts_summary_run_{token}.csv"),
        "alerts_detail": _pick(artifacts.ARTIFACT_RULE_ALERTS_CSV, processed / f"alerts_run_{token}.csv"),
        "rules_report": _pick(artifacts.ARTIFACT_RULE_REPORT, processed / f"rules_report_run_{token}.md"),
        "isolation_scores": _pick_iso(artifacts.ARTIFACT_ANOMALY_SCORES_CSV, processed / f"anomaly_scores_run_{token}.csv", "anomaly_flag"),
        "isolation_report": _pick(artifacts.ARTIFACT_ANOMALY_REPORT, processed / f"anomaly_report_run_{token}.md"),
        "isolation_metadata": _pick(artifacts.ARTIFACT_MODEL_METADATA, models / f"isolation_forest_run_{token}_metadata.json"),
        "autoencoder_scores": processed / f"autoencoder_scores_run_{token}.csv",
        "autoencoder_report": processed / f"autoencoder_report_run_{token}.md",
        "autoencoder_metadata": models / f"autoencoder_model_run_{token}_metadata.json",
        "supervised_predictions": {},
        "supervised_metadata": {},
    }

    models_rows = db.query(ModelRegistry).filter(ModelRegistry.source_run == normalized).all()
    by_algorithm = {(m.algorithm or "").lower(): m for m in models_rows}
    for model in SUPERVISED_MODELS:
        row = by_algorithm.get(model)
        p = Path(row.scores_file) if row and row.scores_file else processed / f"supervised_human_{model}_predictions_run_{token}.csv"
        m = Path(row.metadata_file) if row and row.metadata_file else models / f"supervised_human_{model}_run_{token}_metadata.json"
        paths["supervised_predictions"][model] = p
        paths["supervised_metadata"][model] = m

    if not paths["alerts_summary"].exists():
        warnings.append("Rules summary artifact not available")
    if not paths["alerts_detail"].exists():
        warnings.append("Rules detail artifact not available")
    if not paths["isolation_scores"].exists():
        warnings.append("Isolation Forest scores artifact not available")
    if not paths["autoencoder_scores"].exists():
        warnings.append("Autoencoder scores artifact not available")
    paths["warnings"] = warnings
    return paths


def _run_full_supervised_inference(
    alert_df: pd.DataFrame,
    db: Session,
    normalized: str,
    algorithm: str,
) -> tuple[list[int], list[float] | None]:
    model, feature_columns, _ = _load_model(db, normalized, algorithm)
    X = _feature_frame_for_scoring(alert_df, feature_columns)
    return _predict(model, X)


def build_model_evaluation_comparison(db: Session, source_run: str) -> dict[str, Any]:
    paths = _resolve_paths(db, source_run)
    normalized = paths["source_run"]
    token = paths["run_token"]
    processed = artifacts.default_processed_dir()
    processed.mkdir(parents=True, exist_ok=True)
    warnings = list(paths["warnings"])

    alerts_summary = _safe_read_csv(paths["alerts_summary"])
    alerts_detail = _safe_read_csv(paths["alerts_detail"])
    iso = _safe_read_csv(paths["isolation_scores"])
    auto = _safe_read_csv(paths["autoencoder_scores"])
    reviews = _latest_reviews_by_summary(db, normalized)

    methods_available = {"rules": not alerts_summary.empty, "isolation_forest": not iso.empty, "autoencoder": not auto.empty}
    supervised_frames: dict[str, pd.DataFrame] = {}
    supervised_metrics: dict[str, Any] = {}
    for model in SUPERVISED_MODELS:
        pred = _safe_read_csv(paths["supervised_predictions"][model])
        supervised_frames[model] = pred
        methods_available[model] = not pred.empty
        if pred.empty:
            warnings.append(f"Supervised model NOT_AVAILABLE: {model}")
            continue
        y_true = _series_or_default(pred, "y_true", default=float("nan"))
        y_pred = _series_or_default(pred, "y_pred", default=float("nan"))
        valid = (~y_true.isna()) & (~y_pred.isna())
        accuracy = float((y_true[valid] == y_pred[valid]).mean()) if valid.any() else None
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
        f1 = float((2 * precision * recall) / (precision + recall)) if (precision + recall) else 0.0
        meta = _load_json(paths["supervised_metadata"][model])
        roc_auc = meta.get("metrics", {}).get("roc_auc") or meta.get("roc_auc")
        supervised_metrics[model] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
            "false_positive_count": fp,
            "false_negative_count": fn,
        }

    alert_df = alerts_summary.copy() if not alerts_summary.empty else pd.DataFrame(columns=["summary_alert_id"])
    rename_map = {"max_risk_score": "max_score", "count_transactions": "transactions_detected", "top_merchant_rubro_proxy": "merchant_rubro_proxy"}
    for old, new in rename_map.items():
        if old in alert_df.columns and new not in alert_df.columns:
            alert_df[new] = alert_df[old]
    if "source_run" in alert_df.columns:
        alert_df["source_run"] = normalized
    else:
        alert_df["source_run"] = normalized
    if not reviews.empty:
        alert_df = alert_df.merge(reviews, on="summary_alert_id", how="left")
    for model in SUPERVISED_MODELS:
        pred = supervised_frames[model]
        base = "logistic" if model == "logistic_regression" else model
        if not pred.empty:
            pred2 = pred[["summary_alert_id", "y_pred", "y_proba", "evaluation_result"]].copy()
            pred2.columns = ["summary_alert_id", f"{base}_y_pred", f"{base}_y_proba", f"{base}_evaluation_result"]
            alert_df = alert_df.merge(pred2, on="summary_alert_id", how="left")
        # Fill after merge so column always exists without causing _x/_y name conflicts
        for col in (f"{base}_y_pred", f"{base}_y_proba", f"{base}_evaluation_result"):
            if col not in alert_df.columns:
                alert_df[col] = None

    # Full inference: apply trained models to ALL alerts (overrides sparse test-set CSV predictions)
    for model in SUPERVISED_MODELS:
        base = "logistic" if model == "logistic_regression" else model
        pred_col = f"{base}_y_pred"
        proba_col = f"{base}_y_proba"
        try:
            y_pred_all, y_proba_all = _run_full_supervised_inference(alert_df, db, normalized, model)
            alert_df[pred_col] = y_pred_all
            if y_proba_all is not None:
                alert_df[proba_col] = [round(float(p), 4) for p in y_proba_all]
        except Exception:
            pass  # Model not registered or not AVAILABLE — keep CSV values (or None)

    child_map: dict[str, list[str]] = {}
    if "child_transaction_ids" in alert_df.columns:
        for _, row in alert_df[["summary_alert_id", "child_transaction_ids", "representative_transaction_id"]].fillna("").iterrows():
            txs = [t for t in str(row["child_transaction_ids"]).split("|") if t]
            rep = str(row["representative_transaction_id"]).strip()
            if rep and rep not in txs:
                txs.append(rep)
            child_map[str(row["summary_alert_id"])] = txs
    elif "representative_transaction_id" in alert_df.columns:
        for _, row in alert_df[["summary_alert_id", "representative_transaction_id"]].fillna("").iterrows():
            rep = str(row["representative_transaction_id"]).strip()
            child_map[str(row["summary_alert_id"])] = [rep] if rep else []
        warnings.append("child_transaction_ids not found; using representative_transaction_id fallback")
    else:
        warnings.append("Could not map alert-level with transaction-level anomalies")

    iso_flags = set(iso.loc[pd.to_numeric(iso.get("anomaly_flag"), errors="coerce") == 1, "transaction_id"].astype(str)) if not iso.empty and "anomaly_flag" in iso.columns else set()
    auto_flags = set(auto.loc[pd.to_numeric(auto.get("autoencoder_anomaly_flag"), errors="coerce") == 1, "transaction_id"].astype(str)) if not auto.empty and "autoencoder_anomaly_flag" in auto.columns else set()
    iso_rank_map = dict(zip(_string_series_or_default(iso, "transaction_id"), _series_or_default(iso, "anomaly_rank", default=float("nan")))) if not iso.empty else {}
    auto_err_map = dict(zip(_string_series_or_default(auto, "transaction_id"), _series_or_default(auto, "reconstruction_error", default=float("nan")))) if not auto.empty else {}

    has_iso, iso_count, iso_best_rank = [], [], []
    has_auto, auto_count, auto_max_err = [], [], []
    for sid in alert_df.get("summary_alert_id", pd.Series(dtype=str)).astype(str):
        children = child_map.get(sid, [])
        i_hits = [tx for tx in children if tx in iso_flags]
        a_hits = [tx for tx in children if tx in auto_flags]
        has_iso.append(bool(i_hits))
        iso_count.append(len(i_hits))
        ranks = [iso_rank_map.get(tx) for tx in i_hits if pd.notna(iso_rank_map.get(tx))]
        iso_best_rank.append(int(min(ranks)) if ranks else None)
        has_auto.append(bool(a_hits))
        auto_count.append(len(a_hits))
        errs = [auto_err_map.get(tx) for tx in a_hits if pd.notna(auto_err_map.get(tx))]
        auto_max_err.append(float(max(errs)) if errs else None)
    alert_df["has_isolation_forest_anomaly_child"] = has_iso
    alert_df["isolation_forest_anomaly_child_count"] = iso_count
    alert_df["isolation_forest_max_rank_in_children"] = iso_best_rank
    alert_df["has_autoencoder_anomaly_child"] = has_auto
    alert_df["autoencoder_anomaly_child_count"] = auto_count
    alert_df["autoencoder_max_reconstruction_error_in_children"] = auto_max_err

    sup_cols = [c for c in ("logistic_y_pred", "random_forest_y_pred", "gradient_boosting_y_pred") if c in alert_df.columns]
    if sup_cols:
        alert_df["supervised_positive_any"] = alert_df[sup_cols].fillna(0).astype(float).ge(1).any(axis=1)
    else:
        alert_df["supervised_positive_any"] = False
    alert_df["unsupervised_anomaly_any"] = alert_df["has_isolation_forest_anomaly_child"] | alert_df["has_autoencoder_anomaly_child"]
    alert_df["rule_present"] = True
    alert_df["methods_agree_count"] = (
        alert_df["rule_present"].astype(int)
        + alert_df["supervised_positive_any"].astype(int)
        + alert_df["unsupervised_anomaly_any"].astype(int)
    )
    high_risk = alert_df.get("risk_level", pd.Series(dtype=str)).astype(str).str.upper().eq("HIGH")
    tx_gt_one = _series_or_default(alert_df, "transactions_detected", default=0.0).fillna(0).gt(1)
    human_confirmed = alert_df.get("human_review_status", pd.Series(dtype=str)).astype(str).str.upper().eq("CONFIRMED_FRAUD")
    alert_df["comparison_priority_score"] = (
        human_confirmed.astype(int) * 3
        + alert_df["supervised_positive_any"].astype(int) * 2
        + alert_df["has_isolation_forest_anomaly_child"].astype(int)
        + alert_df["has_autoencoder_anomaly_child"].astype(int)
        + high_risk.astype(int)
        + tx_gt_one.astype(int)
        + alert_df["methods_agree_count"].ge(2).astype(int)
    )
    alert_df["priority_reason"] = alert_df["methods_agree_count"].apply(lambda x: "MULTI_METHOD_OVERLAP" if int(x) >= 2 else "SINGLE_METHOD")

    tx_df = _build_transaction_level_from_alerts(alerts_detail)
    if tx_df.empty:
        tx_df = pd.DataFrame({"transaction_id": sorted(iso_flags | auto_flags)})
        tx_df["has_rule_alert"] = False
    tx_df["source_run"] = normalized
    if not iso.empty:
        iso_merge = _frame_with_columns(
            iso,
            {
                "transaction_id": "",
                "anomaly_score": None,
                "anomaly_flag": 0,
                "anomaly_rank": None,
            },
        )
        tx_df = tx_df.merge(
            iso_merge[["transaction_id", "anomaly_score", "anomaly_flag", "anomaly_rank"]].rename(
                columns={"anomaly_score": "isolation_anomaly_score", "anomaly_flag": "isolation_anomaly_flag", "anomaly_rank": "isolation_anomaly_rank"}
            ),
            on="transaction_id",
            how="left",
        )
    if not auto.empty:
        auto_merge = _frame_with_columns(
            auto,
            {
                "transaction_id": "",
                "reconstruction_error": None,
                "autoencoder_anomaly_score": None,
                "autoencoder_anomaly_flag": 0,
                "anomaly_rank": None,
            },
        )
        tx_df = tx_df.merge(auto_merge[["transaction_id", "reconstruction_error", "autoencoder_anomaly_score", "autoencoder_anomaly_flag", "anomaly_rank"]].rename(columns={"anomaly_rank": "autoencoder_anomaly_rank"}), on="transaction_id", how="left")
    has_rule_series = tx_df["has_rule_alert"] if "has_rule_alert" in tx_df.columns else pd.Series([False] * len(tx_df), index=tx_df.index)
    tx_df["flagged_by_rules"] = has_rule_series.fillna(False).astype(bool)
    tx_df["flagged_by_isolation_forest"] = _series_or_default(tx_df, "isolation_anomaly_flag", default=0.0).fillna(0).astype(int).eq(1)
    tx_df["flagged_by_autoencoder"] = _series_or_default(tx_df, "autoencoder_anomaly_flag", default=0.0).fillna(0).astype(int).eq(1)
    tx_df["unsupervised_methods_count"] = tx_df["flagged_by_isolation_forest"].astype(int) + tx_df["flagged_by_autoencoder"].astype(int)
    tx_df["total_signal_count"] = tx_df["flagged_by_rules"].astype(int) + tx_df["unsupervised_methods_count"]
    tx_df["comparison_priority"] = tx_df["total_signal_count"].apply(lambda x: "HIGH" if int(x) >= 2 else "MEDIUM" if int(x) == 1 else "LOW")

    has_any_supervised_data = any(not f.empty for f in supervised_frames.values())
    has_any_unsupervised_data = not iso.empty or not auto.empty
    max_possible_agree = int(not alerts_summary.empty) + int(has_any_supervised_data) + int(has_any_unsupervised_data)

    metrics = {
        "rules": {
            "total_alerts_grouped": int(len(alert_df)),
            "total_alerts_detailed": int(len(alerts_detail)),
            "total_rules": int(alert_df.get("rule_code", pd.Series(dtype=str)).nunique()) if "rule_code" in alert_df.columns else 0,
            "alerts_by_rule": alert_df["rule_code"].value_counts().to_dict() if "rule_code" in alert_df.columns else {},
            "alerts_by_risk_level": alert_df["risk_level"].value_counts().to_dict() if "risk_level" in alert_df.columns else {},
        },
        "isolation_forest": {
            "total_records": int(len(iso)),
            "anomaly_count": int(_series_or_default(iso, "anomaly_flag", default=0.0).fillna(0).sum()) if not iso.empty else 0,
            "anomaly_rate": float(_series_or_default(iso, "anomaly_flag", default=0.0).fillna(0).mean()) if not iso.empty else 0.0,
            "contamination": _load_json(paths["isolation_metadata"]).get("contamination") if paths["isolation_metadata"].exists() else None,
        },
        "autoencoder": {
            "total_records": int(len(auto)),
            "anomaly_count": int(_series_or_default(auto, "autoencoder_anomaly_flag", default=0.0).fillna(0).sum()) if not auto.empty else 0,
            "anomaly_rate": float(_series_or_default(auto, "autoencoder_anomaly_flag", default=0.0).fillna(0).mean()) if not auto.empty else 0.0,
            "reconstruction_threshold": _load_json(paths["autoencoder_metadata"]).get("threshold") if paths["autoencoder_metadata"].exists() else None,
        },
        "supervised": supervised_metrics,
    }
    intersections = {
        "rules_and_isolation_count": int((tx_df["flagged_by_rules"] & tx_df["flagged_by_isolation_forest"]).sum()),
        "rules_and_autoencoder_count": int((tx_df["flagged_by_rules"] & tx_df["flagged_by_autoencoder"]).sum()),
        "isolation_and_autoencoder_count": int((tx_df["flagged_by_isolation_forest"] & tx_df["flagged_by_autoencoder"]).sum()),
        "rules_and_supervised_positive_count": int(alert_df["supervised_positive_any"].sum()) if "supervised_positive_any" in alert_df.columns else 0,
        "all_available_methods_count": int(alert_df["methods_agree_count"].eq(max_possible_agree).sum()) if "methods_agree_count" in alert_df.columns and max_possible_agree > 1 else 0,
    }

    alert_df = _drop_forbidden(alert_df)
    tx_df = _drop_forbidden(tx_df)
    alert_file = processed / f"model_comparison_alert_level_run_{token}.csv"
    tx_file = processed / f"model_comparison_transaction_level_run_{token}.csv"
    report_file = processed / f"model_evaluation_comparison_report_run_{token}.md"
    metadata_file = processed / f"model_evaluation_comparison_metadata_run_{token}.json"
    alert_df.to_csv(alert_file, index=False)
    tx_df.to_csv(tx_file, index=False)

    top = alert_df.sort_values(by=["comparison_priority_score", "methods_agree_count"], ascending=False).head(20)
    lines = [
        f"# Model Evaluation Comparison Report ({normalized})",
        "",
        f"- source_run: `{normalized}`",
        f"- generated_at: `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "## Methods",
        f"- available_methods: `{[k for k, v in methods_available.items() if v]}`",
        f"- missing_methods: `{[k for k, v in methods_available.items() if not v]}`",
        "",
        "## Rules",
        f"- grouped alerts: {metrics['rules']['total_alerts_grouped']}",
        f"- detailed alerts: {metrics['rules']['total_alerts_detailed']}",
        "",
        "## Isolation Forest",
        f"- anomaly_count: {metrics['isolation_forest']['anomaly_count']}",
        f"- anomaly_rate: {metrics['isolation_forest']['anomaly_rate']}",
        "",
        "## Autoencoder",
        f"- anomaly_count: {metrics['autoencoder']['anomaly_count']}",
        f"- anomaly_rate: {metrics['autoencoder']['anomaly_rate']}",
        "",
        "## Supervised Metrics",
        json.dumps(supervised_metrics, ensure_ascii=False, indent=2),
        "",
        "## Intersections",
        json.dumps(intersections, ensure_ascii=False, indent=2),
        "",
        "## Top Cases",
    ]
    for _, row in top.iterrows():
        lines.append(f"- {row.get('summary_alert_id')} | priority={row.get('comparison_priority_score')} | agree={row.get('methods_agree_count')}")
    lines.extend(
        [
            "",
            "## Limitations",
            "Las métricas supervisadas son preliminares si el dataset etiquetado solo cumple el mínimo técnico.",
            "",
            "## Methodological Warning",
            "Las reglas, anomalías no supervisadas y predicciones supervisadas son señales de apoyo analítico. Ninguna de estas salidas constituye fraude confirmado automático. La confirmación depende de revisión humana.",
        ]
    )
    report_file.write_text("\n".join(lines), encoding="utf-8")

    metadata = {
        "source_run": normalized,
        "run_token": token,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "available_methods": [k for k, v in methods_available.items() if v],
        "missing_methods": [k for k, v in methods_available.items() if not v],
        "files_used": {k: str(v) for k, v in paths.items() if k not in {"warnings", "supervised_predictions", "supervised_metadata"}},
        "output_files": {"alert_level": str(alert_file), "transaction_level": str(tx_file), "report": str(report_file), "metadata": str(metadata_file)},
        "metrics": metrics,
        "intersections": intersections,
        "warnings": warnings,
        "methodological_notes": [
            "No anomaly_flag, autoencoder_anomaly_flag ni y_pred fueron convertidos a fraude confirmado.",
            "No se modificó rule_alert_reviews ni estados de alerta.",
        ],
    }
    metadata_file.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    for output, role in (
        (alert_file, "alert_level"),
        (tx_file, "transaction_level"),
        (report_file, "report"),
        (metadata_file, "metadata"),
    ):
        typ = ARTIFACT_MODEL_EVALUATION_DATASET if role in {"alert_level", "transaction_level"} else ARTIFACT_MODEL_EVALUATION_REPORT if role == "report" else ARTIFACT_MODEL_EVALUATION_METADATA
        artifacts.register_or_update_artifact(
            db,
            artifact_type=typ,
            phase=PHASE_C5,
            source_run=normalized,
            run_token=token,
            file_path=output,
            metadata={"artifact_role": role, "source": "model_evaluation_c5_1"},
        )
    return metadata


def _read_output_csv(source_run: str, kind: str) -> pd.DataFrame:
    token = artifacts.normalize_run_token(source_run)
    processed = artifacts.default_processed_dir()
    path = processed / f"model_comparison_{kind}_level_run_{token}.csv"
    return _safe_read_csv(path)


def get_summary(source_run: str) -> dict[str, Any]:
    token = artifacts.normalize_run_token(source_run)
    path = artifacts.default_processed_dir() / f"model_evaluation_comparison_metadata_run_{token}.json"
    if not path.exists():
        return {"status": "NOT_AVAILABLE", "source_run": artifacts.normalize_source_run(source_run)}
    return _load_json(path)


def get_alert_level(source_run: str, page: int, page_size: int, filters: dict[str, Any] | None = None) -> dict[str, Any]:
    df = _read_output_csv(source_run, "alert")
    filters = filters or {}
    for key, value in filters.items():
        if value is None or str(value).strip() == "" or key not in df.columns:
            continue
        df = df[df[key].astype(str).str.lower() == str(value).strip().lower()]
    total = len(df)
    start = max(page - 1, 0) * page_size
    items = _sanitize_records(df.iloc[start : start + page_size].to_dict(orient="records"))
    return {"total": total, "page": page, "page_size": page_size, "items": items}


def get_transaction_level(source_run: str, page: int, page_size: int, filters: dict[str, Any] | None = None) -> dict[str, Any]:
    df = _read_output_csv(source_run, "transaction")
    filters = filters or {}
    for key, value in filters.items():
        if value is None or str(value).strip() == "" or key not in df.columns:
            continue
        df = df[df[key].astype(str).str.lower() == str(value).strip().lower()]
    total = len(df)
    start = max(page - 1, 0) * page_size
    items = _sanitize_records(df.iloc[start : start + page_size].to_dict(orient="records"))
    return {"total": total, "page": page, "page_size": page_size, "items": items}


def get_report_markdown(source_run: str) -> str:
    token = artifacts.normalize_run_token(source_run)
    path = artifacts.default_processed_dir() / f"model_evaluation_comparison_report_run_{token}.md"
    return path.read_text(encoding="utf-8") if path.exists() else ""


def get_metadata(source_run: str) -> dict[str, Any]:
    token = artifacts.normalize_run_token(source_run)
    path = artifacts.default_processed_dir() / f"model_evaluation_comparison_metadata_run_{token}.json"
    return _load_json(path) if path.exists() else {}


def get_top_cases(source_run: str, limit: int = 20) -> list[dict[str, Any]]:
    df = _read_output_csv(source_run, "alert")
    if df.empty or "comparison_priority_score" not in df.columns:
        return []
    rows = df.sort_values(by=["comparison_priority_score", "methods_agree_count"], ascending=False).head(max(limit, 1)).to_dict(orient="records")
    return _sanitize_records(rows)

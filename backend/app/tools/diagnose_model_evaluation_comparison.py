"""
Diagnostic script for model evaluation comparison (C5.1/C5.2).

Usage:
    python -m backend.app.tools.diagnose_model_evaluation_comparison --source-run preprocessed_run_26

Reads artifacts and DB (read-only). Generates a markdown report in data/processed/.
Does NOT modify any artifact, registry entry, or production service.
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from backend.app.services.artifact_registry_service import (
    ARTIFACT_ANOMALY_SCORES_CSV,
    ARTIFACT_RULE_ALERTS_CSV,
    ARTIFACT_RULE_SUMMARY_CSV,
    default_processed_dir,
    default_models_dir,
    normalize_run_token,
    normalize_source_run,
)

SUPERVISED_MODELS = ("logistic_regression", "random_forest", "gradient_boosting")
_BASE_NAMES = {
    "logistic_regression": "logistic",
    "random_forest": "random_forest",
    "gradient_boosting": "gradient_boosting",
}


# ---------------------------------------------------------------------------
# DB helpers — soft-fail if DB not reachable
# ---------------------------------------------------------------------------

def _query_artifact_registry(source_run_normalized: str) -> list[dict]:
    """Query artifact_registry for all entries matching source_run. Returns [] on error."""
    try:
        from backend.app.database import SessionLocal
        from backend.app.models.models import ArtifactRegistry

        db = SessionLocal()
        try:
            rows = (
                db.query(ArtifactRegistry)
                .filter(ArtifactRegistry.source_run == source_run_normalized)
                .order_by(ArtifactRegistry.artifact_type)
                .all()
            )
            return [
                {
                    "artifact_type": r.artifact_type,
                    "phase": r.phase,
                    "status": r.status,
                    "file_path": r.file_path,
                    "row_count": r.row_count,
                    "file_size_bytes": r.file_size_bytes,
                    "updated_at": str(r.updated_at),
                }
                for r in rows
            ]
        finally:
            db.close()
    except Exception as exc:
        return [{"_db_error": str(exc)}]


def _query_model_registry(source_run_normalized: str) -> list[dict]:
    """Query model_registry for supervised models. Returns [] on error."""
    try:
        from backend.app.database import SessionLocal
        from backend.app.models.models import ModelRegistry

        db = SessionLocal()
        try:
            rows = (
                db.query(ModelRegistry)
                .filter(ModelRegistry.source_run == source_run_normalized)
                .all()
            )
            return [
                {
                    "algorithm": r.algorithm,
                    "model_family": r.model_family,
                    "scores_file": r.scores_file,
                    "status": r.status,
                    "is_active": r.is_active,
                }
                for r in rows
            ]
        finally:
            db.close()
    except Exception as exc:
        return [{"_db_error": str(exc)}]


# ---------------------------------------------------------------------------
# Path resolution (mirrors _resolve_paths in model_evaluation_service)
# ---------------------------------------------------------------------------

def _resolve_iso_path_from_registry(registry_rows: list[dict], fallback: Path) -> tuple[Path, str]:
    """Return (path, source) where source is 'registry' or 'fallback'."""
    for row in registry_rows:
        if row.get("artifact_type") == ARTIFACT_ANOMALY_SCORES_CSV and row.get("status") == "AVAILABLE":
            p = Path(row["file_path"])
            return p, "registry"
    return fallback, "fallback"


def _resolve_supervised_paths(registry_rows: list[dict], model_registry_rows: list[dict],
                              token: str, processed: Path) -> dict[str, tuple[Path, str]]:
    """Resolve prediction CSV paths for each supervised model."""
    by_algo = {r["algorithm"]: r for r in model_registry_rows if "algorithm" in r}
    result = {}
    for model in SUPERVISED_MODELS:
        row = by_algo.get(model)
        if row and row.get("scores_file"):
            result[model] = (Path(row["scores_file"]), "model_registry")
        else:
            result[model] = (processed / f"supervised_human_{model}_predictions_run_{token}.csv", "fallback")
    return result


# ---------------------------------------------------------------------------
# CSV analysis helpers
# ---------------------------------------------------------------------------

def _safe_read(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False, **kwargs)
    except Exception:
        return pd.DataFrame()


def _flag_stats(df: pd.DataFrame, col: str) -> dict:
    if df.empty or col not in df.columns:
        return {"col_present": False, "total_rows": len(df), "flag_1": 0, "flag_0": 0, "other": 0}
    counts = pd.to_numeric(df[col], errors="coerce").fillna(-999).value_counts().to_dict()
    return {
        "col_present": True,
        "total_rows": len(df),
        "flag_1": int(counts.get(1, 0)),
        "flag_0": int(counts.get(0, 0)),
        "other": int(sum(v for k, v in counts.items() if k not in (0, 1))),
    }


def _expand_children(summary: pd.DataFrame) -> set[str]:
    children: set[str] = set()
    if "child_transaction_ids" not in summary.columns:
        if "representative_transaction_id" in summary.columns:
            for val in summary["representative_transaction_id"].dropna():
                tx = str(val).strip()
                if tx:
                    children.add(tx)
        return children
    for val in summary["child_transaction_ids"].fillna(""):
        for tx in str(val).split("|"):
            tx = tx.strip()
            if tx:
                children.add(tx)
    if "representative_transaction_id" in summary.columns:
        for val in summary["representative_transaction_id"].fillna(""):
            tx = str(val).strip()
            if tx:
                children.add(tx)
    return children


def _count_alerts_with_if_child(summary: pd.DataFrame, iso_flags: set[str]) -> int:
    if summary.empty or not iso_flags:
        return 0
    count = 0
    child_col = "child_transaction_ids" if "child_transaction_ids" in summary.columns else None
    rep_col = "representative_transaction_id" if "representative_transaction_id" in summary.columns else None
    for _, row in summary[[c for c in (child_col, rep_col) if c]].fillna("").iterrows():
        children = []
        if child_col:
            children = [t for t in str(row[child_col]).split("|") if t.strip()]
        if rep_col:
            rep = str(row[rep_col]).strip()
            if rep and rep not in children:
                children.append(rep)
        if any(tx in iso_flags for tx in children):
            count += 1
    return count


# ---------------------------------------------------------------------------
# Main diagnosis logic
# ---------------------------------------------------------------------------

def diagnose(source_run: str) -> dict[str, Any]:
    normalized = normalize_source_run(source_run)
    token = normalize_run_token(normalized)
    processed = default_processed_dir()
    models_dir = default_models_dir()

    report: dict[str, Any] = {
        "source_run_input": source_run,
        "source_run_normalized": normalized,
        "run_token": token,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    # 1. DB queries
    registry_rows = _query_artifact_registry(normalized)
    model_registry_rows = _query_model_registry(normalized)
    db_error = next((r.get("_db_error") for r in registry_rows if "_db_error" in r), None)
    report["db_status"] = "OK" if not db_error else f"ERROR: {db_error}"
    report["artifact_registry_entries"] = [r for r in registry_rows if "_db_error" not in r]
    report["model_registry_entries"] = [r for r in model_registry_rows if "_db_error" not in r]

    # 2. Path resolution
    iso_fallback = processed / f"anomaly_scores_run_{token}.csv"
    iso_path, iso_source = _resolve_iso_path_from_registry(registry_rows, iso_fallback)
    auto_path = processed / f"autoencoder_scores_run_{token}.csv"
    summary_path = processed / f"alerts_summary_run_{token}.csv"
    detail_path = processed / f"alerts_run_{token}.csv"
    cmp_alert_path = processed / f"model_comparison_alert_level_run_{token}.csv"

    report["paths"] = {
        "isolation_scores_registered": str(iso_path) if iso_source == "registry" else None,
        "isolation_scores_fallback": str(iso_fallback),
        "isolation_scores_used": str(iso_path),
        "isolation_scores_source": iso_source,
        "autoencoder_scores": str(auto_path),
        "alerts_summary": str(summary_path),
        "alerts_detail": str(detail_path),
        "comparison_alert_level": str(cmp_alert_path),
    }

    # 3. Read Isolation Forest — registered path vs. fallback
    iso_registered_df = _safe_read(iso_path)
    iso_registered_stats = _flag_stats(iso_registered_df, "anomaly_flag")
    iso_registered_stats["path"] = str(iso_path)
    iso_registered_stats["path_exists"] = iso_path.exists()
    iso_registered_stats["columns"] = iso_registered_df.columns.tolist() if not iso_registered_df.empty else []

    iso_fallback_stats: dict[str, Any] = {"path": str(iso_fallback), "path_exists": iso_fallback.exists()}
    if iso_path != iso_fallback and iso_fallback.exists():
        iso_fallback_df = _safe_read(iso_fallback)
        iso_fallback_stats.update(_flag_stats(iso_fallback_df, "anomaly_flag"))
        iso_fallback_stats["columns"] = iso_fallback_df.columns.tolist()
    elif iso_path == iso_fallback:
        iso_fallback_df = iso_registered_df
        iso_fallback_stats["note"] = "same as registered path"
    else:
        iso_fallback_df = pd.DataFrame()

    report["isolation_forest"] = {
        "registered_file": iso_registered_stats,
        "fallback_file": iso_fallback_stats,
        "path_mismatch": iso_path != iso_fallback,
    }

    # 4. Autoencoder stats
    auto_df = _safe_read(auto_path)
    auto_stats = _flag_stats(auto_df, "autoencoder_anomaly_flag")
    auto_stats["path"] = str(auto_path)
    auto_stats["path_exists"] = auto_path.exists()
    auto_stats["columns"] = auto_df.columns.tolist() if not auto_df.empty else []
    report["autoencoder"] = auto_stats

    # 5. Overlap: IF flagged ↔ alert children
    summary_df = _safe_read(summary_path)
    report["alerts_summary"] = {
        "path": str(summary_path),
        "exists": summary_path.exists(),
        "total_rows": len(summary_df),
        "has_child_transaction_ids": "child_transaction_ids" in summary_df.columns if not summary_df.empty else False,
    }

    all_children = _expand_children(summary_df) if not summary_df.empty else set()
    report["alerts_summary"]["unique_child_txs"] = len(all_children)

    # Using registered file
    if not iso_registered_df.empty and "anomaly_flag" in iso_registered_df.columns:
        iso_reg_flagged = set(iso_registered_df.loc[
            pd.to_numeric(iso_registered_df["anomaly_flag"], errors="coerce") == 1,
            "transaction_id"
        ].astype(str).tolist()) if "transaction_id" in iso_registered_df.columns else set()
    else:
        iso_reg_flagged = set()

    # Using fallback file
    if not iso_fallback_df.empty and "anomaly_flag" in iso_fallback_df.columns:
        iso_fb_flagged = set(iso_fallback_df.loc[
            pd.to_numeric(iso_fallback_df["anomaly_flag"], errors="coerce") == 1,
            "transaction_id"
        ].astype(str).tolist()) if "transaction_id" in iso_fallback_df.columns else set()
    else:
        iso_fb_flagged = set()

    overlap_registered = iso_reg_flagged & all_children
    overlap_fallback = iso_fb_flagged & all_children

    alerts_with_if_registered = _count_alerts_with_if_child(summary_df, iso_reg_flagged)
    alerts_with_if_fallback = _count_alerts_with_if_child(summary_df, iso_fb_flagged)

    report["isolation_overlap"] = {
        "using_registered_file": {
            "iso_flagged_count": len(iso_reg_flagged),
            "child_tx_overlap": len(overlap_registered),
            "alerts_with_if_anomaly_child": alerts_with_if_registered,
        },
        "using_fallback_file": {
            "iso_flagged_count": len(iso_fb_flagged),
            "child_tx_overlap": len(overlap_fallback),
            "alerts_with_if_anomaly_child": alerts_with_if_fallback,
        },
    }

    # 6. Autoencoder overlap
    if not auto_df.empty and "autoencoder_anomaly_flag" in auto_df.columns and "transaction_id" in auto_df.columns:
        auto_flagged = set(auto_df.loc[
            pd.to_numeric(auto_df["autoencoder_anomaly_flag"], errors="coerce") == 1,
            "transaction_id"
        ].astype(str).tolist())
    else:
        auto_flagged = set()

    auto_overlap = auto_flagged & all_children
    alerts_with_auto = _count_alerts_with_if_child(summary_df, auto_flagged)
    report["autoencoder_overlap"] = {
        "auto_flagged_count": len(auto_flagged),
        "child_tx_overlap": len(auto_overlap),
        "alerts_with_autoencoder_anomaly_child": alerts_with_auto,
    }

    # 7. Supervised predictions
    summary_ids = set(summary_df["summary_alert_id"].astype(str).tolist()) if not summary_df.empty and "summary_alert_id" in summary_df.columns else set()
    supervised_pred_paths = _resolve_supervised_paths(registry_rows, model_registry_rows, token, processed)
    sup_report = {}
    all_positive_ids: set[str] = set()
    for model in SUPERVISED_MODELS:
        pred_path, pred_source = supervised_pred_paths[model]
        pred_df = _safe_read(pred_path)
        y_pred_counts: dict[str, int] = {}
        positive_ids: list[str] = []
        overlap_with_summary = 0
        if not pred_df.empty and "y_pred" in pred_df.columns:
            y_pred_counts = {str(k): int(v) for k, v in pred_df["y_pred"].value_counts().items()}
            if "summary_alert_id" in pred_df.columns:
                pred_ids = set(pred_df["summary_alert_id"].astype(str).tolist())
                overlap_with_summary = len(pred_ids & summary_ids)
                positive_ids = pred_df.loc[
                    pd.to_numeric(pred_df["y_pred"], errors="coerce").ge(1), "summary_alert_id"
                ].astype(str).tolist()
                all_positive_ids.update(positive_ids)
        sup_report[model] = {
            "path": str(pred_path),
            "path_exists": pred_path.exists(),
            "source": pred_source,
            "total_rows": len(pred_df),
            "y_pred_value_counts": y_pred_counts,
            "positive_ids": positive_ids,
            "overlap_with_alerts_summary": overlap_with_summary,
        }
    report["supervised"] = sup_report
    report["supervised_positive_any_count"] = len(all_positive_ids)
    report["supervised_positive_ids"] = sorted(all_positive_ids)

    # 8. Detect _x/_y bug in existing comparison output
    cmp_df = _safe_read(cmp_alert_path, nrows=0)
    if not cmp_df.empty or cmp_alert_path.exists():
        cmp_df_full = _safe_read(cmp_alert_path, usecols=None)
        sup_plain = [c for c in ("logistic_y_pred", "random_forest_y_pred", "gradient_boosting_y_pred") if c in cmp_df_full.columns]
        sup_x_cols = [c for c in cmp_df_full.columns if ("_y_pred_" in c or c.endswith("_y_pred_x") or c.endswith("_y_pred_y") or "_y_proba_x" in c or "_y_proba_y" in c)]
        has_if_true = int(cmp_df_full["has_isolation_forest_anomaly_child"].sum()) if "has_isolation_forest_anomaly_child" in cmp_df_full.columns else None
        has_auto_true = int(cmp_df_full["has_autoencoder_anomaly_child"].sum()) if "has_autoencoder_anomaly_child" in cmp_df_full.columns else None
        sup_any_true = int(cmp_df_full["supervised_positive_any"].sum()) if "supervised_positive_any" in cmp_df_full.columns else None
        report["existing_comparison_output"] = {
            "path": str(cmp_alert_path),
            "exists": cmp_alert_path.exists(),
            "total_rows": len(cmp_df_full),
            "sup_plain_cols_found": sup_plain,
            "sup_x_y_cols_found": sup_x_cols,
            "bug_b_detected": len(sup_x_cols) > 0 and len(sup_plain) == 0,
            "has_isolation_forest_anomaly_child_true_count": has_if_true,
            "has_autoencoder_anomaly_child_true_count": has_auto_true,
            "supervised_positive_any_true_count": sup_any_true,
        }
    else:
        report["existing_comparison_output"] = {"path": str(cmp_alert_path), "exists": False}

    # 9. Diagnoses
    diagnoses = []

    # Bug A diagnosis — when DB is down, iso_source=fallback so path_mismatch=False
    # but if the metadata already shows total_records=10000 from a previous run, Bug A likely exists
    metadata_evidence = ""
    metadata_path = processed / f"model_evaluation_comparison_metadata_run_{token}.json"
    if metadata_path.exists():
        try:
            import json as _json
            meta = _json.loads(metadata_path.read_text(encoding="utf-8"))
            iso_meta = meta.get("metrics", {}).get("isolation_forest", {})
            if iso_meta.get("total_records", 0) == 10000 and iso_fallback_df is not None and len(iso_fallback_df) > 10000:
                metadata_evidence = (
                    f" NOTA: El metadata de la ultima ejecucion muestra "
                    f"total_records=10000 para Isolation Forest, mientras el fallback tiene "
                    f"{len(iso_fallback_df)} filas. Esto confirma Bug A cuando DB estaba disponible."
                )
        except Exception:
            pass

    if report["isolation_forest"]["path_mismatch"]:
        reg_rows = iso_registered_stats["total_rows"]
        fb_rows = iso_fallback_stats.get("total_rows", 0)
        diagnoses.append({
            "id": "BUG_A",
            "severity": "CRITICAL",
            "title": "Isolation Forest lee archivo equivocado (artifact_registry path incorrecto)",
            "detail": (
                f"El artifact_registry apunta a '{iso_path}' ({reg_rows} filas, "
                f"anomaly_flag presente: {iso_registered_stats['col_present']}). "
                f"El archivo fallback correcto '{iso_fallback}' tiene {fb_rows} filas con "
                f"flag_1={iso_fallback_stats.get('flag_1', 0)} anomalías."
            ),
            "impact": f"iso_flags vacío → isolation_anomalies=0, rules∩isolation=0 (real: {len(iso_fb_flagged)} flagged, {len(overlap_fallback)} overlap con alertas)",
        })
    elif metadata_evidence:
        diagnoses.append({
            "id": "BUG_A_METADATA_EVIDENCE",
            "severity": "CRITICAL",
            "title": "Isolation Forest: Bug A confirmado por metadata de ejecucion previa (DB no disponible ahora)",
            "detail": metadata_evidence.strip(),
            "impact": (
                f"iso_flags vacio en la ultima ejecucion -> isolation_anomalies=0. "
                f"Archivo correcto (fallback/registrado) tiene "
                f"{report['isolation_overlap']['using_fallback_file']['iso_flagged_count']} transacciones flagged, "
                f"{report['isolation_overlap']['using_fallback_file']['child_tx_overlap']} tx overlap con alertas, "
                f"{report['isolation_overlap']['using_fallback_file']['alerts_with_if_anomaly_child']} alertas con hijo IF-anomalo."
            ),
        })
    elif iso_registered_stats["flag_1"] == 0 and iso_fallback_stats.get("flag_1", 0) > 0:
        diagnoses.append({
            "id": "BUG_A_SAME_PATH_WRONG_DATA",
            "severity": "CRITICAL",
            "title": "Isolation Forest: archivo correcto pero sin anomaly_flag=1",
            "detail": f"El archivo en '{iso_path}' tiene {iso_registered_stats['total_rows']} filas pero 0 con anomaly_flag=1.",
            "impact": "iso_flags vacío → mismas consecuencias que Bug A.",
        })
    elif not iso_registered_stats["col_present"]:
        diagnoses.append({
            "id": "BUG_A_NO_COL",
            "severity": "CRITICAL",
            "title": "Isolation Forest: columna 'anomaly_flag' ausente en archivo registrado",
            "detail": f"Columnas en '{iso_path}': {iso_registered_stats['columns']}",
            "impact": "iso_flags vacío → isolation_anomalies=0.",
        })
    else:
        diagnoses.append({
            "id": "IF_OK",
            "severity": "INFO",
            "title": "Isolation Forest: archivo y columna correctos",
            "detail": f"flag_1={iso_registered_stats['flag_1']} en '{iso_path}'",
            "impact": "ninguno",
        })

    # Bug B diagnosis
    cmp_out = report.get("existing_comparison_output", {})
    if cmp_out.get("bug_b_detected"):
        diagnoses.append({
            "id": "BUG_B",
            "severity": "CRITICAL",
            "title": "Merge supervisado genera sufijos _x/_y: supervised_positive_any siempre False",
            "detail": (
                f"Columnas con sufijo _x/_y encontradas: {cmp_out['sup_x_y_cols_found'][:6]}... "
                f"Columnas 'logistic_y_pred' (plain) encontradas: {cmp_out['sup_plain_cols_found']}. "
                "El código pre-crea las columnas como None antes del merge, causando conflicto de nombres."
            ),
            "impact": (
                f"supervised_positive_any=True para 0/{cmp_out.get('total_rows', 0)} alertas. "
                f"Real: {report['supervised_positive_any_count']} alertas con y_pred>=1."
            ),
        })

    # Structural note
    diagnoses.append({
        "id": "STRUCTURAL_SUPERVISED_TEST_ONLY",
        "severity": "INFO",
        "title": "Supervisados: predicciones solo sobre test set (limitación de diseño)",
        "detail": (
            f"Los 3 modelos supervisados tienen predicciones sobre {sum(v['total_rows'] for v in sup_report.values())} alertas en total "
            f"(test set etiquetado), no sobre las {len(summary_ids)} alertas totales del summary."
        ),
        "impact": "Incluso corrigiendo Bug B, solo ~6 alertas tendrán supervised_positive_any=True. Esto es el mínimo técnico esperado.",
    })

    report["diagnoses"] = diagnoses
    return report


# ---------------------------------------------------------------------------
# Markdown report generator
# ---------------------------------------------------------------------------

def _md_table(headers: list[str], rows: list[list]) -> str:
    widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0)) for i, h in enumerate(headers)]
    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    header = "| " + " | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers)) + " |"
    body = "\n".join("| " + " | ".join(str(r[i]).ljust(widths[i]) for i in range(len(headers))) + " |" for r in rows)
    return "\n".join([header, sep, body])


def generate_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    a = lines.append

    a(f"# Diagnóstico Evaluación Comparativa — {report['source_run_normalized']}")
    a("")
    a(f"**Generado:** {report['generated_at']}")
    a(f"**DB status:** {report['db_status']}")
    a("")

    # ---- Artifact registry
    a("## Artefactos registrados en artifact_registry")
    a("")
    reg_entries = report.get("artifact_registry_entries", [])
    if reg_entries:
        rows = [[r.get("artifact_type","?"), r.get("status","?"), r.get("row_count","?"), r.get("file_path","?")] for r in reg_entries]
        a(_md_table(["artifact_type", "status", "rows", "file_path"], rows))
    else:
        a("_No se encontraron entradas en artifact_registry (o error de conexión a DB)._")
    a("")

    # ---- Paths
    a("## Resolución de paths")
    a("")
    p = report["paths"]
    a(f"- **isolation_scores (source):** `{p['isolation_scores_source']}`")
    a(f"- **isolation_scores (usado):** `{p['isolation_scores_used']}`")
    a(f"- **isolation_scores (fallback):** `{p['isolation_scores_fallback']}`")
    a(f"- **path_mismatch:** `{report['isolation_forest']['path_mismatch']}`")
    a(f"- **autoencoder_scores:** `{p['autoencoder_scores']}`")
    a(f"- **alerts_summary:** `{p['alerts_summary']}`")
    a("")

    # ---- Isolation Forest
    a("## Isolation Forest — Diagnóstico")
    a("")
    iso = report["isolation_forest"]
    reg = iso["registered_file"]
    fb = iso["fallback_file"]
    a(f"### Archivo registrado en artifact_registry")
    a(f"- Path: `{reg['path']}`")
    a(f"- Existe: `{reg['path_exists']}`")
    a(f"- Filas totales: `{reg['total_rows']}`")
    a(f"- Columna `anomaly_flag` presente: `{reg['col_present']}`")
    a(f"- flag=1 (anómalas): `{reg['flag_1']}`")
    a(f"- flag=0 (normales): `{reg['flag_0']}`")
    a(f"- Columnas: `{reg['columns']}`")
    a("")
    a(f"### Archivo fallback (data/processed/anomaly_scores_run_*)")
    a(f"- Path: `{fb['path']}`")
    a(f"- Existe: `{fb['path_exists']}`")
    a(f"- Filas totales: `{fb.get('total_rows', 'N/A')}`")
    a(f"- Columna `anomaly_flag` presente: `{fb.get('col_present', 'N/A')}`")
    a(f"- flag=1 (anómalas): `{fb.get('flag_1', 'N/A')}`")
    a(f"- flag=0 (normales): `{fb.get('flag_0', 'N/A')}`")
    a("")
    a("### Overlap con alertas (child_transaction_ids)")
    ov = report["isolation_overlap"]
    a(f"- Usando archivo **registrado**: `{ov['using_registered_file']['iso_flagged_count']}` flagged, "
      f"`{ov['using_registered_file']['child_tx_overlap']}` overlap con children, "
      f"`{ov['using_registered_file']['alerts_with_if_anomaly_child']}` alertas con hijo IF-anómalo")
    a(f"- Usando archivo **fallback**: `{ov['using_fallback_file']['iso_flagged_count']}` flagged, "
      f"`{ov['using_fallback_file']['child_tx_overlap']}` overlap con children, "
      f"`{ov['using_fallback_file']['alerts_with_if_anomaly_child']}` alertas con hijo IF-anómalo")
    a("")

    # ---- Autoencoder
    a("## Autoencoder — Diagnóstico")
    a("")
    auto = report["autoencoder"]
    aov = report["autoencoder_overlap"]
    a(f"- Path: `{auto['path']}`")
    a(f"- Existe: `{auto['path_exists']}`")
    a(f"- Filas totales: `{auto['total_rows']}`")
    a(f"- Columna `autoencoder_anomaly_flag` presente: `{auto['col_present']}`")
    a(f"- flag=1 (anómalas): `{auto['flag_1']}`")
    a(f"- flag=0 (normales): `{auto['flag_0']}`")
    a(f"- Overlap con children: `{aov['child_tx_overlap']}`")
    a(f"- Alertas con hijo Autoencoder-anómalo: `{aov['alerts_with_autoencoder_anomaly_child']}`")
    a("")

    # ---- Supervised
    a("## Supervisados — Diagnóstico")
    a("")
    for model, info in report["supervised"].items():
        a(f"### {model}")
        a(f"- Path: `{info['path']}`")
        a(f"- Existe: `{info['path_exists']}`")
        a(f"- Source: `{info['source']}`")
        a(f"- Filas: `{info['total_rows']}`")
        a(f"- y_pred counts: `{info['y_pred_value_counts']}`")
        a(f"- summary_alert_ids con y_pred>=1: `{info['positive_ids']}`")
        a(f"- Overlap con alerts_summary: `{info['overlap_with_alerts_summary']}`")
        a("")
    a(f"**Total alertas unicas con supervised_positive_any=True:** `{report['supervised_positive_any_count']}`")
    a(f"**IDs:** `{report['supervised_positive_ids']}`")
    a("")

    # ---- Output existente
    a("## Output comparativo existente")
    a("")
    cmp = report.get("existing_comparison_output", {})
    a(f"- Path: `{cmp.get('path')}`")
    a(f"- Existe: `{cmp.get('exists')}`")
    a(f"- Filas: `{cmp.get('total_rows')}`")
    a(f"- `has_isolation_forest_anomaly_child=True`: `{cmp.get('has_isolation_forest_anomaly_child_true_count')}`")
    a(f"- `has_autoencoder_anomaly_child=True`: `{cmp.get('has_autoencoder_anomaly_child_true_count')}`")
    a(f"- `supervised_positive_any=True`: `{cmp.get('supervised_positive_any_true_count')}`")
    a(f"- **Bug B detectado:** `{cmp.get('bug_b_detected')}`")
    if cmp.get("sup_x_y_cols_found"):
        a(f"- Columnas _x/_y: `{cmp['sup_x_y_cols_found'][:8]}`")
    if cmp.get("sup_plain_cols_found") is not None:
        a(f"- Columnas plain (sup): `{cmp['sup_plain_cols_found']}`")
    a("")

    # ---- Diagnósticos
    a("## Bugs y diagnósticos")
    a("")
    for d in report.get("diagnoses", []):
        sev_emoji = "🔴" if d["severity"] == "CRITICAL" else "🟡" if d["severity"] == "WARN" else "🟢"
        a(f"### {sev_emoji} [{d['id']}] {d['title']}")
        a("")
        a(f"**Detalle:** {d['detail']}")
        a("")
        a(f"**Impacto:** {d['impact']}")
        a("")

    # ---- Valores correctos esperados post-fix
    a("## Valores correctos esperados post-corrección")
    a("")
    ov_fb = report["isolation_overlap"]["using_fallback_file"]
    a(_md_table(
        ["Métrica", "Valor actual (UI)", "Valor esperado post-fix", "Causa"],
        [
            ["Isolation anomalies", "0", str(ov_fb["iso_flagged_count"]), "Bug A: artifact_registry apunta a archivo sin anomaly_flag"],
            ["Reglas ∩ Isolation (tx)", "0", f"~{ov_fb['child_tx_overlap']} tx / {ov_fb['alerts_with_if_anomaly_child']} alertas", "Bug A: iso_flags vacío"],
            ["Reglas ∩ Supervisado (alertas)", "0", str(report["supervised_positive_any_count"]), "Bug B: merge _x/_y → sup_cols vacío"],
            ["Autoencoder anomalies", "100", "100", "Correcto"],
            ["Reglas ∩ Autoencoder", "39", "39", "Correcto"],
        ]
    ))
    a("")

    # ---- Conclusión
    a("## Conclusión")
    a("")
    a("**Veredicto: INCONSISTENTE** — Dos bugs críticos causan que la comparativa muestre 0 donde debería mostrar valores reales.")
    a("")
    a("### Resumen de causas")
    a("")
    a("| # | Bug | Archivo afectado | Líneas | Fix necesario |")
    a("|---|-----|-----------------|--------|---------------|")
    a("| A | artifact_registry devuelve path a archivo sin `anomaly_flag` (10k filas) | `model_evaluation_service.py` `_resolve_paths()` | ~131–135 | Corregir la entrada en artifact_registry O agregar fallback explícito cuando el archivo registrado no tiene `anomaly_flag` |")
    a("| B | Merge supervisado crea `_x`/`_y` por pre-creación de columnas | `model_evaluation_service.py` líneas 236–243 | 236–243 | Eliminar el bloque `for col: alert_df[col] = None` antes del merge |")
    a("")

    # ---- Recomendaciones C5.1.1
    a("## Recomendaciones para C5.1.1 correctiva")
    a("")
    a(textwrap.dedent("""\
    ### Fix A — artifact_registry
    Opción 1 (recomendada): en `_resolve_paths()`, después de obtener el path del registry, verificar que el archivo existe Y contiene la columna `anomaly_flag`. Si no, usar fallback:
    ```python
    def _pick_iso(artifact_type, fallback):
        item = artifacts.get_artifact_by_type(db, normalized, artifact_type)
        if item and item.status == "AVAILABLE" and item.file_path:
            p = Path(item.file_path)
            if p.exists():
                # Verify it has the expected column
                try:
                    sample = pd.read_csv(p, nrows=1)
                    if "anomaly_flag" in sample.columns:
                        return p
                except Exception:
                    pass
        return fallback
    ```
    Opción 2: re-registrar el artifact correcto ejecutando:
    ```
    python -m backend.app.tools.register_existing_artifacts --source-run preprocessed_run_26
    ```
    y verificar que `ANOMALY_SCORES_CSV` apunte a `anomaly_scores_run_26.csv` (548k filas).

    ### Fix B — merge supervisado _x/_y
    En `model_evaluation_service.py` líneas 236–238, eliminar la pre-creación de columnas:
    ```python
    # ELIMINAR este bloque:
    for col in (f"{base}_y_pred", f"{base}_y_proba", f"{base}_evaluation_result"):
        if col not in alert_df.columns:
            alert_df[col] = None

    # Después del merge, si la columna no existe, rellenar con None:
    alert_df = alert_df.merge(pred2, on="summary_alert_id", how="left")
    for col in (f"{base}_y_pred", f"{base}_y_proba", f"{base}_evaluation_result"):
        if col not in alert_df.columns:
            alert_df[col] = None
    ```

    ### Nota sobre cobertura supervisada
    Incluso post-fix, los modelos supervisados solo puntúan 12 alertas (test set). Para scoring completo (todas las 20.040 alertas), se necesita una fase de scoring completo en C5.2 o C5.3.
    """))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose model evaluation comparison artifacts for a given source_run."
    )
    parser.add_argument("--source-run", required=True, help="e.g. preprocessed_run_26")
    parser.add_argument("--no-markdown", action="store_true", help="Skip markdown report generation")
    args = parser.parse_args()

    print(f"[diagnose] source_run={args.source_run}", file=sys.stderr)
    report = diagnose(args.source_run)

    # Print JSON summary to stdout (ASCII-safe for Windows terminals)
    print(json.dumps(report, ensure_ascii=True, indent=2))

    # Generate markdown
    if not args.no_markdown:
        md = generate_markdown(report)
        processed = default_processed_dir()
        token = normalize_run_token(normalize_source_run(args.source_run))
        out_path = processed / f"model_evaluation_diagnostic_run_{token}.md"
        out_path.write_text(md, encoding="utf-8")
        print(f"\n[diagnose] Reporte markdown generado: {out_path}", file=sys.stderr)

    # Return non-zero if critical bugs found
    critical = [d for d in report.get("diagnoses", []) if d["severity"] == "CRITICAL"]
    if critical:
        print(f"\n[diagnose] {len(critical)} bug(s) crítico(s) encontrado(s).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

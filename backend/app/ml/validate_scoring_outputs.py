from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

VERDICT_READY   = "SCORING_OUTPUTS_READY"
VERDICT_INVALID = "SCORING_OUTPUTS_INVALID"

REQUIRED_COLUMNS = [
    "source_run",
    "summary_alert_id",
    "customer_hash",
    "ml_risk_score",
    "ml_risk_level",
    "algorithm",
    "scored_at",
]

FORBIDDEN_COLUMNS = {
    "is_fraud",
    "confirmed_fraud",
    "PAN_TARJETA",
    "TARJETA",
    "pan_card",
    "raw_card",
    "target_human_label",
    "target_label_source",
}

VALID_RISK_LEVELS = {"HIGH", "MEDIUM", "LOW"}
VALID_ALGORITHMS  = {"logistic_regression", "random_forest", "gradient_boosting"}


def validate_scoring_outputs(
    results_file: str | Path,
    metadata_file: str | Path | None = None,
) -> dict[str, Any]:
    results_path = Path(results_file)
    errors: list[str] = []

    if not results_path.exists():
        return {
            "verdict": VERDICT_INVALID,
            "status": "INVALID",
            "results_file": str(results_path),
            "row_count": 0,
            "errors": [f"results_file no existe: {results_path}"],
        }

    try:
        df = pd.read_csv(results_path)
    except Exception as exc:
        return {
            "verdict": VERDICT_INVALID,
            "status": "INVALID",
            "results_file": str(results_path),
            "row_count": 0,
            "errors": [f"Error leyendo results_file: {exc}"],
        }

    row_count = len(df)
    cols = set(df.columns.tolist())

    for req in REQUIRED_COLUMNS:
        if req not in cols:
            errors.append(f"Columna requerida ausente: {req}")

    for forbidden in FORBIDDEN_COLUMNS:
        if forbidden in cols:
            errors.append(f"Columna prohibida presente: {forbidden}")

    if "ml_risk_score" in cols:
        try:
            scores = pd.to_numeric(df["ml_risk_score"], errors="coerce")
            if scores.isna().any():
                errors.append("ml_risk_score contiene valores NaN o no numéricos")
            elif not scores.between(0.0, 1.0).all():
                bad = scores[(scores < 0) | (scores > 1)].count()
                errors.append(f"ml_risk_score tiene {bad} valores fuera de [0.0, 1.0]")
        except Exception as exc:
            errors.append(f"Error validando ml_risk_score: {exc}")

    if "ml_risk_level" in cols:
        invalid_levels = set(df["ml_risk_level"].dropna().unique()) - VALID_RISK_LEVELS
        if invalid_levels:
            errors.append(f"ml_risk_level contiene valores inválidos: {sorted(invalid_levels)}")

    if "algorithm" in cols:
        invalid_algos = set(df["algorithm"].dropna().unique()) - VALID_ALGORITHMS
        if invalid_algos:
            errors.append(f"algorithm contiene valores inválidos: {sorted(invalid_algos)}")

    if "summary_alert_id" in cols:
        dup_count = int(df["summary_alert_id"].duplicated().sum())
        if dup_count > 0:
            errors.append(f"summary_alert_id tiene {dup_count} duplicados")

    metadata_errors: list[str] = []
    if metadata_file is not None:
        meta_path = Path(metadata_file)
        if not meta_path.exists():
            metadata_errors.append(f"metadata_file no existe: {meta_path}")
        else:
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                for key in ("algorithm", "total_scored", "source_run"):
                    if key not in meta:
                        metadata_errors.append(f"metadata_file falta campo: {key}")
                if "total_scored" in meta and meta["total_scored"] != row_count:
                    metadata_errors.append(
                        f"total_scored en metadata ({meta['total_scored']}) "
                        f"no coincide con filas en CSV ({row_count})"
                    )
            except Exception as exc:
                metadata_errors.append(f"Error leyendo metadata_file: {exc}")

    all_errors = errors + metadata_errors
    verdict = VERDICT_READY if not all_errors else VERDICT_INVALID

    return {
        "verdict": verdict,
        "status": "VALID" if verdict == VERDICT_READY else "INVALID",
        "results_file": str(results_path),
        "row_count": row_count,
        "errors": all_errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Valida los archivos de salida del scoring por lotes D1."
    )
    parser.add_argument("--results-file", required=True, help="CSV de resultados de scoring")
    parser.add_argument("--metadata-file", default=None, help="JSON de metadata del scoring")
    args = parser.parse_args()

    result = validate_scoring_outputs(args.results_file, args.metadata_file)
    print(json.dumps(result, ensure_ascii=True, indent=2))

    if result["verdict"] != VERDICT_READY:
        import sys
        sys.exit(1)


if __name__ == "__main__":
    main()

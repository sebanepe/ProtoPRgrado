from __future__ import annotations

import argparse
import json
import sys

from backend.app.database import SessionLocal
from backend.app.ml.batch_scoring_service import VALID_ALGORITHMS, run_batch_scoring


def main() -> None:
    parser = argparse.ArgumentParser(
        description="D1 batch scoring: aplica un modelo supervisado sobre alertas para generar predicciones de riesgo."
    )
    parser.add_argument(
        "--source-run",
        required=True,
        help="Run fuente, e.g. preprocessed_run_26",
    )
    parser.add_argument(
        "--algorithm",
        required=True,
        choices=sorted(VALID_ALGORITHMS),
        help="Algoritmo supervisado a usar",
    )
    parser.add_argument(
        "--input-dataset",
        default=None,
        dest="input_dataset",
        help="Path alternativo al supervised_human_alert_dataset_run_N.csv",
    )
    args = parser.parse_args()

    session = SessionLocal()
    try:
        result = run_batch_scoring(
            args.source_run,
            args.algorithm,
            db=session,
            input_dataset_path=args.input_dataset,
        )
    finally:
        session.close()

    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))

    if result.get("status") != "COMPLETED":
        sys.exit(1)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json

from backend.app.database import SessionLocal
from backend.app.ml.model_evaluation_service import build_model_evaluation_comparison


def main() -> int:
    parser = argparse.ArgumentParser(description="Build model evaluation comparison artifacts")
    parser.add_argument("--source-run", required=True)
    args = parser.parse_args()
    db = SessionLocal()
    try:
        result = build_model_evaluation_comparison(db, args.source_run)
        print(json.dumps({"status": "ok", "summary": result}, ensure_ascii=False, indent=2))
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())

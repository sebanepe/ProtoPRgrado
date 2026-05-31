from __future__ import annotations

import argparse

from backend.app.database import SessionLocal
from backend.app.init_db import ensure_traceability_tables
from backend.app.services import artifact_registry_service, model_registry_service, rule_run_service


def main() -> None:
    parser = argparse.ArgumentParser(description="Register existing run artifacts without modifying artifact files.")
    parser.add_argument("--source-run", required=True, help="Example: preprocessed_run_26")
    args = parser.parse_args()

    ensure_traceability_tables()
    db = SessionLocal()
    try:
        scan = artifact_registry_service.scan_existing_artifacts(db, args.source_run)
        rule_run = rule_run_service.register_rule_run_from_artifacts(db, args.source_run)
        model = model_registry_service.register_unsupervised_model_from_artifacts(db, args.source_run)

        print(f"source_run={scan['source_run']}")
        print(f"run_token={scan['run_token']}")
        print(f"registered_count={scan['registered_count']}")
        print(f"missing_count={scan['missing_count']}")
        print("registered_artifacts:")
        for item in scan["registered"]:
            print(
                f"- {item['artifact_type']} {item['file_name']} "
                f"rows={item['row_count']} bytes={item['file_size_bytes']} checksum={item['checksum']}"
            )
        print("missing_artifacts:")
        for item in scan["missing"]:
            print(f"- {item['artifact_type']} {item['file_path']}")
        print(
            "rule_run="
            f"{rule_run.source_run} detailed={rule_run.detailed_alert_count} grouped={rule_run.grouped_alert_count} status={rule_run.status}"
        )
        print(
            "model_registry="
            f"{model.source_run} family={model.model_family} algorithm={model.algorithm} active={model.is_active} status={model.status}"
        )
        if scan["warnings"]:
            print("warnings:")
            for warning in scan["warnings"]:
                print(f"- {warning}")
    finally:
        db.close()


if __name__ == "__main__":
    main()

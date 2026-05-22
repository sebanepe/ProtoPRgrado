import os
from datetime import datetime, timezone
from backend.app.ml.preprocessing import fetch_transactions_df, preprocess_dataframe, save_processed
from sqlalchemy.orm import Session
from backend.app.models.models import PreprocessingRun
from backend.app.models.models import FeatureSet
import json

# persistent processed folder inside project for training/artifacts (absolute path)
PROJECT_PROCESSED_DIR = r"C:\Users\seban\Documents\GitHub\Sistema-GIS-La-Paz-Microservicios\ProtoPRgrado\data\processed"


DEFAULT_OUTPUT = os.path.join("data", "processed", "preprocessed_transactions.csv")


def run_preprocessing(db: Session, output_path: str | None = None, apply_smote: bool = True):
    """Run preprocessing over available transactions and record a PreprocessingRun.

    This function creates a PreprocessingRun DB row (status tracking), executes the
    preprocessing pipeline, saves results to disk and updates the run record.
    """
    output = output_path or DEFAULT_OUTPUT

    # create run record
    run = PreprocessingRun(
        status="RUNNING",
        started_at=datetime.now(timezone.utc),
        params_json=str({"apply_smote": bool(apply_smote)}),
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    try:
        df = fetch_transactions_df(db)
        processed, summary = preprocess_dataframe(df, apply_smote=apply_smote)
        if not processed.empty:
            # save default output (existing behavior)
            save_processed(processed, output)
            summary["output_path"] = output
            # also save a copy into project processed folder for training and use it as canonical output
            os.makedirs(PROJECT_PROCESSED_DIR, exist_ok=True)
            project_path = os.path.join(PROJECT_PROCESSED_DIR, f"preprocessed_run_{run.id}.csv")
            save_processed(processed, project_path)
            summary["project_output_path"] = project_path
            # prefer the project-scoped path as the run's output file
            summary["output_path"] = project_path
            # create FeatureSet record in DB for traceability
            try:
                feature_cols = list(processed.columns)
                fs = FeatureSet(
                    dataset_id=None,
                    preprocessing_run_id=run.id,
                    name=f"preprocessed_run_{run.id}",
                    file_path=project_path,
                    row_count=len(processed),
                    feature_columns_json=json.dumps(feature_cols),
                )
                db.add(fs)
                db.commit()
                db.refresh(fs)
                summary["feature_set_id"] = fs.id
            except Exception:
                # do not fail the whole run if FeatureSet creation fails
                pass
        else:
            summary["output_path"] = None

        # update run record with summary (ensure output_file_path points to project file when available)
        run.output_file_path = summary.get("output_path")
        run.total_records = summary.get("before", 0)
        run.processed_records = summary.get("after_clean", 0) or summary.get("after", 0) or 0
        run.removed_records = (run.total_records or 0) - (run.processed_records or 0)
        run.status = "COMPLETED"
        run.finished_at = datetime.now(timezone.utc)
        db.add(run)
        db.commit()
        db.refresh(run)
        return summary
    except Exception as e:
        run.status = "FAILED"
        run.error_message = str(e)
        run.finished_at = datetime.now(timezone.utc)
        db.add(run)
        db.commit()
        raise

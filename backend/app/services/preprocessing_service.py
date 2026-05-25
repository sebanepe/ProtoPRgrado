import os
from datetime import datetime, timezone
from backend.app.ml.preprocessing import fetch_transactions_df, preprocess_dataframe, save_processed
from sqlalchemy.orm import Session
from backend.app.models.models import PreprocessingRun
from backend.app.models.models import FeatureSet
import json
import pandas as pd
from backend.app.models.models import Dataset

# persistent processed folder inside project for training/artifacts (absolute path)
PROJECT_PROCESSED_DIR = r"C:\Users\seban\Documents\GitHub\Sistema-GIS-La-Paz-Microservicios\ProtoPRgrado\data\processed"


DEFAULT_OUTPUT = os.path.join("data", "processed", "preprocessed_transactions.csv")


def run_preprocessing(db: Session, output_path: str | None = None, apply_smote: bool = True, dataset_id: int | None = None):
    """Run preprocessing over available transactions and record a PreprocessingRun.

    This function creates a PreprocessingRun DB row (status tracking), executes the
    preprocessing pipeline, saves results to disk and updates the run record.
    """
    output = output_path or DEFAULT_OUTPUT

    # create run record
    run = PreprocessingRun(
        status="RUNNING",
        started_at=datetime.now(timezone.utc),
        params_json=str({"apply_smote": bool(apply_smote), "dataset_id": dataset_id}),
        input_dataset_id=dataset_id,
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    try:
        df = fetch_transactions_df(db, dataset_id=dataset_id)

        # If dataset_id provided but no transactions found, attempt to load original dataset CSV
        if (df is None or df.empty) and dataset_id is not None:
            ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
            if ds and ds.file_path and os.path.exists(ds.file_path):
                try:
                    df_file = pd.read_csv(ds.file_path)
                    # Let preprocess_dataframe normalize & validate columns
                    df = df_file
                except Exception:
                    # ignore and continue with empty df
                    df = df
        cleaned_df, summary = preprocess_dataframe(df)

        if cleaned_df is not None and not cleaned_df.empty:
            # save cleaned dataset (FASE A)
            save_processed(cleaned_df, output)
            summary["output_path"] = output

            # also save canonical cleaned dataset into project processed folder
            os.makedirs(PROJECT_PROCESSED_DIR, exist_ok=True)
            project_path = os.path.join(PROJECT_PROCESSED_DIR, f"cleaned_dataset_run_{run.id}.csv")
            save_processed(cleaned_df, project_path)
            summary["project_output_path"] = project_path
            # create a training_dataset copy (without sensitive columns, no OneHot, no SMOTE)
            training_path = os.path.join(PROJECT_PROCESSED_DIR, f"training_dataset_run_{run.id}.csv")
            try:
                # remove sensitive columns before saving training dataset
                cleaned_nosensitive = cleaned_df.copy()
                sensitive = ["pan_card", "masked_card", "tarjeta", "pan_tarjeta", "numero_cuenta", "documento_identidad"]
                cols_to_drop = [c for c in cleaned_nosensitive.columns if c.lower() in sensitive]
                cleaned_nosensitive = cleaned_nosensitive.drop(columns=cols_to_drop, errors="ignore")
                save_processed(cleaned_nosensitive, training_path)
                summary["training_dataset_path"] = training_path
            except Exception:
                pass

            # prefer the project-scoped cleaned path as the run's output file
            summary["output_path"] = project_path
        else:
            summary["output_path"] = None

        # update run record with summary
        run.output_file_path = summary.get("output_path")
        run.total_records = summary.get("before", 0)
        run.processed_records = summary.get("after_clean", 0) or 0
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


def rerun_preprocessing(db: Session, run_id: int):
    """Re-execute preprocessing updating an existing PreprocessingRun row instead of creating a new one."""
    run = db.query(PreprocessingRun).filter(PreprocessingRun.id == run_id).first()
    if not run:
        raise ValueError("run not found")

    # mark running
    run.status = "RUNNING"
    run.started_at = datetime.now(timezone.utc)
    db.add(run)
    db.commit()
    db.refresh(run)

    try:
        dataset_id = run.input_dataset_id
        df = fetch_transactions_df(db, dataset_id=dataset_id)

        # fallback to dataset CSV if no transactions found
        if (df is None or df.empty) and dataset_id is not None:
            ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
            if ds and ds.file_path and os.path.exists(ds.file_path):
                try:
                    df = pd.read_csv(ds.file_path)
                    # Try to normalize mixed timezones if transaction_datetime exists
                    if "transaction_datetime" in df.columns:
                        try:
                            # remove explicit timezone suffixes to avoid mixed tz parsing issues
                            s = df["transaction_datetime"].astype(str).fillna("")
                            s = s.str.replace(r"(\+|-)\d{2}:?\d{2}$|Z$", "", regex=True)
                            try:
                                ts = pd.to_datetime(s, errors="coerce", utc=True)
                                if hasattr(ts.dt, "tz"):
                                    df["transaction_datetime"] = ts.dt.tz_convert("UTC").dt.tz_localize(None)
                                else:
                                    df["transaction_datetime"] = ts
                            except Exception:
                                df["transaction_datetime"] = pd.to_datetime(s, errors="coerce")
                        except Exception:
                            try:
                                df["transaction_datetime"] = pd.to_datetime(df["transaction_datetime"], errors="coerce")
                            except Exception:
                                pass
                except Exception:
                    df = df

        cleaned_df, summary = preprocess_dataframe(df)

        if cleaned_df is not None and not cleaned_df.empty:
            output = DEFAULT_OUTPUT
            save_processed(cleaned_df, output)
            summary["output_path"] = output

            os.makedirs(PROJECT_PROCESSED_DIR, exist_ok=True)
            project_path = os.path.join(PROJECT_PROCESSED_DIR, f"cleaned_dataset_run_{run.id}.csv")
            save_processed(cleaned_df, project_path)
            summary["project_output_path"] = project_path

            training_path = os.path.join(PROJECT_PROCESSED_DIR, f"training_dataset_run_{run.id}.csv")
            try:
                cleaned_nosensitive = cleaned_df.copy()
                sensitive = ["pan_card", "masked_card", "tarjeta", "pan_tarjeta", "numero_cuenta", "documento_identidad"]
                cols_to_drop = [c for c in cleaned_nosensitive.columns if c.lower() in sensitive]
                cleaned_nosensitive = cleaned_nosensitive.drop(columns=cols_to_drop, errors="ignore")
                save_processed(cleaned_nosensitive, training_path)
                summary["training_dataset_path"] = training_path
            except Exception:
                pass

            summary["output_path"] = project_path
        else:
            summary["output_path"] = None

        run.output_file_path = summary.get("output_path")
        run.total_records = summary.get("before", 0)
        run.processed_records = summary.get("after_clean", 0) or 0
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


def delete_preprocessing_run(db: Session, run_id: int):
    """Delete a PreprocessingRun record and any associated output files.

    This will remove the files saved under the project processed directory
    (cleaned/training copies) and delete the DB row. It is idempotent: if
    the run doesn't exist, raises ValueError; if files are missing, it
    ignores file-not-found errors.
    """
    run = db.query(PreprocessingRun).filter(PreprocessingRun.id == run_id).first()
    if not run:
        raise ValueError("run not found")

    # attempt to delete output file and training file (if present)
    paths = []
    if run.output_file_path:
        paths.append(run.output_file_path)

    # also attempt canonical project files
    project_clean = os.path.join(PROJECT_PROCESSED_DIR, f"cleaned_dataset_run_{run.id}.csv")
    project_training = os.path.join(PROJECT_PROCESSED_DIR, f"training_dataset_run_{run.id}.csv")
    paths.extend([project_clean, project_training])

    for p in paths:
        try:
            if p and os.path.exists(p):
                os.remove(p)
        except Exception:
            # ignore removal errors
            pass

    # delete DB row
    try:
        db.delete(run)
        db.commit()
    except Exception as e:
        db.rollback()
        raise

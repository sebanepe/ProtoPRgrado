import os
from datetime import datetime, timezone
from backend.app.ml.preprocessing import fetch_transactions_df, preprocess_dataframe, save_processed
from sqlalchemy.orm import Session
from backend.app.models.models import PreprocessingRun
from backend.app.models.models import FeatureSet
import json
import pandas as pd
from backend.app.models.models import Dataset
from backend.app.database import SessionLocal

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


def create_run(db: Session, dataset_id: int | None = None, apply_smote: bool = True) -> PreprocessingRun:
    """Create a PreprocessingRun row in PENDING state and return it."""
    run = PreprocessingRun(
        status="PENDING",
        started_at=None,
        params_json=str({"apply_smote": bool(apply_smote), "dataset_id": dataset_id}),
        input_dataset_id=dataset_id,
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


def run_preprocessing_background(run_id: int):
    """Background runner that creates its own DB session and executes the preprocessing run.

    This is safe to enqueue via FastAPI BackgroundTasks and will call `rerun_preprocessing`
    which updates the PreprocessingRun row in-place.
    """
    db = SessionLocal()
    try:
        # reuse existing rerun logic which marks the run RUNNING and performs processing
        rerun_preprocessing(db, run_id=run_id)
    except Exception:
        # rerun_preprocessing already persists failure state; ensure session closed
        db.rollback()
    finally:
        db.close()


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


def run_preprocessing_for_training(db: Session, run_id: int | None = None, training_dataset_path: str | None = None, apply_smote: bool = True):
    """Run Phase B preprocessing for training: load a processed CSV (or run id),
    select features, split train/test, build pipeline, optionally apply SMOTE,
    and persist a FeatureSet record.
    """
    import json
    from backend.app.ml import preprocessing as mlp
    from backend.app.ml.build_training_dataset import build_training_dataset

    # resolve dataset path
    resolved_path = None
    if training_dataset_path:
        resolved_path = training_dataset_path
    elif run_id is not None:
        # prefer training_dataset_run_{id}.csv then cleaned_dataset_run_{id}.csv
        tpath = os.path.join(PROJECT_PROCESSED_DIR, f"training_dataset_run_{run_id}.csv")
        cpath = os.path.join(PROJECT_PROCESSED_DIR, f"cleaned_dataset_run_{run_id}.csv")
        if os.path.exists(tpath):
            resolved_path = tpath
        elif os.path.exists(cpath):
            resolved_path = cpath

    if not resolved_path or not os.path.exists(resolved_path):
        raise ValueError("training dataset file not found")

    # Use build_training_dataset to produce feature set CSV and preprocessing report
    feature_set_path = None
    preprocessing_report_path = None
    try:
        feature_set_path, preprocessing_report_path = build_training_dataset(resolved_path, out_name=f"training_dataset_{run_id or 'manual'}")
    except Exception:
        # fallback to using resolved_path as feature set if build fails
        feature_set_path = resolved_path

    # Validate generated feature set before proceeding to training
    try:
        from backend.app.ml import validate_feature_set
        vreport = validate_feature_set.validate(feature_set_path)
        if vreport.get("verdict") != "READY_FOR_SUPERVISED_TRAINING":
            # return a clear report instead of attempting to train on an invalid feature set
            return {
                "feature_set_id": None,
                "feature_set_path": feature_set_path,
                "preprocessing_report_path": preprocessing_report_path,
                "ready": False,
                "validation": vreport,
            }
    except Exception:
        # if validation fails unexpectedly, abort to avoid training on bad data
        raise

    # load the generated feature set (or resolved_path)
    df_fs = pd.read_csv(feature_set_path)

    # get X, y according to Phase B rules
    X, y = mlp.get_training_columns(df_fs)

    # split
    X_train, X_test, y_train, y_test = mlp.split_train_test(X, y)

    # build preprocessing pipeline based on X (use full X to detect types)
    preprocessor, pipeline = mlp.build_preprocessing_pipeline(X)

    # apply SMOTE only on X_train/y_train if requested
    smote_report = None
    X_train_res, y_train_res = X_train, y_train
    if apply_smote:
        X_train_res, y_train_res, smote_report = mlp.apply_smote_if_needed(X_train, y_train)

    # Fit the pipeline BEFORE persisting metadata to DB to avoid holding DB transactions
    pipeline_path = None
    try:
        try:
            pipeline.fit(X_train_res)
        except Exception:
            pipeline.fit(X)
        os.makedirs(PROJECT_PROCESSED_DIR, exist_ok=True)
    except Exception:
        # fitting failed but we'll still persist metadata without pipeline
        pass

    # persist a FeatureSet record (we store metadata in feature_columns_json as a dict)
    feature_cols = list(X.columns)
    excluded = [c for c in df_fs.columns if c not in feature_cols]

    # create initial FeatureSet row using a new short-lived DB session
    local_db = SessionLocal()
    try:
        fs = FeatureSet(
            dataset_id=None,
            preprocessing_run_id=run_id,
            name=f"feature_set_{run_id or 'manual'}_{int(datetime.now().timestamp())}",
            file_path=feature_set_path,
            row_count=len(X),
            feature_columns_json=json.dumps({"features": feature_cols}),
            excluded_columns_json=json.dumps(excluded),
        )
        local_db.add(fs)
        local_db.commit()
        local_db.refresh(fs)

        # persist fitted pipeline to disk referencing fs.id (if fit succeeded)
        try:
            import pickle
            pipeline_path = os.path.join(PROJECT_PROCESSED_DIR, f"pipeline_feature_set_{fs.id}.pkl")
            with open(pipeline_path, "wb") as f:
                pickle.dump(pipeline, f)
            fs.pipeline_path = pipeline_path
        except Exception:
            fs.pipeline_path = None

        fs.smote_report_json = json.dumps(smote_report) if smote_report is not None else None
        # keep legacy metadata too
        meta = {"features": feature_cols}
        fs.feature_columns_json = json.dumps(meta)
        local_db.add(fs)
        local_db.commit()
        local_db.refresh(fs)
    finally:
        local_db.close()

    report = {
        "feature_set_id": fs.id,
        "feature_count": len(feature_cols),
        "row_count": len(X),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "smote_report": smote_report,
        "pipeline_path": pipeline_path,
        "feature_set_path": feature_set_path,
        "preprocessing_report_path": preprocessing_report_path,
    }
    return report

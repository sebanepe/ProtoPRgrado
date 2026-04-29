import os
from backend.app.ml.preprocessing import fetch_transactions_df, preprocess_dataframe, save_processed
from sqlalchemy.orm import Session


DEFAULT_OUTPUT = os.path.join("data", "processed", "preprocessed_transactions.csv")


def run_preprocessing(db: Session, output_path: str | None = None, apply_smote: bool = True):
    output = output_path or DEFAULT_OUTPUT
    df = fetch_transactions_df(db)
    processed, summary = preprocess_dataframe(df, apply_smote=apply_smote)
    if not processed.empty:
        save_processed(processed, output)
        summary["output_path"] = output
    else:
        summary["output_path"] = None
    return summary

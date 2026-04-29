import pandas as pd
from typing import BinaryIO, List, Dict
from backend.app.repositories import transaction_repository, dataset_repository
from sqlalchemy.orm import Session
from io import BytesIO


REQUIRED_COLUMNS = {
    "transaction_id",
    "amount",
    "transaction_type",
    "channel",
    "location",
    "device_id",
    "customer_hash",
    "transaction_datetime",
    "is_fraud",
}


def _read_csv(file: BinaryIO) -> pd.DataFrame:
    file.seek(0)
    try:
        df = pd.read_csv(file)
    except Exception:
        # try bytes buffer
        file.seek(0)
        df = pd.read_csv(BytesIO(file.read()))
    return df


def import_dataset(db: Session, file, name: str, file_name: str):
    df = _read_csv(file)
    total = len(df)
    if total == 0:
        raise ValueError("Uploaded file is empty")

    missing = REQUIRED_COLUMNS - set(df.columns.astype(str))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    # Normalize columns
    df = df.loc[:, list(REQUIRED_COLUMNS)]

    # Parse datetimes
    df["transaction_datetime"] = pd.to_datetime(df["transaction_datetime"], errors="coerce")

    # Convert amount
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

    # Normalize is_fraud
    df["is_fraud"] = df["is_fraud"].astype(bool)

    # Valid rows: transaction_id not null, amount not null, transaction_datetime not null
    valid_mask = df["transaction_id"].notna() & df["amount"].notna() & df["transaction_datetime"].notna()
    valid_df = df[valid_mask]
    invalid_df = df[~valid_mask]

    valid_records = []
    for _, row in valid_df.iterrows():
        rec = {
            "transaction_id": str(row["transaction_id"]),
            "amount": float(row["amount"]),
            "transaction_type": row.get("transaction_type"),
            "channel": row.get("channel"),
            "location": row.get("location"),
            "device_id": row.get("device_id"),
            "customer_hash": row.get("customer_hash"),
            "transaction_datetime": row["transaction_datetime"].to_pydatetime(),
            "is_fraud": bool(row["is_fraud"]),
        }
        valid_records.append(rec)

    num_inserted = 0
    if valid_records:
        num_inserted = transaction_repository.insert_transactions(db, valid_records)

    dataset = dataset_repository.create_dataset(
        db,
        name=name,
        file_name=file_name,
        total_records=total,
        valid_records=len(valid_records),
        invalid_records=len(invalid_df),
        status="imported",
    )

    return {"dataset": dataset, "inserted": num_inserted, "total": total, "valid": len(valid_records), "invalid": len(invalid_df)}

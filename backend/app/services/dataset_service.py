import pandas as pd
from typing import BinaryIO, List, Dict
from backend.app.repositories import transaction_repository, dataset_repository
from sqlalchemy.orm import Session
from io import BytesIO
import os
from datetime import datetime
import unicodedata


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


COLUMN_ALIASES = {
    "transaction_id": ["transaction_id", "trx", "numero_referencia", "nro_referencia", "referencia", "id_transaccion", "trans_id", "transactionid", "referencia_transaccion"],
    "amount": ["amount", "monto", "importe", "valor", "transaction_amount"],
    "transaction_type": ["transaction_type", "tipo_transaccion", "categoria_trans", "categoria", "tipo"],
    "channel": ["channel", "canal", "channel_type"],
    "location": ["location", "establecimiento", "codigo_establecimiento", "establishment", "ciudad", "sucursal", "branch"],
    "device_id": ["device_id", "codigo_terminal", "terminal", "terminal_id", "terminal_code"],
    "customer_hash": ["customer_hash", "pan_tarjeta", "card_pan", "pan", "card_number", "tarjeta"],
    "transaction_datetime": ["transaction_datetime", "fecha", "date", "fecha_transaccion", "fecha_trans", "fecha_operacion", "datetime", "timestamp", "hora"],
    "is_fraud": ["is_fraud", "fraud", "es_fraude", "fraude", "label", "isfraud"],
}


def _map_columns(df):
    # Return a dict mapping expected column -> actual df column name (or None)
    cols = {c: c for c in df.columns}
    # build normalized name mapping: normalized -> original
    def _normalize(s: str):
        if not isinstance(s, str):
            s = str(s)
        s = unicodedata.normalize('NFKD', s)
        s = ''.join(ch for ch in s if not unicodedata.combining(ch))
        s = s.lower()
        # replace non-alphanumeric with underscore
        import re
        s = re.sub(r'[^a-z0-9]+', '_', s).strip('_')
        return s

    normalized = {orig: _normalize(orig) for orig in df.columns}
    # reverse map
    norm_to_orig = {v: k for k, v in normalized.items()}
    found = {}
    # helper to find by exact lowered match or contains
    for expected, aliases in COLUMN_ALIASES.items():
        found_col = None
        # normalize aliases and try exact match against normalized names
        for a in aliases:
            na = _normalize(a)
            if na in norm_to_orig:
                found_col = norm_to_orig[na]
                break
        if not found_col:
            # try substring match on normalized names
            for a in aliases:
                na = _normalize(a)
                for norm, orig in norm_to_orig.items():
                    if na in norm:
                        found_col = orig
                        break
                if found_col:
                    break
        found[expected] = found_col

    # special handling: if fecha + hora present, combine into transaction_datetime
    if (not found.get("transaction_datetime")):
        date_col = None
        time_col = None
        for orig, norm in normalized.items():
            if norm in ("fecha", "date", "fecha_transaccion", "fecha_operacion", "fecha_trans", "fecha_operacion") or "fecha" in norm or "date" in norm:
                date_col = orig
            if norm in ("hora", "time", "tiempo") or "hora" in norm or "time" in norm or "hora" in norm:
                time_col = orig
        if date_col:
            if time_col:
                found["transaction_datetime"] = (date_col, time_col)
            else:
                found["transaction_datetime"] = date_col

    return found


def _read_csv(file: BinaryIO) -> pd.DataFrame:
    file.seek(0)
    try:
        df = pd.read_csv(file)
    except pd.errors.EmptyDataError:
        raise ValueError("Uploaded file is empty")
    except Exception:
        # try bytes buffer
        file.seek(0)
        try:
            df = pd.read_csv(BytesIO(file.read()))
        except pd.errors.EmptyDataError:
            raise ValueError("Uploaded file is empty")
    return df


def import_dataset(db: Session, file, name: str, file_name: str):
    df = _read_csv(file)
    total = len(df)
    if total == 0:
        raise ValueError("Uploaded file is empty")

    # Attempt to map incoming columns to expected names (case-insensitive, common Spanish aliases)
    mapped = _map_columns(df)

    # build rename mapping
    rename_map = {}
    for expected, actual in mapped.items():
        if actual is None:
            continue
        # if datetime was detected as tuple (date,time) we will handle below
        if expected == "transaction_datetime" and isinstance(actual, tuple):
            continue
        if actual != expected:
            rename_map[actual] = expected

    if rename_map:
        df = df.rename(columns=rename_map)

    # handle combined date+time into transaction_datetime
    td = mapped.get("transaction_datetime")
    if isinstance(td, tuple):
        date_col, time_col = td
        # create a combined column
        df["transaction_datetime"] = df[date_col].astype(str).str.strip() + ' ' + df[time_col].astype(str).str.strip()
    # normalize column set after renaming
    available = set(df.columns.astype(str))
    missing = REQUIRED_COLUMNS - available
    # allow missing is_fraud by default (unlabeled raw data). Fill with False
    if 'is_fraud' in missing:
        df['is_fraud'] = False
        missing.remove('is_fraud')

    if missing:
        # build diagnostic mapping info
        try:
            detected = {k: (v if not isinstance(v, tuple) else v) for k, v in mapped.items()}
        except Exception:
            detected = mapped
        norm_cols = list(sorted(normalized.values())) if 'normalized' in locals() else []
        msg = f"Missing required columns: {', '.join(sorted(missing))}. Available columns: {', '.join(sorted(available))}. Detected mappings: {detected}. Normalized column names: {norm_cols}"
        raise ValueError(msg)

    # Normalize columns
    df = df.loc[:, list(REQUIRED_COLUMNS)]

    # Parse datetimes: allow mixed timezones by parsing with utc=True and then
    # converting to naive UTC datetimes for storage.
    try:
        parsed_dt = pd.to_datetime(df["transaction_datetime"], errors="coerce", utc=True)
        # convert to UTC and drop tzinfo to store naive UTC datetimes
        parsed_dt = parsed_dt.dt.tz_convert('UTC').dt.tz_localize(None)
    except Exception:
        # fallback to naive parsing if something unexpected happens
        try:
            parsed_dt = pd.to_datetime(df["transaction_datetime"], errors="coerce", utc=True)
            if hasattr(parsed_dt.dt, "tz"):
                parsed_dt = parsed_dt.dt.tz_convert("UTC").dt.tz_localize(None)
        except Exception:
            s = df["transaction_datetime"].astype(str).fillna("")
            s = s.str.replace(r"(\+|-)\d{2}:?\d{2}$|Z$", "", regex=True)
            parsed_dt = pd.to_datetime(s, errors="coerce")
    df["transaction_datetime"] = parsed_dt

    # Clean and convert amount: handle comma decimals, remove quotes/spaces and non-numeric chars
    df["amount"] = df["amount"].astype(str).str.replace(r'["\s]', '', regex=True)
    df["amount"] = df["amount"].str.replace(',', '.', regex=False)
    df["amount"] = df["amount"].str.replace(r'[^0-9\.\-]', '', regex=True)
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

    # Persist original uploaded file to storage for raw access in preprocessing
    storage_dir = os.environ.get('DATASET_STORAGE', '/tmp/datasets')
    try:
        os.makedirs(storage_dir, exist_ok=True)
    except Exception:
        pass
    timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
    safe_name = f"{timestamp}_{os.path.basename(file_name)}"
    dest_path = os.path.join(storage_dir, safe_name)
    # write file bytes to disk
    try:
        file.seek(0)
        with open(dest_path, 'wb') as out:
            out.write(file.read())
    except Exception:
        dest_path = None

    # Create dataset record first so we can associate inserted transactions
    dataset = dataset_repository.create_dataset(
        db,
        name=name,
        file_name=file_name,
        file_path=dest_path,
        original_filename=file_name,
        total_records=total,
        valid_records=0,
        invalid_records=len(invalid_df),
        status="importing",
    )

    if valid_records:
        num_inserted = transaction_repository.insert_transactions(db, valid_records, dataset_id=dataset.id)

    # Update dataset with accurate counts and mark imported
    try:
        dataset.total_records = total
        dataset.valid_records = num_inserted
        dataset.invalid_records = len(invalid_df)
        dataset.status = "imported"
        dataset.file_path = dest_path
        db.add(dataset)
        db.commit()
    except Exception:
        db.rollback()

    return {"dataset": dataset, "inserted": num_inserted, "total": total, "valid": num_inserted, "invalid": len(invalid_df)}

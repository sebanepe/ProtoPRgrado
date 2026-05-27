import pandas as pd
import logging
import time
import json
import uuid
from typing import BinaryIO, List, Dict, Optional
from backend.app.repositories import transaction_repository, dataset_repository
from sqlalchemy.orm import Session
from sqlalchemy.orm import sessionmaker
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
    # Persist original uploaded file to storage for raw access first (avoid holding whole file in memory)
    storage_dir = os.environ.get('DATASET_STORAGE', os.path.join(os.getcwd(), 'data', 'uploads'))
    try:
        os.makedirs(storage_dir, exist_ok=True)
    except Exception:
        pass
    timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
    safe_name = f"{timestamp}_{os.path.basename(file_name)}"
    dest_path = os.path.join(storage_dir, safe_name)
    try:
        file.seek(0)
        with open(dest_path, 'wb') as out:
            # stream write to avoid reading all into memory
            chunk = file.read(1024 * 1024)
            while chunk:
                out.write(chunk)
                chunk = file.read(1024 * 1024)
    except Exception as e:
        raise ValueError(f"Failed saving uploaded file: {e}")

    # Read a small sample to infer columns and mappings
    try:
        sample = pd.read_csv(dest_path, nrows=5)
    except Exception as e:
        raise ValueError(f"Failed reading uploaded CSV sample: {e}")

    mapped = _map_columns(sample)

    # build rename mapping based on detected mappings (ignore datetime tuple for now)
    rename_map = {}
    for expected, actual in mapped.items():
        if actual is None:
            continue
        if expected == 'transaction_datetime' and isinstance(actual, tuple):
            continue
        if actual != expected:
            rename_map[actual] = expected

    # create dataset record (status importing) before ingesting rows
    # we will update counts as we process chunks
    total = 0
    num_inserted = 0
    invalid_count = 0

    dataset = dataset_repository.create_dataset(
        db,
        name=name,
        file_name=file_name,
        file_path=dest_path,
        original_filename=file_name,
        total_records=0,
        valid_records=0,
        invalid_records=0,
        status="importing",
    )

    # determine chunksize from env
    try:
        CHUNK_SIZE = int(os.environ.get('DATASET_IMPORT_CHUNKSIZE', '10000'))
    except Exception:
        CHUNK_SIZE = 10000

    logger = logging.getLogger(__name__)

    # Process in chunks to handle very wide/tall CSVs without exhausting memory
    chunk_iter = pd.read_csv(dest_path, chunksize=CHUNK_SIZE, dtype=str, low_memory=False)
    chunk_index = 0
    for chunk in chunk_iter:
        chunk_index += 1
        start_chunk = time.perf_counter()
        rows_in_chunk = len(chunk)
        total += rows_in_chunk

        # rename columns if needed
        if rename_map:
            chunk = chunk.rename(columns=rename_map)

        # handle combined date+time if mapping detected in sample
        td = mapped.get('transaction_datetime')
        if isinstance(td, tuple):
            date_col, time_col = td
            if date_col in chunk.columns and time_col in chunk.columns:
                chunk['transaction_datetime'] = chunk[date_col].astype(str).str.strip() + ' ' + chunk[time_col].astype(str).str.strip()

        # ensure is_fraud exists
        if 'is_fraud' not in chunk.columns:
            chunk['is_fraud'] = False

        # Ensure required columns exist for this chunk
        available = set(chunk.columns.astype(str))
        missing = REQUIRED_COLUMNS - available
        if 'is_fraud' in missing:
            missing.remove('is_fraud')
        if missing:
            # treat all rows in this chunk as invalid
            invalid_count += rows_in_chunk
            logger.warning("[dataset_id=%s] chunk=%s missing required columns: %s", dataset.id, chunk_index, missing)
            continue

        # normalize and parse columns in vectorized manner
        # parse datetimes
        try:
            parsed_dt = pd.to_datetime(chunk['transaction_datetime'], errors='coerce', utc=True)
            if hasattr(parsed_dt.dt, 'tz'):
                parsed_dt = parsed_dt.dt.tz_convert('UTC').dt.tz_localize(None)
        except Exception:
            s = chunk['transaction_datetime'].astype(str).fillna('')
            s = s.str.replace(r"(\+|-)\d{2}:?\d{2}$|Z$", '', regex=True)
            parsed_dt = pd.to_datetime(s, errors='coerce')
        chunk['transaction_datetime'] = parsed_dt

        # clean amount
        chunk['amount'] = chunk['amount'].astype(str).str.replace(r'["\s]', '', regex=True)
        chunk['amount'] = chunk['amount'].str.replace(',', '.', regex=False)
        chunk['amount'] = chunk['amount'].str.replace(r'[^0-9\.\-]', '', regex=True)
        chunk['amount'] = pd.to_numeric(chunk['amount'], errors='coerce')

        # normalize is_fraud
        chunk['is_fraud'] = chunk['is_fraud'].astype(bool)

        # determine valid rows
        valid_mask = chunk['transaction_id'].notna() & chunk['amount'].notna() & chunk['transaction_datetime'].notna()
        valid_df = chunk.loc[valid_mask]
        invalid_df = chunk.loc[~valid_mask]
        invalid_count += len(invalid_df)

        # build list of records for insertion
        valid_records = []
        for _, row in valid_df.iterrows():
            try:
                rec = {
                    'transaction_id': str(row['transaction_id']),
                    'amount': float(row['amount']),
                    'transaction_type': row.get('transaction_type'),
                    'channel': row.get('channel'),
                    'location': row.get('location'),
                    'device_id': row.get('device_id'),
                    'customer_hash': row.get('customer_hash'),
                    'transaction_datetime': row['transaction_datetime'].to_pydatetime() if not pd.isna(row['transaction_datetime']) else None,
                    'is_fraud': bool(row['is_fraud']),
                }
                valid_records.append(rec)
            except Exception:
                # skip malformed row
                invalid_count += 1

        # insert records in bulk and measure time
        insert_start = time.perf_counter()
        inserted = 0
        if valid_records:
            inserted = transaction_repository.insert_transactions(db, valid_records, dataset_id=dataset.id)
            num_inserted += inserted
        insert_time = time.perf_counter() - insert_start

        total_chunk_time = time.perf_counter() - start_chunk
        logger.info("[dataset_id=%s] chunk=%s rows=%s valid=%s inserted=%s clean_time=%.3fs insert_time=%.3fs total=%.3fs",
                    dataset.id, chunk_index, rows_in_chunk, len(valid_records), inserted, total_chunk_time - insert_time, insert_time, total_chunk_time)

        # update dataset counts periodically
        try:
            dataset.total_records = total
            dataset.valid_records = num_inserted
            dataset.invalid_records = invalid_count
            db.add(dataset)
            db.commit()
        except Exception:
            db.rollback()

    # finalize dataset
    try:
        dataset.total_records = total
        dataset.valid_records = num_inserted
        dataset.invalid_records = invalid_count
        dataset.status = 'imported'
        dataset.file_path = dest_path
        db.add(dataset)
        db.commit()
    except Exception:
        db.rollback()

    return {'dataset': dataset, 'inserted': num_inserted, 'total': total, 'valid': num_inserted, 'invalid': invalid_count}


def import_dataset_background(dest_path: str, dataset_id: int, name: str, file_name: str, db_bind: Optional[object] = None):
    """Background worker variant: open a new DB session and process the CSV at dest_path,
    updating the dataset record identified by dataset_id. This can be scheduled via
    FastAPI BackgroundTasks so the HTTP request returns immediately after the file is saved.
    """
    # Create a DB session. Prefer a provided bind (from the request/session) so
    # background tasks use the same engine as the web app/test harness. If no
    # bind is provided, fall back to the application SessionLocal.
    if db_bind is None:
        from backend.app.database import SessionLocal
        db = SessionLocal()
    else:
        SessionLocalLocal = sessionmaker(autocommit=False, autoflush=False, bind=db_bind)
        db = SessionLocalLocal()
    try:
        logger = logging.getLogger(__name__)
        # mark dataset as processing and set started_at
        ds = dataset_repository.get_dataset(db, dataset_id)
        if ds:
            ds.status = 'PROCESSING'
            ds.started_at = datetime.utcnow()
            db.add(ds)
            db.commit()

        # read sample and detect mappings
        try:
            sample = pd.read_csv(dest_path, nrows=5)
        except Exception as e:
            # mark dataset as failed
            ds = dataset_repository.get_dataset(db, dataset_id)
            if ds:
                ds.status = 'FAILED'
                ds.error_message = f"Failed reading sample: {e}"
                ds.finished_at = datetime.utcnow()
                db.add(ds)
                db.commit()
            return

        mapped = _map_columns(sample)

        rename_map = {}
        for expected, actual in mapped.items():
            if actual is None:
                continue
            if expected == 'transaction_datetime' and isinstance(actual, tuple):
                continue
            if actual != expected:
                rename_map[actual] = expected

        total = 0
        num_inserted = 0
        invalid_count = 0

        # determine chunksize from env
        try:
            CHUNK_SIZE = int(os.environ.get('DATASET_IMPORT_CHUNKSIZE', '10000'))
        except Exception:
            CHUNK_SIZE = 10000

        chunk_iter = pd.read_csv(dest_path, chunksize=CHUNK_SIZE, dtype=str, low_memory=False)
        chunk_index = 0
        for chunk in chunk_iter:
            chunk_index += 1
            start_chunk = time.perf_counter()
            rows_in_chunk = len(chunk)
            total += rows_in_chunk

            if rename_map:
                chunk = chunk.rename(columns=rename_map)

            td = mapped.get('transaction_datetime')
            if isinstance(td, tuple):
                date_col, time_col = td
                if date_col in chunk.columns and time_col in chunk.columns:
                    chunk['transaction_datetime'] = chunk[date_col].astype(str).str.strip() + ' ' + chunk[time_col].astype(str).str.strip()

            if 'is_fraud' not in chunk.columns:
                chunk['is_fraud'] = False

            available = set(chunk.columns.astype(str))
            missing = REQUIRED_COLUMNS - available
            if 'is_fraud' in missing:
                missing.remove('is_fraud')
            if missing:
                invalid_count += rows_in_chunk
                logger.warning("[dataset_id=%s] chunk=%s missing required columns: %s", dataset_id, chunk_index, missing)
                continue

            # normalize/parse
            try:
                parsed_dt = pd.to_datetime(chunk['transaction_datetime'], errors='coerce', utc=True)
                if hasattr(parsed_dt.dt, 'tz'):
                    parsed_dt = parsed_dt.dt.tz_convert('UTC').dt.tz_localize(None)
            except Exception:
                s = chunk['transaction_datetime'].astype(str).fillna('')
                s = s.str.replace(r"(\+|-)\d{2}:?\d{2}$|Z$", '', regex=True)
                parsed_dt = pd.to_datetime(s, errors='coerce')
            chunk['transaction_datetime'] = parsed_dt

            chunk['amount'] = chunk['amount'].astype(str).str.replace(r'["\s]', '', regex=True)
            chunk['amount'] = chunk['amount'].str.replace(',', '.', regex=False)
            chunk['amount'] = chunk['amount'].str.replace(r'[^0-9\.\-]', '', regex=True)
            chunk['amount'] = pd.to_numeric(chunk['amount'], errors='coerce')

            chunk['is_fraud'] = chunk['is_fraud'].astype(bool)

            valid_mask = chunk['transaction_id'].notna() & chunk['amount'].notna() & chunk['transaction_datetime'].notna()
            valid_df = chunk.loc[valid_mask]
            invalid_df = chunk.loc[~valid_mask]
            invalid_count += len(invalid_df)

            valid_records = []
            for _, row in valid_df.iterrows():
                try:
                    rec = {
                        'transaction_id': str(row['transaction_id']),
                        'amount': float(row['amount']),
                        'transaction_type': row.get('transaction_type'),
                        'channel': row.get('channel'),
                        'location': row.get('location'),
                        'device_id': row.get('device_id'),
                        'customer_hash': row.get('customer_hash'),
                        'transaction_datetime': row['transaction_datetime'].to_pydatetime() if not pd.isna(row['transaction_datetime']) else None,
                        'is_fraud': bool(row['is_fraud']),
                    }
                    valid_records.append(rec)
                except Exception:
                    invalid_count += 1

            insert_start = time.perf_counter()
            inserted = 0
            if valid_records:
                try:
                    # mark status as inserting
                    ds = dataset_repository.get_dataset(db, dataset_id)
                    if ds:
                        ds.status = 'INSERTING'
                        db.add(ds)
                        db.commit()
                    inserted = transaction_repository.insert_transactions(db, valid_records, dataset_id=dataset_id)
                    num_inserted += inserted
                except Exception as e:
                    logger.exception("Error inserting chunk %s for dataset %s: %s", chunk_index, dataset_id, e)
                    # record error but continue with next chunks
                    ds = dataset_repository.get_dataset(db, dataset_id)
                    if ds:
                        ds.error_message = (ds.error_message or '') + f"; chunk_{chunk_index}_error: {e}"
                        db.add(ds)
                        db.commit()
            insert_time = time.perf_counter() - insert_start

            total_chunk_time = time.perf_counter() - start_chunk
            logger.info("[dataset_id=%s] chunk=%s rows=%s valid=%s inserted=%s clean_time=%.3fs insert_time=%.3fs total=%.3fs",
                        dataset_id, chunk_index, rows_in_chunk, len(valid_records), inserted, total_chunk_time - insert_time, insert_time, total_chunk_time)

            try:
                ds = dataset_repository.get_dataset(db, dataset_id)
                if ds:
                    ds.total_records = total
                    ds.valid_records = num_inserted
                    ds.invalid_records = invalid_count
                    db.add(ds)
                    db.commit()
            except Exception:
                db.rollback()

        # finalize
        try:
            ds = dataset_repository.get_dataset(db, dataset_id)
            if ds:
                ds.total_records = total
                ds.valid_records = num_inserted
                ds.invalid_records = invalid_count
                ds.status = 'COMPLETED'
                ds.finished_at = datetime.utcnow()
                ds.file_path = dest_path
                ds.processed_at = datetime.utcnow()
                db.add(ds)
                db.commit()
        except Exception as e:
            db.rollback()
            ds = dataset_repository.get_dataset(db, dataset_id)
            if ds:
                ds.status = 'FAILED'
                ds.error_message = (ds.error_message or '') + f"; finalize_error: {e}"
                ds.finished_at = datetime.utcnow()
                db.add(ds)
                db.commit()
    finally:
        db.close()

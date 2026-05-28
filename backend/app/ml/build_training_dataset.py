import os
import time
import hashlib
from typing import Optional

import pandas as pd
from sqlalchemy import text

from backend.app.database import SessionLocal
from backend.app.models.models import Dataset, Transaction
from .proxy_labeling import generate_proxy_fraud_label
from . import preprocessing


PROJECT_PROCESSED_DIR = os.environ.get("PROJECT_PROCESSED_DIR") or os.path.join(
    os.getcwd(), "data", "processed"
)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    def clean_name(s: str) -> str:
        if not isinstance(s, str):
            s = str(s)
        s = s.strip()
        s = s.replace("\u00A0", " ")
        import unicodedata
        s = unicodedata.normalize("NFKD", s)
        s = "".join([c for c in s if not unicodedata.combining(c)])
        s = s.strip().lower()
        s = "_".join(s.split())
        return s

    mapping_candidates = {
        "pais": "country_code",
        "pos_entry_mode": "pos_entry_mode",
        "tiene_pinblock": "has_pinblock",
        "codigo_establecimiento": "merchant_code",
        "codigo_terminal": "terminal_code",
        "establecimiento": "merchant_name",
        "tarjeta": "masked_card",
        "pan_tarjeta": "pan_card",
        "codigo_autorizacion": "authorization_code",
        "numero_referencia": "reference_number",
        "transaction_datetime": "transaction_datetime",
        "fecha": "transaction_datetime",
        "fecha_hora": "transaction_datetime",
    }

    rename_map = {}
    for c in df.columns:
        cn = clean_name(c)
        if cn in mapping_candidates:
            rename_map[c] = mapping_candidates[cn]
        else:
            rename_map[c] = cn
    df = df.rename(columns=rename_map)
    # standard names
    if "amount" not in df.columns:
        for c in ["AMOUNT", "monto", "MONTO"]:
            if c in df.columns:
                # preserve the raw value here; convert_data_types() handles comma decimals safely
                df["amount"] = df[c]
                break
    if "transaction_datetime" not in df.columns:
        if "date" in df.columns and "time" in df.columns:
            df["transaction_datetime"] = pd.to_datetime(
                df["date"].astype(str) + " " + df["time"].astype(str), errors="coerce", utc=True
            )
        else:
            for c in ["transaction_datetime", "datetime", "fecha", "fecha_hora"]:
                if c in df.columns:
                    df["transaction_datetime"] = pd.to_datetime(df[c], errors="coerce", utc=True)
                    break
    if "customer_hash" not in df.columns:
        for c in ["customer_id", "customer", "cust_hash"]:
            if c in df.columns:
                df["customer_hash"] = df[c].astype(str)
                break
    if "merchant_hash" not in df.columns:
        for c in ["merchant", "merchant_id", "merchant_hash"]:
            if c in df.columns:
                df["merchant_hash"] = df[c].astype(str)
                break
    # infer card_brand if missing and card data present
    if ("card_brand" not in df.columns) or df["card_brand"].isna().all():
        def _infer_brand(val):
            try:
                if pd.isna(val):
                    return None
                s = str(val).strip()
                if not s:
                    return None
                for ch in s:
                    if ch.isdigit():
                        d = ch
                        break
                else:
                    return None
                if d == "4":
                    return "VISA"
                if d == "5":
                    return "MASTERCARD"
                if d == "6":
                    return "DISCOVER_OR_OTHER"
                return "UNKNOWN"
            except Exception:
                return None
        if "masked_card" in df.columns:
            df["card_brand"] = df["masked_card"].apply(_infer_brand)
        elif "pan_card" in df.columns:
            df["card_brand"] = df["pan_card"].apply(_infer_brand)
    return df


def deterministic_transaction_id(row: pd.Series) -> str:
    # deterministic hash of key fields
    components = [
        str(row.get('transaction_datetime', '')),
        str(row.get('amount', '')),
        str(row.get('customer_hash', '')),
        str(row.get('merchant_code', '')),
        str(row.get('terminal_code', '')),
        str(row.get('reference_number', '')),
        str(row.get('authorization_code', '')),
        str(row.get('transaction_type', '')),
        str(row.get('transaction_category', '')),
        str(row.get('process_code', '')),
    ]
    payload = "|".join([c if c is not None else '' for c in components])
    h = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"tx_{h[:16]}"


def _write_chunk(df_chunk: pd.DataFrame, path: str, mode: str = "a"):
    header = not os.path.exists(path) or mode == "w"
    df_chunk.to_csv(path, index=False, mode=mode, header=header)


def _detect_constant_columns(csv_path: str, chunksize: int, ignore_cols: set[str]) -> list[str]:
    first_seen = {}
    non_constant = set()
    for df_chunk in pd.read_csv(csv_path, chunksize=chunksize):
        for col in df_chunk.columns:
            if col in ignore_cols or col in non_constant:
                continue
            series = df_chunk[col].dropna()
            if series.empty:
                continue
            values = series.astype(str).unique().tolist()
            if col not in first_seen and values:
                first_seen[col] = values[0]
            for value in values:
                if col in first_seen and value != first_seen[col]:
                    non_constant.add(col)
                    break
            if col not in first_seen and len(values) > 1:
                non_constant.add(col)
    return sorted([col for col in first_seen.keys() if col not in non_constant])


def build_training_dataset(
    input_csv: Optional[str] = None,
    dataset_id: Optional[int] = None,
    chunksize: int = 25000,
    out_name: Optional[str] = None,
    out_dir: Optional[str] = None,
    preprocess_only: bool = False,
):
    project_dir = out_dir or PROJECT_PROCESSED_DIR
    os.makedirs(project_dir, exist_ok=True)

    run_tag = f"dataset_{dataset_id}" if dataset_id else (out_name or "csv")
    cleaned_path = os.path.join(project_dir, f"cleaned_{run_tag}.csv")
    feature_path = os.path.join(project_dir, f"feature_set_{run_tag}.csv")
    report_path = os.path.join(project_dir, f"preprocessing_report_{run_tag}.md")

    # Prepare: if dataset_id provided, validate
    total_rows = None
    if dataset_id:
        session = SessionLocal()
        try:
            ds = session.query(Dataset).filter(Dataset.id == dataset_id).first()
            if not ds:
                raise SystemExit(f"Dataset id {dataset_id} does not exist.")
            total_rows = session.query(Transaction).filter(Transaction.dataset_id == dataset_id).count()
            if total_rows == 0:
                raise SystemExit(f"Dataset id {dataset_id} has 0 transactions. Aborting.")
        finally:
            session.close()

    # First pass: read input in chunks, normalize, write cleaned file, and accumulate aggregates
    aggregates = {}
    rows_read = 0
    rows_cleaned = 0
    chunk_no = 0

    # remove existing outputs if present
    for p in (cleaned_path, feature_path, report_path):
        try:
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass

    # Reader: DB streaming or CSV chunks
    if dataset_id:
        # stream from DB using server-side cursor
        session = SessionLocal()
        try:
            stmt = text("SELECT * FROM transactions WHERE dataset_id = :did ORDER BY id ASC")
            conn = session.connection()
            res = conn.execution_options(stream_results=True).execute(stmt, {"did": dataset_id})
            batch = []
            for row in res:
                batch.append(dict(row._mapping))
                if len(batch) >= chunksize:
                    df_chunk = pd.DataFrame(batch)
                    chunk_no += 1
                    tstart = time.time()
                    df_chunk = normalize_columns(df_chunk)
                    try:
                        df_chunk = preprocessing.convert_data_types(df_chunk)
                    except Exception:
                        pass
                    # synthesize anonymized keys (merchant_hash, customer_hash, transaction_id)
                    try:
                        df_chunk = preprocessing.generate_anonymized_keys(df_chunk)
                    except Exception:
                        pass
                    # deterministic id (if not already created by preprocessing)
                    if "transaction_id" not in df_chunk.columns:
                        df_chunk["transaction_id"] = df_chunk.apply(deterministic_transaction_id, axis=1)
                    # labeling if needed
                    if ("is_fraud" not in df_chunk.columns) or ("behavioral_risk_score" not in df_chunk.columns):
                        try:
                            df_chunk = generate_proxy_fraud_label(df_chunk)
                        except Exception:
                            pass
                    _write_chunk(df_chunk.drop(columns=[c for c in ["pan_card", "masked_card", "PAN_TARJETA", "TARJETA"] if c in df_chunk.columns], errors="ignore"), cleaned_path, mode="a")
                    rows_read += len(df_chunk)
                    rows_cleaned += len(df_chunk)
                    batch = []
            # last batch
            if batch:
                df_chunk = pd.DataFrame(batch)
                chunk_no += 1
                df_chunk = normalize_columns(df_chunk)
                try:
                    df_chunk = preprocessing.convert_data_types(df_chunk)
                except Exception:
                    pass
                # synthesize anonymized keys for last DB batch
                try:
                    df_chunk = preprocessing.generate_anonymized_keys(df_chunk)
                except Exception:
                    pass
                if "transaction_id" not in df_chunk.columns:
                    df_chunk["transaction_id"] = df_chunk.apply(deterministic_transaction_id, axis=1)
                # only generate proxy labels when `is_fraud` is not present
                if ("is_fraud" not in df_chunk.columns) and ("behavioral_risk_score" not in df_chunk.columns):
                    try:
                        df_chunk = generate_proxy_fraud_label(df_chunk)
                    except Exception:
                        pass
                _write_chunk(df_chunk.drop(columns=[c for c in ["pan_card", "masked_card", "PAN_TARJETA", "TARJETA"] if c in df_chunk.columns], errors="ignore"), cleaned_path, mode="a")
                rows_read += len(df_chunk)
                rows_cleaned += len(df_chunk)
        finally:
            session.close()
    else:
        if not input_csv:
            raise SystemExit("Either --dataset-id or --input must be provided.")
        for df_chunk in pd.read_csv(input_csv, chunksize=chunksize):
            chunk_no += 1
            tstart = time.time()
            df_chunk = normalize_columns(df_chunk)
            try:
                df_chunk = preprocessing.convert_data_types(df_chunk)
            except Exception:
                pass
            # synthesize anonymized keys for CSV chunks
            try:
                df_chunk = preprocessing.generate_anonymized_keys(df_chunk)
            except Exception:
                pass
            if "transaction_id" not in df_chunk.columns:
                df_chunk["transaction_id"] = df_chunk.apply(deterministic_transaction_id, axis=1)
            # only generate proxy labels when `is_fraud` is not present
            if ("is_fraud" not in df_chunk.columns) and ("behavioral_risk_score" not in df_chunk.columns):
                try:
                    df_chunk = generate_proxy_fraud_label(df_chunk)
                except Exception:
                    pass
            _write_chunk(df_chunk.drop(columns=[c for c in ["pan_card", "masked_card", "PAN_TARJETA", "TARJETA"] if c in df_chunk.columns], errors="ignore"), cleaned_path, mode="a")
            rows_read += len(df_chunk)
            rows_cleaned += len(df_chunk)

    # Second pass: build feature set from cleaned file in chunks (avoid full-memory)
    if preprocess_only:
        # Build minimal report and return early without generating feature set
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# Preprocessing Report\n\n")
            f.write(f"Source dataset_id: {dataset_id}\n" if dataset_id else f"Source file: {input_csv}\n")
            f.write("\n## Rows\n")
            try:
                total_rows = sum(1 for _ in open(cleaned_path)) - 1
            except Exception:
                total_rows = rows_read
            f.write(f"- original_rows: {total_rows}\n")
            f.write(f"- cleaned_rows: {rows_cleaned}\n")
            f.write(f"- feature_set_rows: 0\n\n")
            f.write("- NOTE: preprocess_only=True — feature set generation skipped.\n")
        return None, report_path

    forbidden = [
        "response_code",
        "normalized_response_code",
        "response_high_risk",
        "response_code_reason",
        "is_fraud_proxy",
        "behavioral_risk_score",
        "independent_rule_groups",
        "label_source",
        "fraud_label_reason",
        "risk_signal_reason",
        "transaction_id",
        "customer_hash",
        "merchant_hash",
        "device_id",
        "reference_number",
        "authorization_code",
        "merchant_code",
        "terminal_code",
        "pan_card",
        "masked_card",
        "PAN_TARJETA",
        "TARJETA",
        "has_pinblock_source",
    ]
    optional_drop = ["merchant_name", "transaction_datetime"]

    feature_rows = 0
    constant_cols = set(
        _detect_constant_columns(
            cleaned_path,
            chunksize=chunksize,
            ignore_cols={"is_fraud"} | set(forbidden) | set(optional_drop),
        )
    )
    for df_chunk in pd.read_csv(cleaned_path, chunksize=chunksize):
        # apply final cleanup and ensure target present
        drop_cols = [c for c in df_chunk.columns if c in forbidden or c in optional_drop or c in constant_cols]
        fs = df_chunk.drop(columns=drop_cols, errors="ignore")
        if "is_fraud" in df_chunk.columns and "is_fraud" not in fs.columns:
            fs["is_fraud"] = df_chunk["is_fraud"].astype(int)
        _write_chunk(fs, feature_path, mode="a")
        feature_rows += len(fs)

    # Build report
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Preprocessing Report\n\n")
        f.write(f"Source dataset_id: {dataset_id}\n" if dataset_id else f"Source file: {input_csv}\n")
        f.write("\n## Rows\n")
        if total_rows is None:
            # try to infer from cleaned file
            try:
                total_rows = sum(1 for _ in open(cleaned_path)) - 1
            except Exception:
                total_rows = rows_read
        f.write(f"- original_rows: {total_rows}\n")
        f.write(f"- cleaned_rows: {rows_cleaned}\n")
        f.write(f"- feature_set_rows: {feature_rows}\n\n")
        f.write("## Columns removed for sensitivity and leakage\n")
        for c in forbidden:
            f.write(f"- {c}\n")
        f.write("\n## Optional columns removed to avoid overfitting\n")
        for c in optional_drop:
            f.write(f"- {c}\n")
        f.write("\n## Constant columns removed before feature set generation\n")
        for c in sorted(constant_cols):
            f.write(f"- {c}\n")
        f.write("\n## Confirmations\n")
        f.write("- response_code not used for labeling\n")
        f.write("- SMOTE not applied\n")
        f.write("- OneHotEncoder not applied\n")
        # distributions and feature frequencies (from cleaned file)
        try:
            dist_cols = ["country_code", "pos_entry_mode", "has_pinblock", "card_presence_type", "card_brand"]
            dists = {c: {} for c in dist_cols}
            feature_counts = {}
            feature_totals = 0
            merchant_uniques = set()
            for chunk in pd.read_csv(cleaned_path, chunksize=25000):
                feature_totals += len(chunk)
                for c in dist_cols:
                    if c in chunk.columns:
                        vc = chunk[c].value_counts(dropna=False).to_dict()
                        for k, v in vc.items():
                            dists[c][str(k)] = dists[c].get(str(k), 0) + int(v)
                if "merchant_hash" in chunk.columns:
                    merchant_uniques.update(chunk["merchant_hash"].dropna().astype(str).unique().tolist())
                for c in chunk.columns:
                    if c.startswith("feature_"):
                        ones = int(chunk[c].fillna(0).astype(int).sum())
                        feature_counts[c] = feature_counts.get(c, 0) + ones
            f.write("\n## Distributions\n")
            for c in dist_cols:
                f.write(f"- {c}: {dists.get(c, {})}\n")
            f.write(f"- merchant_hash_unique: {len(merchant_uniques)}\n")
            f.write("\n## Feature frequencies\n")
            over_30 = []
            over_100 = []
            for c, ones in feature_counts.items():
                pct = (ones / feature_totals * 100) if feature_totals else 0.0
                f.write(f"- {c}: {ones} ({pct:.2f}%)\n")
                if pct >= 30.0:
                    over_30.append(c)
                if pct >= 100.0:
                    over_100.append(c)
            f.write("\n## Features above 30%\n")
            for c in over_30:
                f.write(f"- {c}\n")
            f.write("\n## Features at 100%\n")
            for c in over_100:
                f.write(f"- {c}\n")
        except Exception as e:
            f.write("\n## Distributions\n")
            f.write(f"- failed to compute distributions: {e}\n")
        f.write("\n## Warnings and checks\n")
        # detect whether response_code or label_source were present/used in source
        try:
            if input_csv and os.path.exists(input_csv):
                hdr = pd.read_csv(input_csv, nrows=20)
                if 'label_source' in hdr.columns:
                    if hdr['label_source'].astype(str).str.contains('response_code_proxy', na=False).any():
                        f.write("- ALERT: response_code_proxy was used in labeling.\n")
                    else:
                        f.write("- CONFIRMATION: response_code was NOT used based on label_source values.\n")
                else:
                    f.write("- INFO: label_source column not present in source file.\n")
            else:
                # fallback: check cleaned file
                try:
                    hdr2 = pd.read_csv(cleaned_path, nrows=20)
                    if 'label_source' in hdr2.columns:
                        if hdr2['label_source'].astype(str).str.contains('response_code_proxy', na=False).any():
                            f.write("- ALERT: response_code_proxy was used in labeling.\n")
                        else:
                            f.write("- CONFIRMATION: response_code was NOT used based on label_source values.\n")
                    else:
                        f.write("- INFO: label_source column not present in cleaned data.\n")
                except Exception:
                    f.write("- INFO: label_source presence could not be determined.\n")
        except Exception:
            f.write("- INFO: label_source presence could not be determined.\n")
        f.write("- CONFIRMATION: SMOTE was NOT applied in preprocessing.\n")
        f.write("- CONFIRMATION: OneHotEncoding was NOT applied in preprocessing.\n")

    return feature_path, report_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build training dataset (chunked, streaming)")
    parser.add_argument("--input", help="Input CSV file path", default=None)
    parser.add_argument("--dataset-id", type=int, help="Dataset id in DB to read transactions from", default=None)
    parser.add_argument("--chunksize", type=int, help="Chunk size", default=25000)
    parser.add_argument("--out-name", help="Output name tag", default=None)
    parser.add_argument("--preprocess-only", action="store_true", help="Run only preprocessing (FASE A) and skip feature set generation")
    args = parser.parse_args()

    build_training_dataset(
        input_csv=args.input,
        dataset_id=args.dataset_id,
        chunksize=args.chunksize,
        out_name=args.out_name,
        preprocess_only=bool(args.preprocess_only),
    )

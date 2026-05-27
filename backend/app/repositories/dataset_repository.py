from sqlalchemy.orm import Session
from backend.app.models.models import Dataset
from sqlalchemy.sql import text
import os


def create_dataset(db: Session, *, name: str, file_name: str, total_records: int, valid_records: int, invalid_records: int, status: str = "imported", file_path: str = None, original_filename: str = None, source_type: str = "CSV"):
    """
    Create a dataset record using an explicit INSERT that only includes the
    legacy columns. This avoids failing when the models have additional
    columns not yet present in the database schema (no migration applied).
    After insert, refresh and return the Dataset instance.
    """
    orig = original_filename or file_name
    # Use explicit column list matching the existing DB schema
    sql = text("""
    INSERT INTO datasets (name, original_filename, file_name, file_path, source_type, total_records, valid_records, invalid_records, status)
    VALUES (:name, :original_filename, :file_name, :file_path, :source_type, :total_records, :valid_records, :invalid_records, :status)
    RETURNING id
    """)
    try:
        res = db.execute(sql, {
            'name': name,
            'original_filename': orig,
            'file_name': file_name,
            'file_path': file_path,
            'source_type': source_type,
            'total_records': total_records,
            'valid_records': valid_records,
            'invalid_records': invalid_records,
            'status': status,
        })
        new_id = res.scalar()
        db.commit()
    except Exception:
        db.rollback()
        # fallback to ORM path if raw insert fails for any reason
        ds = Dataset(
            name=name,
            original_filename=orig,
            file_name=file_name,
            file_path=file_path,
            source_type=source_type,
            total_records=total_records,
            valid_records=valid_records,
            invalid_records=invalid_records,
            status=status,
        )
        db.add(ds)
        db.commit()
        db.refresh(ds)
        return ds

    # fetch the dataset row via ORM to return a Dataset instance
    ds = db.query(Dataset).filter(Dataset.id == new_id).first()
    return ds


def list_datasets(db: Session, limit: int = 50, offset: int = 0):
    # Attempt ORM query; if DB schema lacks newly added columns, fall back to
    # selecting a minimal set of columns to remain backwards-compatible.
    try:
        return db.query(Dataset).order_by(Dataset.created_at.desc()).offset(offset).limit(limit).all()
    except Exception:
        # raw select of legacy columns
        from sqlalchemy.sql import text
        sql = text("SELECT id, name, original_filename, file_name, file_path, total_records, valid_records, invalid_records, status, created_at FROM datasets ORDER BY created_at DESC LIMIT :limit OFFSET :offset")
        res = db.execute(sql, {'limit': limit, 'offset': offset})
        rows = res.fetchall()
        out = []
        for r in rows:
            out.append({
                'id': r['id'],
                'name': r['name'],
                'original_filename': r['original_filename'],
                'file_name': r['file_name'],
                'file_path': r['file_path'],
                'total_records': r['total_records'],
                'valid_records': r['valid_records'],
                'invalid_records': r['invalid_records'],
                'status': r['status'],
                'created_at': r['created_at'],
            })
        return out


def get_dataset(db: Session, dataset_id: int):
    try:
        return db.query(Dataset).filter(Dataset.id == dataset_id).first()
    except Exception:
        # fallback to minimal select
        from types import SimpleNamespace
        from sqlalchemy.sql import text
        sql = text("SELECT id, name, original_filename, file_name, file_path, total_records, valid_records, invalid_records, status, created_at FROM datasets WHERE id = :id")
        res = db.execute(sql, {'id': dataset_id}).fetchone()
        if not res:
            return None
        return SimpleNamespace(**{
            'id': res['id'],
            'name': res['name'],
            'original_filename': res['original_filename'],
            'file_name': res['file_name'],
            'file_path': res['file_path'],
            'total_records': res['total_records'],
            'valid_records': res['valid_records'],
            'invalid_records': res['invalid_records'],
            'status': res['status'],
            'created_at': res['created_at'],
        })


def delete_dataset(db: Session, dataset_id: int):
    ds = get_dataset(db, dataset_id)
    if not ds:
        return False
    # attempt to remove file from disk
    try:
        if ds.file_path and os.path.exists(ds.file_path):
            os.remove(ds.file_path)
    except Exception:
        pass
    try:
        db.delete(ds)
        db.commit()
    except Exception:
        db.rollback()
        raise
    return True

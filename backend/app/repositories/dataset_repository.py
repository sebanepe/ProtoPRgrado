from sqlalchemy.orm import Session
from backend.app.models.models import Dataset
import os


def create_dataset(db: Session, *, name: str, file_name: str, total_records: int, valid_records: int, invalid_records: int, status: str = "imported", file_path: str = None, original_filename: str = None, source_type: str = "CSV"):
    ds = Dataset(
        name=name,
        original_filename=original_filename or file_name,
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


def list_datasets(db: Session, limit: int = 50, offset: int = 0):
    return db.query(Dataset).order_by(Dataset.created_at.desc()).offset(offset).limit(limit).all()


def get_dataset(db: Session, dataset_id: int):
    return db.query(Dataset).filter(Dataset.id == dataset_id).first()


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

from sqlalchemy.orm import Session
from backend.app.models.models import Dataset


def create_dataset(db: Session, *, name: str, file_name: str, total_records: int, valid_records: int, invalid_records: int, status: str = "imported"):
    ds = Dataset(
        name=name,
        file_name=file_name,
        total_records=total_records,
        valid_records=valid_records,
        invalid_records=invalid_records,
        status=status,
    )
    db.add(ds)
    db.commit()
    db.refresh(ds)
    return ds

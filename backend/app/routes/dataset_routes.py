from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status
from sqlalchemy.orm import Session
from backend.app.database import get_db
from backend.app.services import dataset_service

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.post("/import")
def import_dataset(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only CSV files are supported")
    try:
        result = dataset_service.import_dataset(db, file.file, name=file.filename, file_name=file.filename)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    return {"message": "imported", "details": {"inserted": result["inserted"], "total": result["total"], "valid": result["valid"], "invalid": result["invalid"]}}

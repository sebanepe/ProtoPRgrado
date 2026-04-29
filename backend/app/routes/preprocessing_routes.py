from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from backend.app.database import get_db
from backend.app.services import preprocessing_service

router = APIRouter(prefix="/preprocessing", tags=["preprocessing"])


@router.post("/run")
def run_preprocessing(db: Session = Depends(get_db)):
    try:
        summary = preprocessing_service.run_preprocessing(db)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    return {"status": "ok", "summary": summary}

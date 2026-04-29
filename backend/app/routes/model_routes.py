from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from backend.app.database import get_db
from backend.app.services import model_service

router = APIRouter(prefix="/models", tags=["models"])


@router.post("/train")
def train_models(db: Session = Depends(get_db)):
    try:
        results = model_service.train_and_record(db)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    return {"status": "ok", "results": results}


@router.get("/results")
def get_results(db: Session = Depends(get_db)):
    return {"results": model_service.list_results(db)}


@router.post("/{id}/activate")
def activate(id: int, db: Session = Depends(get_db)):
    try:
        mr = model_service.activate_model(db, id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    return {"status": "ok", "activated": mr.id}

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from backend.app.database import get_db
from backend.app.services import evaluation_service
from fastapi.responses import FileResponse

router = APIRouter(prefix="/models", tags=["models"])


@router.get("/comparison")
def comparison(input_path: str = "data/processed/preprocessed_transactions.csv", db: Session = Depends(get_db)):
    try:
        results = evaluation_service.compare_models(db, input_path=input_path)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    return {"results": results}


@router.get("/export-results")
def export_results(input_path: str = "data/processed/preprocessed_transactions.csv", db: Session = Depends(get_db)):
    try:
        export_path = "data/processed/model_comparison.csv"
        evaluation_service.compare_models(db, input_path=input_path, export_path=export_path)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    return FileResponse(export_path, media_type="text/csv", filename="model_comparison.csv")

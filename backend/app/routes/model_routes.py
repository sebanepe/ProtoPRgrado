from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from backend.app.database import get_db
from backend.app.services import model_service
from backend.app.services.permission_service import require_permission
from backend.app.services.authorization import get_user_from_header

router = APIRouter(prefix="/models", tags=["models"])


@router.post("/train")
def train_models(db: Session = Depends(get_db), _auth=Depends(require_permission("train"))):
    try:
        results = model_service.train_and_record(db)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    return {"status": "ok", "results": results}


@router.get("/results")
def get_results(db: Session = Depends(get_db), _auth=Depends(get_user_from_header)):
    return {"results": model_service.list_results(db)}


@router.post("/{id}/activate")
def activate(id: int, db: Session = Depends(get_db), _auth=Depends(require_permission("configure_model"))):
    try:
        mr = model_service.activate_model(db, id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    return {"status": "ok", "activated": mr.id}


@router.get("/{id}/export")
def export_model_results(id: int, db: Session = Depends(get_db), _auth=Depends(get_user_from_header)):
    # Export basic metrics for the specified model as CSV
    results = model_service.list_results(db)
    target = None
    for r in results:
        if r.get('id') == id:
            target = r
            break
    if not target:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Model not found')
    # build CSV content
    import io, csv
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(['id','model_name','version','precision','recall','f1_score','roc_auc','is_active','created_at'])
    writer.writerow([target.get('id'), target.get('model_name'), target.get('version'), target.get('precision'), target.get('recall'), target.get('f1_score'), target.get('roc_auc'), target.get('is_active'), target.get('created_at')])
    return out.getvalue()

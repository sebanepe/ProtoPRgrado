from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.ml import model_evaluation_service

router = APIRouter(prefix="/api/model-evaluation", tags=["model-evaluation"])


@router.post("/build-comparison")
def build_comparison(body: dict, db: Session = Depends(get_db)):
    source_run = body.get("source_run")
    if not source_run:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="source_run is required")
    try:
        result = model_evaluation_service.build_model_evaluation_comparison(db, source_run)
        return {"status": "ok", "result": result}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/summary")
def summary(source_run: str = Query(...)):
    return model_evaluation_service.get_summary(source_run)


@router.get("/alert-level")
def alert_level(
    source_run: str = Query(...),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=500),
    rule_code: str | None = None,
    human_review_status: str | None = None,
    supervised_positive_any: str | None = None,
    unsupervised_anomaly_any: str | None = None,
    risk_level: str | None = None,
):
    return model_evaluation_service.get_alert_level(
        source_run,
        page,
        page_size,
        {
            "rule_code": rule_code,
            "human_review_status": human_review_status,
            "supervised_positive_any": supervised_positive_any,
            "unsupervised_anomaly_any": unsupervised_anomaly_any,
            "risk_level": risk_level,
        },
    )


@router.get("/transaction-level")
def transaction_level(
    source_run: str = Query(...),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=500),
    flagged_by_rules: str | None = None,
    flagged_by_isolation_forest: str | None = None,
    flagged_by_autoencoder: str | None = None,
    country_code: str | None = None,
    merchant_rubro_proxy: str | None = None,
):
    return model_evaluation_service.get_transaction_level(
        source_run,
        page,
        page_size,
        {
            "flagged_by_rules": flagged_by_rules,
            "flagged_by_isolation_forest": flagged_by_isolation_forest,
            "flagged_by_autoencoder": flagged_by_autoencoder,
            "country_code": country_code,
            "merchant_rubro_proxy": merchant_rubro_proxy,
        },
    )


@router.get("/report")
def report(source_run: str = Query(...)):
    return {"source_run": source_run, "markdown": model_evaluation_service.get_report_markdown(source_run)}


@router.get("/metadata")
def metadata(source_run: str = Query(...)):
    return model_evaluation_service.get_metadata(source_run)


@router.get("/top-cases")
def top_cases(source_run: str = Query(...), limit: int = Query(20, ge=1, le=500)):
    return {"source_run": source_run, "items": model_evaluation_service.get_top_cases(source_run, limit)}

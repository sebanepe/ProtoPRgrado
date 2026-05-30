"""
Endpoints for unsupervised anomaly detection results.
Provides access to anomaly scores, metrics, reports, and model metadata.
"""
from fastapi import APIRouter, HTTPException, Query, status
from typing import Optional
from backend.app.services.anomaly_service import AnomalyService

router = APIRouter(prefix="/api/anomaly", tags=["anomaly", "unsupervised"])

# Initialize service
anomaly_service = AnomalyService()


@router.get("/runs")
def list_anomaly_runs():
    """
    List available anomaly detection runs.
    Discovers runs by looking for anomaly_scores_*.csv, anomaly_report_*.md,
    and isolation_forest_*_metadata.json files.
    """
    try:
        runs = anomaly_service.list_anomaly_runs()
        return {
            "count": len(runs),
            "runs": runs,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing anomaly runs: {str(e)}",
        )


@router.get("/scores")
def get_anomaly_scores(
    run_id: str = Query(..., description="Run ID (required), e.g., run_26 or preprocessed_run_26"),
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(50, ge=1, le=200, description="Items per page"),
    anomaly_flag: Optional[int] = Query(None, description="Filter by anomaly_flag (0 or 1)"),
    country_code: Optional[str] = Query(None, description="Filter by country_code"),
    pos_entry_mode: Optional[str] = Query(None, description="Filter by pos_entry_mode"),
    merchant_rubro_proxy: Optional[str] = Query(None, description="Filter by merchant_rubro_proxy"),
    customer_hash: Optional[str] = Query(None, description="Filter by customer_hash"),
    min_score: Optional[float] = Query(None, description="Minimum anomaly_score"),
    max_score: Optional[float] = Query(None, description="Maximum anomaly_score"),
):
    """
    Get paginated anomaly scores with optional filters.
    """
    try:
        result = anomaly_service.get_anomaly_scores(
            run_id=run_id,
            page=page,
            page_size=page_size,
            anomaly_flag=anomaly_flag,
            country_code=country_code,
            pos_entry_mode=pos_entry_mode,
            merchant_rubro_proxy=merchant_rubro_proxy,
            customer_hash=customer_hash,
            min_score=min_score,
            max_score=max_score,
        )
        return result
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Anomaly scores file not found for run: {run_id}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving anomaly scores: {str(e)}",
        )


@router.get("/top")
def get_top_anomalies(
    run_id: str = Query(..., description="Run ID (required)"),
    limit: int = Query(100, ge=1, le=500, description="Maximum number of anomalies to return"),
):
    """
    Get top anomalies ordered by anomaly_rank (ascending).
    """
    try:
        result = anomaly_service.get_top_anomalies(
            run_id=run_id,
            limit=limit,
        )
        return result
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Anomaly scores file not found for run: {run_id}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving top anomalies: {str(e)}",
        )


@router.get("/metrics")
def get_anomaly_metrics(
    run_id: str = Query(..., description="Run ID (required)"),
):
    """
    Get anomaly metrics including distributions by country, pos_entry_mode, mcc, and hour.
    """
    try:
        result = anomaly_service.get_anomaly_metrics(run_id=run_id)
        return result
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Anomaly scores file not found for run: {run_id}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving metrics: {str(e)}",
        )


@router.get("/report")
def get_anomaly_report(
    run_id: str = Query(..., description="Run ID (required)"),
):
    """
    Get anomaly detection report in markdown format.
    """
    try:
        result = anomaly_service.get_anomaly_report(run_id=run_id)
        return result
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Anomaly report file not found for run: {run_id}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving report: {str(e)}",
        )


@router.get("/model-metadata")
def get_model_metadata(
    run_id: str = Query(..., description="Run ID (required)"),
):
    """
    Get model metadata including algorithm, parameters, and feature info.
    """
    try:
        result = anomaly_service.get_model_metadata(run_id=run_id)
        return result
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model metadata file not found for run: {run_id}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving model metadata: {str(e)}",
        )


@router.post("/train")
def train_anomaly_model(
    source_run: str = Query(..., description="Source run (preprocessed data run ID)"),
    model: str = Query("isolation_forest", description="Model type"),
    contamination: float = Query(0.01, ge=0.001, le=0.5, description="Contamination parameter"),
    sample_size: Optional[int] = Query(None, description="Sample size for training"),
    max_categories: int = Query(50, description="Maximum categories for one-hot encoding"),
    n_estimators: int = Query(200, description="Number of estimators for ensemble models"),
):
    """
    Train unsupervised anomaly detection model.
    Executes the training CLI synchronously and returns status.
    """
    try:
        result = anomaly_service.train_anomaly_model(
            source_run=source_run,
            model=model,
            contamination=contamination,
            sample_size=sample_size,
            max_categories=max_categories,
            n_estimators=n_estimators,
        )
        return result
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Training error: {str(e)}",
        )

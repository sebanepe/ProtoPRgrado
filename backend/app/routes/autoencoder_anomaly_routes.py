from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.services.autoencoder_anomaly_service import AutoencoderAnomalyService


router = APIRouter(prefix="/api/anomaly/autoencoder", tags=["anomaly", "autoencoder", "unsupervised"])
autoencoder_service = AutoencoderAnomalyService()


class AutoencoderTrainRequest(BaseModel):
    source_run: str
    epochs: int = Field(30, ge=1)
    batch_size: int = Field(512, ge=1)
    latent_dim: int = Field(16, ge=1)
    learning_rate: float = Field(0.001, gt=0)
    contamination: float = Field(0.01, gt=0, lt=1)
    sample_size: Optional[int] = Field(None, ge=1)


@router.post("/train")
def train_autoencoder(request: AutoencoderTrainRequest, db: Session = Depends(get_db)):
    try:
        result = autoencoder_service.train(
            source_run=request.source_run,
            epochs=request.epochs,
            batch_size=request.batch_size,
            latent_dim=request.latent_dim,
            learning_rate=request.learning_rate,
            contamination=request.contamination,
            sample_size=request.sample_size,
            db=db,
        )
        if result.get("status") == "AUTOENCODER_DEPENDENCY_NOT_AVAILABLE":
            return result
        return result
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Autoencoder training error: {exc}")


@router.get("/metrics")
def get_autoencoder_metrics(source_run: str = Query(...)):
    try:
        return autoencoder_service.get_metrics(source_run)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))


@router.get("/scores")
def get_autoencoder_scores(
    source_run: str = Query(...),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    anomaly_flag: Optional[int] = Query(None, ge=0, le=1),
):
    try:
        return autoencoder_service.get_scores(source_run, page=page, page_size=page_size, anomaly_flag=anomaly_flag)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))


@router.get("/report")
def get_autoencoder_report(source_run: str = Query(...)):
    try:
        return autoencoder_service.get_report(source_run)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))


@router.get("/model-metadata")
def get_autoencoder_model_metadata(source_run: str = Query(...)):
    try:
        return autoencoder_service.get_model_metadata(source_run)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))

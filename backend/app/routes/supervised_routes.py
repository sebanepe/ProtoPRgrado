from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.services import supervised_service


router = APIRouter(prefix="/api/supervised", tags=["supervised"])


@router.get("/human-label-summary")
def get_human_label_summary(
    source_run: Optional[str] = Query(None),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    return supervised_service.get_human_label_summary(db, source_run=source_run)


@router.get("/human-readiness")
def get_human_readiness(
    source_run: Optional[str] = Query(None),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    return supervised_service.get_human_readiness(db, source_run=source_run)

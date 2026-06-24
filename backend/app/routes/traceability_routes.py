from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.services.traceability_service import get_import_alert_summary

router = APIRouter(prefix="/api/traceability", tags=["traceability"])


@router.get("/import-alert-summary")
def import_alert_summary(db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    """
    Resumen de trazabilidad de extremo a extremo: Dataset → PreprocessingRun → RuleRun → Alertas.
    Solo lectura. No modifica datos. No crea alertas ni revisiones. No genera is_fraud.
    """
    return get_import_alert_summary(db)

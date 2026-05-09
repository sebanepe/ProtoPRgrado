from sqlalchemy.orm import Session
from backend.app.repositories import model_config_repository
import os


def get_active_config(db: Session):
    return model_config_repository.get_active_config(db)


def set_model_config(db: Session, *, active_model_id: int | None = None, alert_threshold: float = 0.7, updated_by: str | None = None):
    if alert_threshold is None:
        alert_threshold = float(os.getenv("ALERT_THRESHOLD", "0.7"))
    cfg = model_config_repository.create_config(db, active_model_id=active_model_id, alert_threshold=alert_threshold, updated_by=updated_by)
    return cfg


def get_active_threshold(db: Session) -> float:
    cfg = get_active_config(db)
    if cfg and cfg.alert_threshold is not None:
        return float(cfg.alert_threshold)
    return float(os.getenv("ALERT_THRESHOLD", "0.7"))

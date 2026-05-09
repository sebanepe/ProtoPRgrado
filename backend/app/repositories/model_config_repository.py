from sqlalchemy.orm import Session
from backend.app.models.models import ModelConfig
from datetime import datetime


def get_active_config(db: Session) -> ModelConfig | None:
    return db.query(ModelConfig).filter(ModelConfig.is_active == True).order_by(ModelConfig.updated_at.desc()).first()


def create_config(db: Session, *, active_model_id: int | None, alert_threshold: float, updated_by: str | None) -> ModelConfig:
    # deactivate existing configs
    db.query(ModelConfig).update({"is_active": False})
    cfg = ModelConfig(active_model_id=active_model_id, alert_threshold=alert_threshold, updated_by=updated_by, is_active=True)
    db.add(cfg)
    db.commit()
    db.refresh(cfg)
    return cfg


def list_configs(db: Session):
    return db.query(ModelConfig).order_by(ModelConfig.updated_at.desc()).all()

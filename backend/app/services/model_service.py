import os
from sqlalchemy.orm import Session
from backend.app.ml.training import load_processed, prepare_X_y, train_models, evaluate_model, save_model
from backend.app.models.models import ModelResult
from sklearn.model_selection import train_test_split
from backend.app.database import SessionLocal
from typing import List, Dict


DEFAULT_INPUT = os.path.join("data", "processed", "preprocessed_transactions.csv")
DEFAULT_SAVE_DIR = os.path.join("backend", "app", "ml", "saved_models")


def train_and_record(db: Session, input_path: str | None = None, save_dir: str | None = None) -> List[Dict]:
    input_path = input_path or DEFAULT_INPUT
    save_dir = save_dir or DEFAULT_SAVE_DIR

    df = load_processed(input_path)
    X, y = prepare_X_y(df)
    if X.empty:
        raise ValueError("No data available for training")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if len(y.unique())>1 else None)

    models = train_models(X_train, y_train)
    results = []
    for name, model in models.items():
        metrics = evaluate_model(name, model, X_test, y_test)
        model_path, version = save_model(model, save_dir, name)

        # record in DB
        mr = ModelResult(
            model_name=name,
            version=str(version),
            accuracy=None,
            precision=metrics.get("precision"),
            recall=metrics.get("recall"),
            f1_score=metrics.get("f1_score"),
            roc_auc=metrics.get("roc_auc"),
            is_active=False,
        )
        db.add(mr)
        db.commit()
        db.refresh(mr)

        results.append({"model": name, "metrics": metrics, "path": model_path, "db_id": mr.id})

    return results


def activate_model(db: Session, model_id: int):
    # set all to False then set selected to True
    db.query(ModelResult).update({ModelResult.is_active: False})
    mr = db.query(ModelResult).filter(ModelResult.id == model_id).first()
    if not mr:
        raise ValueError("ModelResult not found")
    mr.is_active = True
    db.add(mr)
    db.commit()
    db.refresh(mr)
    return mr


def list_results(db: Session):
    rows = db.query(ModelResult).order_by(ModelResult.created_at.desc()).all()
    out = []
    for r in rows:
        out.append(
            {
                "id": r.id,
                "model_name": r.model_name,
                "version": r.version,
                "precision": r.precision,
                "recall": r.recall,
                "f1_score": r.f1_score,
                "roc_auc": r.roc_auc,
                "is_active": r.is_active,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
        )
    return out

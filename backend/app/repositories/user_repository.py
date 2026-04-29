from sqlalchemy.orm import Session
from backend.app.models.models import User


def get_user_by_email(db: Session, email: str):
    return db.query(User).filter(User.email == email).first()


def create_user(db: Session, *, full_name: str, email: str, password_hash: str, role: str = "FRAUD_ANALYST"):
    user = User(
        full_name=full_name,
        email=email,
        password_hash=password_hash,
        role=role,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

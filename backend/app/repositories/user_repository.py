from sqlalchemy.orm import Session
from backend.app.models.models import User, Role


def get_user_by_email(db: Session, email: str):
    return db.query(User).filter(User.email == email).first()


def create_user(db: Session, *, full_name: str, email: str, password_hash: str, role: str = "FRAUD_ANALYST"):
    # Attempt to resolve role code to a Role row and set role_id for referential integrity
    role_id = None
    try:
        r = db.query(Role).filter(Role.code == role).first()
        if r:
            role_id = r.id
    except Exception:
        role_id = None

    user = User(
        full_name=full_name,
        email=email,
        password_hash=password_hash,
        role=role,
        role_id=role_id,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

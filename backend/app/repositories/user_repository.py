from datetime import datetime, timezone
from sqlalchemy.orm import Session
from sqlalchemy import func
from backend.app.models.models import User, Role


def get_user_by_email(db: Session, email: str):
    return db.query(User).filter(func.lower(User.email) == func.lower(email)).first()


def get_user_by_username(db: Session, username: str):
    return db.query(User).filter(func.lower(User.username) == func.lower(username)).first()


def create_user(
    db: Session,
    *,
    full_name: str,
    email: str,
    password_hash: str,
    role: str = "FRAUD_ANALYST",
    username: str = None,
    is_active: bool = True,
):
    role_id = None
    try:
        r = db.query(Role).filter(Role.code == role).first()
        if r:
            role_id = r.id
    except Exception:
        role_id = None

    email_norm = email.lower() if isinstance(email, str) else email
    user = User(
        full_name=full_name,
        email=email_norm,
        password_hash=password_hash,
        role=role,
        role_id=role_id,
        username=username,
        is_active=is_active,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def get_user(db: Session, user_id: int):
    return db.query(User).filter(User.id == user_id).first()


def list_users(db: Session, search: str = None, role_id: int = None, is_active: bool = None):
    q = db.query(User)
    if search:
        term = f"%{search.lower()}%"
        q = q.filter(
            func.lower(User.full_name).like(term)
            | func.lower(User.email).like(term)
            | func.lower(User.username).like(term)
        )
    if role_id is not None:
        q = q.filter(User.role_id == role_id)
    if is_active is not None:
        q = q.filter(User.is_active == is_active)
    return q.order_by(User.id).all()


def update_user(db: Session, user_id: int, **fields):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        return None
    for key, value in fields.items():
        if hasattr(user, key):
            setattr(user, key, value)
    user.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(user)
    return user


def activate_user(db: Session, user_id: int):
    return update_user(db, user_id, is_active=True)


def count_active_admins(db: Session) -> int:
    return db.query(User).filter(User.role == "ADMIN", User.is_active == True).count()


def deactivate_user(db: Session, user_id: int):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        return None, None
    if user.role == "ADMIN" and count_active_admins(db) <= 1:
        return None, "No se puede desactivar al único Administrador activo del sistema."
    return update_user(db, user_id, is_active=False), None

from passlib.context import CryptContext
from sqlalchemy.orm import Session
from backend.app.repositories import user_repository
from backend.app.schemas.auth import UserCreate
from backend.app.models.models import User


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def register_user(db: Session, user_in: UserCreate) -> User:
    existing = user_repository.get_user_by_email(db, user_in.email)
    if existing:
        raise ValueError("User already exists")
    hashed = hash_password(user_in.password)
    user = user_repository.create_user(
        db, full_name=user_in.full_name, email=user_in.email, password_hash=hashed, role=user_in.role
    )
    return user


def authenticate_user(db: Session, email: str, password: str) -> User | None:
    user = user_repository.get_user_by_email(db, email)
    if not user:
        return None
    if not user.is_active:
        return None
    if not verify_password(password, user.password_hash):
        return None
    return user

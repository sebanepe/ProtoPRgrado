from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from backend.app.schemas.auth import UserCreate, UserLogin, UserResponse
from backend.app.services import auth_service
from backend.app.database import get_db

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=UserResponse)
def register(user_in: UserCreate, db: Session = Depends(get_db)):
    try:
        user = auth_service.register_user(db, user_in)
        return user
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/login", response_model=UserResponse)
def login(payload: UserLogin, db: Session = Depends(get_db)):
    user = auth_service.authenticate_user(db, payload.email, payload.password)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    return user

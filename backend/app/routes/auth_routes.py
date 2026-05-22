from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from backend.app.schemas.auth import UserCreate, UserLogin, UserResponse
from backend.app.services import auth_service
from backend.app.services.jwt_service import create_access_token
from backend.app.database import get_db
from backend.app.services.authorization import get_user_from_header

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=UserResponse)
def register(user_in: UserCreate, db: Session = Depends(get_db)):
    try:
        user = auth_service.register_user(db, user_in)
        token = create_access_token({"sub": user.email, "user_id": user.id, "role": user.role})
        resp = user.__dict__.copy()
        resp.update({"token": token})
        return resp
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/login", response_model=UserResponse)
def login(payload: UserLogin, db: Session = Depends(get_db)):
    # use strict mode to surface explicit errors to API clients
    user = auth_service.authenticate_user(db, payload.email, payload.password, raise_on_error=True)
    if not user:
        # should not reach here because raise_on_error=True will raise, but keep guard
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    token = create_access_token({"sub": user.email, "user_id": user.id, "role": user.role})
    resp = user.__dict__.copy()
    resp.update({"token": token})
    return resp


@router.get("/me", response_model=UserResponse)
def me(user=Depends(get_user_from_header)):
    # Return the current authenticated user (token not reissued)
    u = user.__dict__.copy()
    u.update({"token": None})
    return u

from datetime import datetime
from pydantic import BaseModel, EmailStr
from typing import Optional


class UserCreate(BaseModel):
    full_name: str
    email: EmailStr
    password: str
    role: Optional[str] = "FRAUD_ANALYST"


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    id: int
    full_name: str
    email: EmailStr
    role: str
    is_active: bool
    created_at: datetime
    token: Optional[str] = None

    class Config:
        orm_mode = True

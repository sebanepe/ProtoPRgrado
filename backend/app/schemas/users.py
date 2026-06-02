from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, EmailStr


class UserAdminCreate(BaseModel):
    username: str
    email: EmailStr
    full_name: str
    role_id: int
    password: str
    is_active: bool = True


class UserAdminUpdate(BaseModel):
    username: Optional[str] = None
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    role_id: Optional[int] = None
    is_active: Optional[bool] = None
    password: Optional[str] = None


class UserAdminResponse(BaseModel):
    id: int
    username: Optional[str]
    email: str
    full_name: str
    role_id: Optional[int]
    role: str
    is_active: bool
    created_at: datetime
    updated_at: Optional[datetime]

    class Config:
        orm_mode = True


class RoleOut(BaseModel):
    id: int
    code: str
    name: str
    description: Optional[str]

    class Config:
        orm_mode = True


class PermissionOut(BaseModel):
    id: int
    code: str
    module: str
    action: str
    description: Optional[str]

    class Config:
        orm_mode = True

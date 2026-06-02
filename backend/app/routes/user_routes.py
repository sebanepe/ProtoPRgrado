from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from typing import List, Optional

from backend.app.database import get_db
from backend.app.models.models import Role, Permission, RolePermission
from backend.app.repositories import user_repository
from backend.app.schemas.users import (
    UserAdminCreate,
    UserAdminUpdate,
    UserAdminResponse,
    RoleOut,
    PermissionOut,
)
from backend.app.services import auth_service
from backend.app.services.authorization import get_user_from_header
from backend.app.services.permission_service import require_permission

ALLOWED_ROLE_CODES = {"ADMIN", "DATA_SCIENTIST", "FRAUD_ANALYST"}

router = APIRouter(prefix="/api/users", tags=["users"])
roles_router = APIRouter(prefix="/api/roles", tags=["roles"])


def _resolve_role(db: Session, role_id: int) -> Role:
    role = db.query(Role).filter(Role.id == role_id).first()
    if not role:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="role_id inválido.")
    if role.code not in ALLOWED_ROLE_CODES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Rol no permitido. Roles válidos: {', '.join(sorted(ALLOWED_ROLE_CODES))}.",
        )
    return role


@router.get("", response_model=List[UserAdminResponse])
def list_users(
    search: Optional[str] = Query(None),
    role_id: Optional[int] = Query(None),
    is_active: Optional[bool] = Query(None),
    db: Session = Depends(get_db),
    _: bool = Depends(require_permission("users.view")),
):
    users = user_repository.list_users(db, search=search, role_id=role_id, is_active=is_active)
    return users


@router.post("", response_model=UserAdminResponse, status_code=status.HTTP_201_CREATED)
def create_user(
    payload: UserAdminCreate,
    db: Session = Depends(get_db),
    _: bool = Depends(require_permission("users.create")),
):
    if not payload.username or not payload.username.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="username es requerido.")
    if not payload.password or len(payload.password) < 8:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="La contraseña debe tener al menos 8 caracteres.")

    if user_repository.get_user_by_email(db, payload.email):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="El email ya está registrado.")
    if user_repository.get_user_by_username(db, payload.username):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="El username ya está registrado.")

    role = _resolve_role(db, payload.role_id)
    pw_hash = auth_service.hash_password(payload.password)

    user = user_repository.create_user(
        db,
        full_name=payload.full_name,
        email=payload.email,
        password_hash=pw_hash,
        role=role.code,
        username=payload.username.strip(),
        is_active=payload.is_active,
    )
    return user


@router.get("/{user_id}", response_model=UserAdminResponse)
def get_user(
    user_id: int,
    db: Session = Depends(get_db),
    _: bool = Depends(require_permission("users.view")),
):
    user = user_repository.get_user(db, user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Usuario no encontrado.")
    return user


@router.patch("/{user_id}", response_model=UserAdminResponse)
def update_user(
    user_id: int,
    payload: UserAdminUpdate,
    db: Session = Depends(get_db),
    _: bool = Depends(require_permission("users.update")),
):
    user = user_repository.get_user(db, user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Usuario no encontrado.")

    fields = {}

    if payload.username is not None:
        uname = payload.username.strip()
        existing = user_repository.get_user_by_username(db, uname)
        if existing and existing.id != user_id:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="El username ya está registrado.")
        fields["username"] = uname

    if payload.email is not None:
        existing = user_repository.get_user_by_email(db, payload.email)
        if existing and existing.id != user_id:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="El email ya está registrado.")
        fields["email"] = payload.email.lower()

    if payload.full_name is not None:
        fields["full_name"] = payload.full_name

    if payload.role_id is not None:
        role = _resolve_role(db, payload.role_id)
        fields["role_id"] = role.id
        fields["role"] = role.code

    if payload.is_active is not None:
        fields["is_active"] = payload.is_active

    if payload.password and payload.password.strip():
        if len(payload.password) < 8:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="La contraseña debe tener al menos 8 caracteres.")
        fields["password_hash"] = auth_service.hash_password(payload.password)

    updated = user_repository.update_user(db, user_id, **fields)
    return updated


@router.post("/{user_id}/activate", response_model=UserAdminResponse)
def activate_user(
    user_id: int,
    db: Session = Depends(get_db),
    _: bool = Depends(require_permission("users.activate")),
):
    user = user_repository.get_user(db, user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Usuario no encontrado.")
    activated = user_repository.activate_user(db, user_id)
    return activated


@router.post("/{user_id}/deactivate", response_model=UserAdminResponse)
def deactivate_user(
    user_id: int,
    db: Session = Depends(get_db),
    _: bool = Depends(require_permission("users.deactivate")),
):
    user = user_repository.get_user(db, user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Usuario no encontrado.")
    updated, err = user_repository.deactivate_user(db, user_id)
    if err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=err)
    return updated


# ── Roles router ──────────────────────────────────────────────────────────────

@roles_router.get("", response_model=List[RoleOut])
def list_roles(
    db: Session = Depends(get_db),
    _=Depends(get_user_from_header),
):
    roles = db.query(Role).filter(Role.code.in_(ALLOWED_ROLE_CODES)).order_by(Role.id).all()
    return roles


@roles_router.get("/{role_id}/permissions", response_model=List[PermissionOut])
def get_role_permissions(
    role_id: int,
    db: Session = Depends(get_db),
    _=Depends(get_user_from_header),
):
    role = db.query(Role).filter(Role.id == role_id).first()
    if not role:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Rol no encontrado.")
    perms = (
        db.query(Permission)
        .join(RolePermission, Permission.id == RolePermission.permission_id)
        .filter(RolePermission.role_id == role_id)
        .all()
    )
    return perms

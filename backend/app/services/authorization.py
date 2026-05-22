from fastapi import Header, HTTPException, status, Depends
from sqlalchemy.orm import Session
from typing import Optional, Set

from backend.app.repositories import user_repository
from backend.app.database import get_db
from backend.app.services.jwt_service import decode_access_token


# Permission strings
PERMISSIONS = {
    "FRAUD_ANALYST": {
        "view_dashboard",
        "scoring",
        "view_alerts",
        "alert_detail",
        "change_alert_state",
    },
    "DATA_SCIENTIST": {
        "view_dashboard",
        "import_data",
        "preprocess",
        "train",
        "evaluate",
        "scoring",
        "configure_model",
    },
    "ADMIN": {
        # ADMIN has full privileges for QA and management
        "view_dashboard",
        "view_alerts",
        "manage_users",
        "configure_model",
        "import_data",
        "preprocess",
        "train",
        "evaluate",
        "scoring",
        "alert_detail",
        "change_alert_state",
    },
}


def get_user_from_header(
    authorization: Optional[str] = Header(None),
    x_user_email: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    """Resolve current user from `Authorization: Bearer <token>` (JWT) or fallback to `X-User-Email` header.
    This maintains backward compatibility with clients that still send the legacy header.
    """
    # Prefer Authorization Bearer token
    if authorization:
        try:
            scheme, token = authorization.split(" ", 1)
            if scheme.lower() != "bearer":
                raise ValueError("Unsupported auth scheme")
        except Exception:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid Authorization header")
        try:
            payload = decode_access_token(token)
        except Exception:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")
        # prefer user_id if present, otherwise fall back to sub (email)
        user = None
        user_id = payload.get("user_id")
        if user_id:
            user = user_repository.get_user(db, user_id)
        if not user:
            sub = payload.get("sub")
            if sub:
                user = user_repository.get_user_by_email(db, sub)
        if not user:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
        if not user.is_active:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User inactive")
        return user

    # Fallback to legacy header
    if not x_user_email:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing Authorization or X-User-Email header")
    user = user_repository.get_user_by_email(db, x_user_email)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User inactive")
    return user


def require_permission(permission: str):
    def _dependency(user=Depends(get_user_from_header)):
        role = (user.role or "").upper()
        allowed: Set[str] = PERMISSIONS.get(role, set())
        if permission not in allowed:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions")
        return True

    return _dependency


def require_role(*roles: str):
    roles_upper = {r.upper() for r in roles}

    def _dependency(user=Depends(get_user_from_header)):
        if (user.role or "").upper() not in roles_upper:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient role")
        return True

    return _dependency

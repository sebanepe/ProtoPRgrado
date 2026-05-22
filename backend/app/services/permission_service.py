from sqlalchemy.orm import Session
from backend.app.models.models import Permission, Role, RolePermission, User


def get_permissions_for_role(db: Session, role_code: str):
    role = db.query(Role).filter(Role.code == role_code).first()
    if not role:
        return []
    perms = db.query(Permission).join(RolePermission, Permission.id == RolePermission.permission_id).filter(RolePermission.role_id == role.id).all()
    return [p.code for p in perms]


def get_user_permissions(db: Session, user_id: int):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        return []
    # first try role_id relation
    if getattr(user, "role_id", None):
        return get_permissions_for_role(db, db.query(Role).filter(Role.id == user.role_id).first().code)
    # fallback to role string mapping
    role_code = (user.role or "").upper()
    return get_permissions_for_role(db, role_code)


def user_has_permission(db: Session, user: User, permission_code: str) -> bool:
    if not user:
        return False
    # admin shortcut
    if (user.role or "").upper() == "ADMIN":
        return True
    perms = get_user_permissions(db, user.id)
    # support legacy permission names mapping to new codes
    LEGACY_MAP = {
        "import_data": "dataset.import",
        "preprocess": "preprocessing.run",
        "train": "model.train",
        "evaluate": "model.evaluate",
        "configure_model": "model.configure_threshold",
        "scoring": "scoring.run",
        "view_alerts": "alerts.view",
        "alert_detail": "alerts.detail",
        "change_alert_state": "alerts.update_status",
        "view_dashboard": "dashboard.view",
        "users.manage": "users.update",
    }

    if permission_code in perms:
        return True
    mapped = LEGACY_MAP.get(permission_code)
    if mapped and mapped in perms:
        return True
    # also try swapping underscore/dot variants
    alt = permission_code.replace("_", ".")
    if alt in perms:
        return True
    return False


# FastAPI dependency factory
from fastapi import Depends, HTTPException, status
from backend.app.database import get_db
from backend.app.services.authorization import get_user_from_header


def require_permission(permission_code: str):
    def _dep(user=Depends(get_user_from_header), db: Session = Depends(get_db)):
        if not user_has_permission(db, user, permission_code):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions")
        return True

    return _dep

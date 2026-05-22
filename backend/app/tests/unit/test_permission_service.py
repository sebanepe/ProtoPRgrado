import pytest
from fastapi import HTTPException

from backend.app.services.permission_service import (
    get_permissions_for_role,
    get_user_permissions,
    user_has_permission,
    require_permission,
)
from backend.app.models.models import Role, Permission, RolePermission, User


def test_permissions_and_user_permissions(db_session):
    # create role and permission
    role = Role(code="DATA_SCIENTIST", name="Data Scientist")
    db_session.add(role)
    db_session.commit()
    db_session.refresh(role)

    perm = Permission(code="dataset.import", module="dataset", action="import")
    db_session.add(perm)
    db_session.commit()
    db_session.refresh(perm)

    rp = RolePermission(role_id=role.id, permission_id=perm.id)
    db_session.add(rp)
    db_session.commit()

    user = User(full_name="Tester", email="t@example.com", password_hash="x", role_id=role.id)
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)

    perms = get_permissions_for_role(db_session, "DATA_SCIENTIST")
    assert "dataset.import" in perms  # permiso asignado al rol debe aparecer

    user_perms = get_user_permissions(db_session, user.id)
    assert "dataset.import" in user_perms  # permiso del rol debe reflejarse en el usuario

    # comprobación directa de permiso
    assert user_has_permission(db_session, user, "dataset.import") is True  # usuario tiene permiso
    # la antigua clave 'import_data' debe mapearse correctamente
    assert user_has_permission(db_session, user, "import_data") is True  # legacy mapping


def test_admin_shortcut_grants_all(db_session):
    user = User(full_name="Admin", email="admin@example.com", password_hash="x", role="admin")
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)

    assert user_has_permission(db_session, user, "any.permission") is True  # shortcut admin -> true


def test_require_permission_dependency_allows_and_denies(db_session):
    # allowed case
    role = Role(code="ANALYST", name="Analyst")
    db_session.add(role)
    db_session.commit()
    db_session.refresh(role)

    perm = Permission(code="alerts.view", module="alerts", action="view")
    db_session.add(perm)
    db_session.commit()
    db_session.refresh(perm)

    rp = RolePermission(role_id=role.id, permission_id=perm.id)
    db_session.add(rp)
    db_session.commit()

    user = User(full_name="Viewer", email="v@example.com", password_hash="x", role_id=role.id)
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)

    dep = require_permission("view_alerts")
    # call dependency directamente con el usuario creado
    assert dep(user=user, db=db_session) is True  # dependencia debe permitir al usuario con el permiso

    # denied case
    role2 = Role(code="LIMITED", name="Limited")
    db_session.add(role2)
    db_session.commit()
    db_session.refresh(role2)

    user2 = User(full_name="NoPerm", email="n@example.com", password_hash="x", role_id=role2.id)
    db_session.add(user2)
    db_session.commit()
    db_session.refresh(user2)

    dep2 = require_permission("view_alerts")
    with pytest.raises(HTTPException) as exc:
        dep2(user=user2, db=db_session)
    assert exc.value.status_code == 403  # usuario sin permiso debe provocar 403

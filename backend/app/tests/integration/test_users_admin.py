"""Integration tests for /api/users and /api/roles endpoints."""
import uuid
import pytest
from fastapi.testclient import TestClient

from backend.app.models.models import Role


def _uid() -> str:
    return uuid.uuid4().hex[:8]


def _email(prefix: str = "") -> str:
    return f"{prefix or 'user'}_{_uid()}@example.com"


@pytest.fixture(autouse=True)
def seed_roles(db_session):
    """Ensure the three permitted roles exist for every test."""
    for code, name in [
        ("ADMIN", "Administrador"),
        ("FRAUD_ANALYST", "Analista de Fraude"),
        ("DATA_SCIENTIST", "Científico de Datos"),
    ]:
        if not db_session.query(Role).filter(Role.code == code).first():
            db_session.add(Role(code=code, name=name, description=name))
    db_session.commit()


@pytest.fixture(autouse=True)
def link_admin_role_id(seed_roles, db_session):
    """Link the default test admin user to its ADMIN role row."""
    from backend.app.models.models import User
    admin_role = db_session.query(Role).filter(Role.code == "ADMIN").first()
    if admin_role:
        db_session.query(User).filter(User.role == "ADMIN").update({"role_id": admin_role.id})
        db_session.commit()


def _role_id(db_session, code: str) -> int:
    role = db_session.query(Role).filter(Role.code == code).first()
    assert role is not None, f"Role {code} not found — seed_roles must run first."
    return role.id


def _create_user(client: TestClient, db_session, role_code="FRAUD_ANALYST", **overrides):
    rid = _role_id(db_session, role_code)
    name = f"u_{_uid()}"
    payload = {
        "username": name,
        "email": _email(name),
        "full_name": "Test User",
        "role_id": rid,
        "password": "Temporal123",
        "is_active": True,
        **overrides,
    }
    resp = client.post("/api/users", json=payload)
    assert resp.status_code == 201, f"create_user failed: {resp.json()}"
    return resp.json()


# ── GET /api/users ────────────────────────────────────────────────────────────

def test_list_users_returns_list(test_client: TestClient):
    resp = test_client.get("/api/users")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_list_users_does_not_contain_password_hash(test_client: TestClient):
    resp = test_client.get("/api/users")
    assert resp.status_code == 200
    for user in resp.json():
        assert "password_hash" not in user
        assert "password" not in user


# ── POST /api/users ───────────────────────────────────────────────────────────

def test_create_user_success(test_client: TestClient, db_session):
    rid = _role_id(db_session, "FRAUD_ANALYST")
    name = f"analyst_{_uid()}"
    payload = {
        "username": name,
        "email": _email(name),
        "full_name": "Analista Test",
        "role_id": rid,
        "password": "Temporal123",
        "is_active": True,
    }
    resp = test_client.post("/api/users", json=payload)
    assert resp.status_code == 201, resp.json()
    data = resp.json()
    assert data["username"] == name
    assert data["email"] == payload["email"]
    assert "password_hash" not in data
    assert "password" not in data


def test_create_user_duplicate_email(test_client: TestClient, db_session):
    user = _create_user(test_client, db_session)
    rid = _role_id(db_session, "FRAUD_ANALYST")
    payload2 = {
        "username": f"other_{_uid()}",
        "email": user["email"],
        "full_name": "X",
        "role_id": rid,
        "password": "Temporal123",
        "is_active": True,
    }
    resp = test_client.post("/api/users", json=payload2)
    assert resp.status_code == 400
    assert "email" in resp.json()["detail"].lower()


def test_create_user_duplicate_username(test_client: TestClient, db_session):
    user = _create_user(test_client, db_session)
    rid = _role_id(db_session, "FRAUD_ANALYST")
    payload2 = {
        "username": user["username"],
        "email": _email("other"),
        "full_name": "X",
        "role_id": rid,
        "password": "Temporal123",
        "is_active": True,
    }
    resp = test_client.post("/api/users", json=payload2)
    assert resp.status_code == 400
    assert "username" in resp.json()["detail"].lower()


def test_create_user_invalid_role_id(test_client: TestClient):
    payload = {
        "username": f"badrole_{_uid()}",
        "email": _email("badrole"),
        "full_name": "X",
        "role_id": 99999,
        "password": "Temporal123",
        "is_active": True,
    }
    resp = test_client.post("/api/users", json=payload)
    assert resp.status_code == 400


def test_create_user_short_password(test_client: TestClient, db_session):
    rid = _role_id(db_session, "FRAUD_ANALYST")
    payload = {
        "username": f"shortpw_{_uid()}",
        "email": _email("shortpw"),
        "full_name": "X",
        "role_id": rid,
        "password": "abc",
        "is_active": True,
    }
    resp = test_client.post("/api/users", json=payload)
    assert resp.status_code == 400


# ── PATCH /api/users/{id} ────────────────────────────────────────────────────

def test_update_user_full_name(test_client: TestClient, db_session):
    user = _create_user(test_client, db_session)
    resp = test_client.patch(f"/api/users/{user['id']}", json={"full_name": "Nombre Nuevo"})
    assert resp.status_code == 200
    assert resp.json()["full_name"] == "Nombre Nuevo"


def test_update_user_does_not_change_password_when_omitted(test_client: TestClient, db_session):
    from backend.app.repositories.user_repository import get_user
    user = _create_user(test_client, db_session)
    original_hash = get_user(db_session, user["id"]).password_hash
    test_client.patch(f"/api/users/{user['id']}", json={"full_name": "Sin Cambio PW"})
    db_session.expire_all()
    assert get_user(db_session, user["id"]).password_hash == original_hash


def test_update_user_password_if_provided(test_client: TestClient, db_session):
    from backend.app.repositories.user_repository import get_user
    user = _create_user(test_client, db_session)
    original_hash = get_user(db_session, user["id"]).password_hash
    test_client.patch(f"/api/users/{user['id']}", json={"password": "NuevaClave99"})
    db_session.expire_all()
    assert get_user(db_session, user["id"]).password_hash != original_hash


def test_update_user_response_no_password_hash(test_client: TestClient, db_session):
    user = _create_user(test_client, db_session)
    resp = test_client.patch(f"/api/users/{user['id']}", json={"full_name": "Safe"})
    assert resp.status_code == 200
    assert "password_hash" not in resp.json()


# ── Activate / Deactivate ────────────────────────────────────────────────────

def test_deactivate_then_activate_user(test_client: TestClient, db_session):
    user = _create_user(test_client, db_session)
    uid = user["id"]

    r_deact = test_client.post(f"/api/users/{uid}/deactivate")
    assert r_deact.status_code == 200, r_deact.json()
    assert r_deact.json()["is_active"] is False

    r_act = test_client.post(f"/api/users/{uid}/activate")
    assert r_act.status_code == 200, r_act.json()
    assert r_act.json()["is_active"] is True


def test_deactivate_only_admin_blocked(test_client: TestClient, db_session):
    """Deactivating the sole active ADMIN must be rejected."""
    from backend.app.models.models import User
    admin = db_session.query(User).filter(User.role == "ADMIN", User.is_active == True).first()
    assert admin is not None
    resp = test_client.post(f"/api/users/{admin.id}/deactivate")
    assert resp.status_code == 400
    assert "Administrador" in resp.json()["detail"]


def test_deactivate_admin_allowed_when_another_exists(test_client: TestClient, db_session):
    """Deactivating one ADMIN is allowed when a second active ADMIN exists."""
    second_admin = _create_user(test_client, db_session, role_code="ADMIN")
    resp = test_client.post(f"/api/users/{second_admin['id']}/deactivate")
    assert resp.status_code == 200, resp.json()
    assert resp.json()["is_active"] is False


# ── GET /api/roles ────────────────────────────────────────────────────────────

def test_list_roles_returns_three(test_client: TestClient):
    resp = test_client.get("/api/roles")
    assert resp.status_code == 200
    codes = {r["code"] for r in resp.json()}
    assert codes == {"ADMIN", "DATA_SCIENTIST", "FRAUD_ANALYST"}


def test_list_roles_no_consulta_or_viewer(test_client: TestClient):
    resp = test_client.get("/api/roles")
    assert resp.status_code == 200
    names = [r["name"].lower() for r in resp.json()]
    for forbidden in ("consulta", "viewer", "read only"):
        assert not any(forbidden in n for n in names)


def test_get_role_permissions(test_client: TestClient, db_session):
    role = db_session.query(Role).filter(Role.code == "ADMIN").first()
    resp = test_client.get(f"/api/roles/{role.id}/permissions")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)

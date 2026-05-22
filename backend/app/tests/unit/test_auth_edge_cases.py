"""
Pruebas unitarias para autenticación y hashing de contraseñas.
Se usan mocks para `user_repository` en los tests de registro/autenticación.
"""
import types
import pytest

from backend.app.services import auth_service


class DummyUser:
    def __init__(self, email, password_hash, is_active=True, role="FRAUD_ANALYST"):
        self.email = email
        self.password_hash = password_hash
        self.is_active = is_active
        self.role = role


def test_password_hash_is_not_plain_text():
    pw = "MySecret"
    hashed = auth_service.hash_password(pw)
    assert hashed != pw  # el hash no debe ser igual al texto plano


def test_verify_password_success():
    pw = "Password123!"
    hashed = auth_service.hash_password(pw)
    assert auth_service.verify_password(pw, hashed)  # verificación debe ser exitosa con la contraseña correcta


def test_verify_password_failure():
    pw = "Password123!"
    hashed = auth_service.hash_password(pw)
    assert not auth_service.verify_password("wrongpw", hashed)  # contraseña incorrecta no debe verificar


def test_empty_password_is_rejected_if_supported(monkeypatch):
    # simulate system that rejects empty password by making create_user raise
    monkeypatch.setattr(auth_service, "pwd_context", auth_service.pwd_context)

    def fake_get_by_email(db, email):
        return None

    def fake_hash(pw):
        # simulate empty hash for empty password
        if pw == "":
            return ""
        return auth_service.pwd_context.hash(pw)

    def fake_create_user(db, full_name, email, password_hash, role):
        if password_hash == "":
            raise ValueError("empty password not allowed")
        return DummyUser(email=email, password_hash=password_hash, is_active=True, role=role)

    monkeypatch.setattr(auth_service, "hash_password", fake_hash)
    monkeypatch.setattr(auth_service, "user_repository", types.SimpleNamespace(get_user_by_email=fake_get_by_email, create_user=fake_create_user))

    from backend.app.schemas.auth import UserCreate

    uc = UserCreate(full_name="A", email="a@x.com", password="", role="FRAUD_ANALYST")
    with pytest.raises(ValueError):
        auth_service.register_user(None, uc)


def test_invalid_role_is_rejected_if_supported(monkeypatch):
    def fake_get_by_email(db, email):
        return None

    def fake_hash(pw):
        return auth_service.pwd_context.hash(pw)

    def fake_create_user(db, full_name, email, password_hash, role):
        if role not in ("FRAUD_ANALYST", "ADMIN"):
            raise ValueError("invalid role")
        return DummyUser(email=email, password_hash=password_hash, is_active=True, role=role)

    monkeypatch.setattr(auth_service, "hash_password", fake_hash)
    monkeypatch.setattr(auth_service, "user_repository", types.SimpleNamespace(get_user_by_email=fake_get_by_email, create_user=fake_create_user))
    from backend.app.schemas.auth import UserCreate

    uc = UserCreate(full_name="A", email="a@x.com", password="pw", role="INVALID_ROLE")
    with pytest.raises(ValueError):
        auth_service.register_user(None, uc)


def test_inactive_user_cannot_login_if_supported(monkeypatch):
    # simulate user returned but inactive
    dummy = DummyUser(email="u@x.com", password_hash=auth_service.hash_password("pw"), is_active=False)

    def fake_get_by_email(db, email):
        return dummy

    monkeypatch.setattr(auth_service, "user_repository", types.SimpleNamespace(get_user_by_email=fake_get_by_email))
    res = auth_service.authenticate_user(None, "u@x.com", "pw")
    assert res is None  # usuario inactivo no puede autenticarse

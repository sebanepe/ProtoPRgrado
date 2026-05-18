from backend.app.models.models import User


def test_register_user_integration(test_client):
    payload = {"full_name": "Alice", "email": "alice@example.com", "password": "pw123", "role": "user"}
    r = test_client.post("/auth/register", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data.get("email") == "alice@example.com"


def test_login_user_integration(test_client):
    payload = {"full_name": "Bob", "email": "bob@example.com", "password": "pw123", "role": "user"}
    r = test_client.post("/auth/register", json=payload)
    assert r.status_code == 200
    login = {"email": "bob@example.com", "password": "pw123"}
    r2 = test_client.post("/auth/login", json=login)
    assert r2.status_code == 200
    assert r2.json().get("email") == "bob@example.com"


def test_login_with_wrong_password_fails(test_client):
    payload = {"full_name": "Eve", "email": "eve@example.com", "password": "rightpw", "role": "user"}
    r = test_client.post("/auth/register", json=payload)
    assert r.status_code == 200
    bad = {"email": "eve@example.com", "password": "wrongpw"}
    r2 = test_client.post("/auth/login", json=bad)
    assert r2.status_code == 401

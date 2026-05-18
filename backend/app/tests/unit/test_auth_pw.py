from backend.app.services import auth_service

"""Unit test for password hashing and verification (extracted)."""


def test_password_hash_and_verify_extracted():
    plain = "SuperSecret123!"
    hashed = auth_service.hash_password(plain)
    assert auth_service.verify_password(plain, hashed)

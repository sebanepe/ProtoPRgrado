from backend.app.services import auth_service

"""Unit tests for auth hashing and verification functions."""

def test_hash_and_verify_password():
    pw = 'SomePassword123!'
    hashed = auth_service.hash_password(pw)
    assert auth_service.verify_password(pw, hashed)

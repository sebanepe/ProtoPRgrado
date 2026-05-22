from backend.app.services import auth_service

"""Prueba unitaria para funciones de hashing y verificación de contraseña.

Verifica que una contraseña hasheada pueda ser verificada correctamente.
"""


def test_hash_and_verify_password():
    pw = 'SomePassword123!'
    hashed = auth_service.hash_password(pw)
    assert auth_service.verify_password(pw, hashed)  # la contraseña original debe coincidir con el hash

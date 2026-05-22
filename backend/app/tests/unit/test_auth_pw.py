from backend.app.services import auth_service

"""Test de hashing/verificación de contraseña.

Confirma que el hash generado permite verificar la contraseña original.
"""


def test_password_hash_and_verify_extracted():
    plain = "SuperSecret123!"
    hashed = auth_service.hash_password(plain)
    assert auth_service.verify_password(plain, hashed)  # verificación correcta

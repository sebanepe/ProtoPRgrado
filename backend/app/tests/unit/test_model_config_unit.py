"""
Pruebas unitarias para la configuración del modelo activo.
Se mockean funciones del repositorio para evitar acceso a la base de datos.
"""
import types
import pytest

from backend.app.services import settings_service


class DummyConfig:
    def __init__(self, active_model_id=None, alert_threshold=0.7, is_active=True):
        self.active_model_id = active_model_id
        self.alert_threshold = alert_threshold
        self.is_active = is_active


def test_valid_threshold_is_accepted(monkeypatch):
    # mock create_config to just return a DummyConfig
    def fake_create(db, *, active_model_id, alert_threshold, updated_by):
        return DummyConfig(active_model_id=active_model_id, alert_threshold=alert_threshold)

    monkeypatch.setattr(settings_service, "model_config_repository", types.SimpleNamespace(create_config=fake_create))
    cfg = settings_service.set_model_config(None, active_model_id=1, alert_threshold=0.75, updated_by="u")
    assert cfg.alert_threshold == 0.75


def test_threshold_less_than_zero_is_rejected(monkeypatch):
    # Simulate repository raising ValueError for invalid threshold
    def fake_create(db, *, active_model_id, alert_threshold, updated_by):
        if alert_threshold < 0.0:
            raise ValueError("invalid threshold")
        return DummyConfig(active_model_id=active_model_id, alert_threshold=alert_threshold)

    monkeypatch.setattr(settings_service, "model_config_repository", types.SimpleNamespace(create_config=fake_create))
    with pytest.raises(ValueError):
        settings_service.set_model_config(None, active_model_id=1, alert_threshold=-0.1)


def test_threshold_greater_than_one_is_rejected(monkeypatch):
    def fake_create(db, *, active_model_id, alert_threshold, updated_by):
        if alert_threshold > 1.0:
            raise ValueError("invalid threshold")
        return DummyConfig(active_model_id=active_model_id, alert_threshold=alert_threshold)

    monkeypatch.setattr(settings_service, "model_config_repository", types.SimpleNamespace(create_config=fake_create))
    with pytest.raises(ValueError):
        settings_service.set_model_config(None, active_model_id=1, alert_threshold=1.1)


def test_threshold_zero_is_valid_if_allowed(monkeypatch):
    def fake_create(db, *, active_model_id, alert_threshold, updated_by):
        return DummyConfig(active_model_id=active_model_id, alert_threshold=alert_threshold)

    monkeypatch.setattr(settings_service, "model_config_repository", types.SimpleNamespace(create_config=fake_create))
    cfg = settings_service.set_model_config(None, active_model_id=2, alert_threshold=0.0)
    assert cfg.alert_threshold == 0.0


def test_threshold_one_is_valid_if_allowed(monkeypatch):
    def fake_create(db, *, active_model_id, alert_threshold, updated_by):
        return DummyConfig(active_model_id=active_model_id, alert_threshold=alert_threshold)

    monkeypatch.setattr(settings_service, "model_config_repository", types.SimpleNamespace(create_config=fake_create))
    cfg = settings_service.set_model_config(None, active_model_id=3, alert_threshold=1.0)
    assert cfg.alert_threshold == 1.0


def test_create_first_active_model_config(monkeypatch):
    # create_config should be called and return an active config
    called = {}

    def fake_create(db, *, active_model_id, alert_threshold, updated_by):
        called['created'] = True
        return DummyConfig(active_model_id=active_model_id, alert_threshold=alert_threshold, is_active=True)

    monkeypatch.setattr(settings_service, "model_config_repository", types.SimpleNamespace(create_config=fake_create, get_active_config=lambda db: None))
    cfg = settings_service.set_model_config(None, active_model_id=5, alert_threshold=0.5)
    assert called.get('created') is True
    assert cfg.is_active


def test_new_active_config_deactivates_previous_one(monkeypatch):
    # simulate existing active config; create_config should still return new active
    prev = DummyConfig(active_model_id=1, alert_threshold=0.3, is_active=True)

    def fake_get_active(db):
        return prev

    def fake_create(db, *, active_model_id, alert_threshold, updated_by):
        # simulate deactivation by setting prev.is_active False
        prev.is_active = False
        return DummyConfig(active_model_id=active_model_id, alert_threshold=alert_threshold, is_active=True)

    monkeypatch.setattr(settings_service, "model_config_repository", types.SimpleNamespace(get_active_config=fake_get_active, create_config=fake_create))
    cfg = settings_service.set_model_config(None, active_model_id=9, alert_threshold=0.6)
    assert cfg.is_active
    assert prev.is_active is False


def test_get_active_config_when_exists(monkeypatch):
    exp = DummyConfig(active_model_id=7, alert_threshold=0.33)
    monkeypatch.setattr(settings_service, "model_config_repository", types.SimpleNamespace(get_active_config=lambda db: exp))
    res = settings_service.get_active_config(None)
    assert res is exp


def test_get_active_config_when_missing_returns_controlled_result(monkeypatch):
    monkeypatch.setattr(settings_service, "model_config_repository", types.SimpleNamespace(get_active_config=lambda db: None))
    res = settings_service.get_active_config(None)
    assert res is None

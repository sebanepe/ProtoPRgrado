"""
Pruebas unitarias para `alert_service`.
Se mockean el modelo activo, el modelo cargado, las funciones de scoring y el repositorio de alertas.
"""
import types
import pytest

from backend.app.services import alert_service


class DummyModelRow:
    def __init__(self, model_name="m", version="v1"):
        self.model_name = model_name
        self.version = version


class DummyAlert:
    def __init__(self, id=1, transaction_id=0, risk_score=0.0, risk_level="LOW"):
        self.id = id
        self.transaction_id = transaction_id
        self.risk_score = risk_score
        self.risk_level = risk_level


def test_alert_not_created_when_score_below_threshold(monkeypatch):
    # active model exists
    monkeypatch.setattr(alert_service, "_get_active_model_row", lambda db: DummyModelRow())
    # mock loading model file to avoid filesystem operations
    monkeypatch.setattr(alert_service, "load_model_by_info", lambda name, ver, models_dir=None: object())
    # model returns scores below threshold
    monkeypatch.setattr(alert_service, "risk_score_from_model", lambda model, name, X: [0.3])
    # threshold is 0.5
    monkeypatch.setattr(alert_service.settings_service, "get_active_threshold", lambda db: 0.5)

    called = {"created": False}

    def fake_create_alert(db, transaction_id, risk_score, risk_level, model_name, status):
        called["created"] = True
        return DummyAlert(id=99, transaction_id=transaction_id, risk_score=risk_score, risk_level=risk_level)

    monkeypatch.setattr(alert_service.alert_repository, "create_alert", fake_create_alert)
    res = alert_service.generate_alerts_from_batch(None, [{"transaction_id": "tx1", "amount": 1.0}])
    assert res == []
    assert called["created"] is False
    # Si la puntuación está por debajo del umbral, no debe crearse alerta


def test_alert_created_when_score_equals_threshold(monkeypatch):
    monkeypatch.setattr(alert_service, "_get_active_model_row", lambda db: DummyModelRow(model_name="mname"))
    monkeypatch.setattr(alert_service, "load_model_by_info", lambda name, ver, models_dir=None: object())
    monkeypatch.setattr(alert_service, "risk_score_from_model", lambda model, name, X: [0.5])
    monkeypatch.setattr(alert_service.settings_service, "get_active_threshold", lambda db: 0.5)

    created_args = {}

    def fake_create_alert(db, transaction_id, risk_score, risk_level, model_name, status):
        created_args.update({"transaction_id": transaction_id, "risk_score": risk_score, "risk_level": risk_level, "model_name": model_name, "status": status})
        return DummyAlert(id=10, transaction_id=transaction_id, risk_score=risk_score, risk_level=risk_level)

    monkeypatch.setattr(alert_service.alert_repository, "create_alert", fake_create_alert)
    out = alert_service.generate_alerts_from_batch(None, [{"transaction_id": "123", "amount": 2.0}])
    assert len(out) == 1
    assert created_args["risk_score"] == 0.5  # la puntuación usada debe ser la esperada (0.5)
    assert created_args["status"] == "NEW"  # el estado por defecto para nuevas alertas es NEW


def test_alert_created_when_score_above_threshold(monkeypatch):
    monkeypatch.setattr(alert_service, "_get_active_model_row", lambda db: DummyModelRow(model_name="mname"))
    monkeypatch.setattr(alert_service, "load_model_by_info", lambda name, ver, models_dir=None: object())
    monkeypatch.setattr(alert_service, "risk_score_from_model", lambda model, name, X: [0.9])
    monkeypatch.setattr(alert_service.settings_service, "get_active_threshold", lambda db: 0.8)

    def fake_create_alert(db, transaction_id, risk_score, risk_level, model_name, status):
        return DummyAlert(id=11, transaction_id=transaction_id, risk_score=risk_score, risk_level=risk_level)

    monkeypatch.setattr(alert_service.alert_repository, "create_alert", fake_create_alert)
    res = alert_service.generate_alerts_from_batch(None, [{"transaction_id": "abc", "amount": 9.0}])
    assert len(res) == 1
    assert res[0]["risk_level"] in ("LOW", "MEDIUM", "HIGH")  # nivel de riesgo debe ser uno de los permitidos


def test_alert_uses_threshold_from_model_config(monkeypatch):
    monkeypatch.setattr(alert_service, "_get_active_model_row", lambda db: DummyModelRow())
    monkeypatch.setattr(alert_service, "load_model_by_info", lambda name, ver, models_dir=None: object())
    monkeypatch.setattr(alert_service, "risk_score_from_model", lambda model, name, X: [0.75])

    # use a high threshold so 0.75 won't create alert
    monkeypatch.setattr(alert_service.settings_service, "get_active_threshold", lambda db: 0.8)

    created = []
    monkeypatch.setattr(alert_service.alert_repository, "create_alert", lambda db, **kwargs: created.append(kwargs) or DummyAlert(id=1, transaction_id=0, risk_score=kwargs.get("risk_score", 0.0)))

    res = alert_service.generate_alerts_from_batch(None, [{"transaction_id": "1", "amount": 1.0}])
    assert res == []  # con umbral alto no se generan alertas

    # now lower threshold and expect creation
    monkeypatch.setattr(alert_service.settings_service, "get_active_threshold", lambda db: 0.5)
    res2 = alert_service.generate_alerts_from_batch(None, [{"transaction_id": "1", "amount": 1.0}])
    assert len(res2) == 1  # bajando el umbral, ahora debe generarse una alerta


def test_alert_uses_default_threshold_when_config_missing_if_supported(monkeypatch, monkeypatching=None):
    # simulate settings_service returning default value when no config
    monkeypatch.setattr(alert_service, "_get_active_model_row", lambda db: DummyModelRow())
    monkeypatch.setattr(alert_service, "load_model_by_info", lambda name, ver, models_dir=None: object())
    monkeypatch.setattr(alert_service, "risk_score_from_model", lambda model, name, X: [0.7])
    # simulate default threshold 0.7
    monkeypatch.setattr(alert_service.settings_service, "get_active_threshold", lambda db: 0.7)

    called = {}

    def fake_create_alert(db, transaction_id, risk_score, risk_level, model_name, status):
        called['ok'] = True
        return DummyAlert(id=5, transaction_id=transaction_id, risk_score=risk_score, risk_level=risk_level)

    monkeypatch.setattr(alert_service.alert_repository, "create_alert", fake_create_alert)
    res = alert_service.generate_alerts_from_batch(None, [{"transaction_id": "x", "amount": 2.0}])
    # since score equals threshold, alert should be created
    assert called.get('ok', False) is True  # igualdad con el umbral debe crear alerta según la política


def test_alert_payload_contains_required_fields_and_status_default_is_new(monkeypatch):
    monkeypatch.setattr(alert_service, "_get_active_model_row", lambda db: DummyModelRow(model_name="mn"))
    monkeypatch.setattr(alert_service, "load_model_by_info", lambda name, ver, models_dir=None: object())
    monkeypatch.setattr(alert_service, "risk_score_from_model", lambda model, name, X: [0.95])
    monkeypatch.setattr(alert_service.settings_service, "get_active_threshold", lambda db: 0.5)

    recorded = {}

    def fake_create_alert(db, transaction_id, risk_score, risk_level, model_name, status):
        recorded.update({"transaction_id": transaction_id, "risk_score": risk_score, "risk_level": risk_level, "model_name": model_name, "status": status})
        return DummyAlert(id=77, transaction_id=transaction_id, risk_score=risk_score, risk_level=risk_level)

    monkeypatch.setattr(alert_service.alert_repository, "create_alert", fake_create_alert)
    out = alert_service.generate_alerts_from_batch(None, [{"transaction_id": "42", "amount": 10.0}])
    assert "transaction_id" in recorded  # el payload debe contener transaction_id
    assert "risk_score" in recorded  # el payload debe contener la puntuación de riesgo
    assert "risk_level" in recorded  # el payload debe contener el nivel de riesgo
    assert "model_name" in recorded  # el payload debe indicar el modelo usado
    assert recorded.get("status") == "NEW"  # el estado por defecto debe ser NEW

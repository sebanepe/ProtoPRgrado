"""
Pruebas unitarias para clasificación de riesgo (umbrales).
Usan la función `classify_risk_level` que normaliza valores y aplica umbrales 0.5/0.8.
"""
from backend.app.ml.scoring import classify_risk_level


def test_score_below_050_is_low():
    assert classify_risk_level(0.49) == "LOW"  # por debajo de 0.5 -> LOW


def test_score_equal_050_is_medium():
    assert classify_risk_level(0.50) == "MEDIUM"  # exactamente 0.5 -> MEDIUM


def test_score_between_050_and_080_is_medium():
    assert classify_risk_level(0.70) == "MEDIUM"  # entre 0.5 y 0.8 -> MEDIUM


def test_score_equal_080_is_high():
    assert classify_risk_level(0.80) == "HIGH"  # 0.8 -> HIGH


def test_score_above_080_is_high():
    assert classify_risk_level(0.95) == "HIGH"  # >0.8 -> HIGH


def test_negative_score_is_handled_safely():
    # negative should be clamped to 0.0 -> LOW
    assert classify_risk_level(-0.2) == "LOW"  # negativo se clampa a 0 -> LOW


def test_score_above_one_is_handled_safely():
    # >1 clamped to 1.0 -> HIGH
    assert classify_risk_level(1.5) == "HIGH"  # >1 se clampa a 1 -> HIGH


def test_bulk_score_classification():
    scores = [0.0, 0.5, 0.75, 0.85]
    expected = ["LOW", "MEDIUM", "MEDIUM", "HIGH"]
    result = [classify_risk_level(s) for s in scores]
    assert result == expected  # clasificación por lotes coincide con esperado

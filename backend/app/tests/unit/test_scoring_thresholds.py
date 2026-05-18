"""
Pruebas unitarias para clasificación de riesgo (umbrales).
Usan la función `classify_risk_level` que normaliza valores y aplica umbrales 0.5/0.8.
"""
from backend.app.ml.scoring import classify_risk_level


def test_score_below_050_is_low():
    assert classify_risk_level(0.49) == "LOW"


def test_score_equal_050_is_medium():
    assert classify_risk_level(0.50) == "MEDIUM"


def test_score_between_050_and_080_is_medium():
    assert classify_risk_level(0.70) == "MEDIUM"


def test_score_equal_080_is_high():
    assert classify_risk_level(0.80) == "HIGH"


def test_score_above_080_is_high():
    assert classify_risk_level(0.95) == "HIGH"


def test_negative_score_is_handled_safely():
    # negative should be clamped to 0.0 -> LOW
    assert classify_risk_level(-0.2) == "LOW"


def test_score_above_one_is_handled_safely():
    # >1 clamped to 1.0 -> HIGH
    assert classify_risk_level(1.5) == "HIGH"


def test_bulk_score_classification():
    scores = [0.0, 0.5, 0.75, 0.85]
    expected = ["LOW", "MEDIUM", "MEDIUM", "HIGH"]
    result = [classify_risk_level(s) for s in scores]
    assert result == expected

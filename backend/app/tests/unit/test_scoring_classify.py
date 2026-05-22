"""Unit test for `classify_risk` moved to unit folder."""

def test_classify_risk_levels_moved():
    from backend.app.ml.scoring import classify_risk
    assert classify_risk(0.8) == "HIGH"  # puntuación alta -> nivel HIGH
    assert classify_risk(0.5) == "MEDIUM"  # umbral medio -> MEDIUM
    assert classify_risk(0.1) == "LOW"  # puntuación baja -> LOW

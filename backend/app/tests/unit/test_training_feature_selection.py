import os
import pandas as pd
from backend.app.ml.build_training_dataset import build_training_dataset, PROJECT_PROCESSED_DIR


# Test que valida la generación del dataset de entrenamiento y que elimina columnas sensibles
def test_build_training_dataset_removes_sensitive_columns(tmp_path):
    df = pd.DataFrame([
        {'transaction_id':'t1','amount':100,'transaction_datetime':'2026-05-20 10:00','response_code':'59','customer_hash':'c1'},
        {'transaction_id':'t2','amount':200,'transaction_datetime':'2026-05-20 11:00','response_code':'00','customer_hash':'c2'},
    ])
    src = tmp_path / "sample.csv"
    df.to_csv(src, index=False)
    out_file, report = build_training_dataset(str(src), out_name=f"test_training_{os.getpid()}.csv")
    # El archivo de salida debe existir
    assert os.path.exists(out_file)  # verifica que se escribió CSV de entrenamiento
    train = pd.read_csv(out_file)
    # Las columnas sensibles no deben aparecer en las features de entrenamiento
    for forbidden in ["response_code", "normalized_response_code", "response_high_risk", "is_fraud", "is_fraud_proxy", "behavioral_risk_score", "independent_rule_groups"]:
        assert forbidden not in train.columns  # evita fugas de información
    # Confirmación adicional: 'is_fraud' no debe estar entre las columnas
    assert 'is_fraud' not in train.columns  # etiqueta removida de features
    # El reporte debe existir
    assert os.path.exists(report)  # comprueba que se generó el informe

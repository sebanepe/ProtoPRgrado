import pandas as pd
from backend.app.ml.preprocessing import preprocess_dataframe

"""Tests unitarios para preprocesamiento independiente de la base de datos.

Comprueba manejo de nulos, encoding y escalado.
"""


def test_preprocess_handles_missing_and_encoding():
    df = pd.DataFrame([
        {'transaction_id':'a','amount': 10, 'transaction_type': 't', 'channel': 'c', 'location':'l', 'transaction_datetime':'2021-01-01', 'is_fraud':0},
        {'transaction_id':'b','amount': None, 'transaction_type': None, 'channel': None, 'location':None, 'transaction_datetime':'2021-01-02', 'is_fraud':1},
    ])
    processed, summary = preprocess_dataframe(df, apply_smote=False)
    assert 'amount_scaled' in processed.columns  # verifica creación de columna escalada
    assert 'is_fraud' in processed.columns  # verifica que la etiqueta de fraude esté presente
    assert summary['after_clean'] == 2  # ambas filas deben permanecer después de la limpieza

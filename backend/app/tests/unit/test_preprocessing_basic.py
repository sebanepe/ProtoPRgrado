import pandas as pd
from backend.app.ml.preprocessing import preprocess_dataframe

"""Pruebas unitarias básicas de preprocesamiento.

Verifican manejo de valores faltantes, codificación y escalado simple.
"""


def test_preprocess_basic_missing_and_scaling_extracted():
    df = pd.DataFrame([
        {
            "transaction_id": "t1",
            "amount": "100",
            "transaction_type": "purchase",
            "channel": None,
            "location": "A",
            "transaction_datetime": "2021-01-01",
            "is_fraud": 0,
        },
        {
            "transaction_id": "t2",
            "amount": None,
            "transaction_type": None,
            "channel": "web",
            "location": "B",
            "transaction_datetime": "2021-01-02",
            "is_fraud": 1,
        },
    ])

    processed, summary = preprocess_dataframe(df, apply_smote=False)

    assert summary["after_clean"] == 2  # confirma que ambas filas sobrevivieron al limpiado
    assert "amount_scaled" in processed.columns  # verifica que se creó columna normalizada
    assert "is_fraud" in processed.columns  # la etiqueta de fraude debe preservarse para auditoría
    assert len(summary["columns_transformed"]) > 0  # se transformaron columnas

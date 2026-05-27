import pandas as pd
import pytest
from backend.app.ml.proxy_labeling import normalize_response_code, generate_proxy_fraud_label, calculate_behavioral_risk_score, calculate_independent_rule_groups

# Pruebas unitarias para la lógica de proxy labeling
# - Cada test valida un aspecto de las reglas débiles/firmes y del cálculo de features.
# - Después de cada assert se añade un comentario en español explicando qué verifica.


def test_normalize_response_code():
    # Normaliza espacios y tipos numéricos a código de dos dígitos
    assert normalize_response_code(' 07 ') == '07'  # elimina espacios y formatea
    assert normalize_response_code(7.0) == '07'  # convierte float->str con padding
    assert normalize_response_code(41) == '41'  # entero convertido a string
    assert normalize_response_code(None) == ''  # None produce cadena vacía


def test_response_code_high_risk_labels():
    df = pd.DataFrame({'response_code': ['59', '43', '00']})
    df = generate_proxy_fraud_label(df)
    # Códigos de respuesta de alto riesgo NO deben marcar como fraude (behavioral-only)
    assert df.iloc[0]['is_fraud'] == 0  # '59' => no fuerza etiqueta
    assert df.iloc[1]['is_fraud'] == 0  # '43' => no fuerza etiqueta
    assert df.iloc[2]['is_fraud'] == 0  # '00' => no debe marcar fraude


def test_behavioral_weak_label_combination():
    # Crea transacciones que deberían activar múltiples grupos de reglas
    # (alto monto 1h, transacción nocturna, internacional, múltiples clientes en el mismo comercio)
    rows = []
    # create 6 transactions for same customer same day including high amounts to trigger counts
    base_dt = pd.Timestamp('2026-05-20 01:30')
    for i in range(6):
        rows.append({
            'transaction_datetime': base_dt + pd.Timedelta(minutes=i),
            'amount': 4000 if i==0 else 2000,
            'customer_hash': 'cust_1',
            'country_code': 'US',
            'merchant_hash': 'm1',
            'card_brand': 'VISA',
            'card_product_proxy': 'P1',
            'pos_entry_mode': 10, # TNP
            'has_pinblock': 0,
        })
    # add several other customers for same merchant within same hour to trigger merchant-distinct-customer features
    for j in range(2, 7):
        rows.append({
            'transaction_datetime': base_dt + pd.Timedelta(minutes=6 + j),
            'amount': 50,
            'customer_hash': f'cust_{j}',
            'country_code': 'US',
            'merchant_hash': 'm1',
            'card_brand': 'VISA',
            'card_product_proxy': 'P1',
            'pos_entry_mode': 10,
            'has_pinblock': 0,
        })
    df = pd.DataFrame(rows)
    df = generate_proxy_fraud_label(df)
    # behavioral_risk_score debe estar normalizado entre 0 y 1
    assert df['behavioral_risk_score'].between(0,1).all()  # verifica rango válido
    # El conteo de grupos independientes debe ser >=3 en la primera fila
    assert df.iloc[0]['independent_rule_groups'] >= 3  # comprueba activación de múltiples grupos
    # El registro debe ser marcado como fraude y la razón debe ser texto no vacío
    assert df.iloc[0]['is_fraud'] == 1  # la combinación de reglas activa la etiqueta
    assert isinstance(df.iloc[0]['fraud_label_reason'], str) and len(df.iloc[0]['fraud_label_reason']) > 0  # razón presente


def test_independent_rule_groups_counting():
    df = pd.DataFrame([
        {'feature_high_amount':1, 'feature_night_transaction':0, 'feature_international_transaction':0, 'feature_many_customer_transactions_day':0, 'feature_many_merchants_customer_day':0, 'feature_tp_pem_07':0},
        {'feature_high_amount':0, 'feature_night_transaction':1, 'feature_international_transaction':1, 'feature_many_customer_transactions_day':1, 'feature_many_merchants_customer_day':0, 'feature_tp_pem_07':0},
    ])
    counts = calculate_independent_rule_groups(df)
    # Primer caso tiene al menos 1 grupo activo
    assert counts.iloc[0] >= 1  # verifica que la función cuente grupos activos
    # Segundo caso debería tener 3 o más grupos activos
    assert counts.iloc[1] >= 3  # verifica que múltiples flags se cuentan correctamente

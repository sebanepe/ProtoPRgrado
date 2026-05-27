# Preprocessing Report

Source file: C:\Users\seban\AppData\Local\Temp\pytest-of-seban\pytest-58\test_build_training_dataset_re0\sample.csv

## Rows
- original_rows: 2
- cleaned_rows: 2
- feature_set_rows: 2

## Rows removed
- total_removed: 0
- note: detailed removal reasons are available in preprocessing logs if any.

## Distributions
### is_fraud distribution
- 0: 2

### behavioral_risk_score summary
- count: 2.0
- mean: 0.10526315789473684
- std: 0.0
- min: 0.10526315789473684
- 25%: 0.10526315789473684
- 50%: 0.10526315789473684
- 75%: 0.10526315789473684
- max: 0.10526315789473684

### independent_rule_groups distribution
- 2: 2

## Columns removed for sensitivity and leakage
- response_code
- normalized_response_code
- response_high_risk
- response_code_reason
- is_fraud_proxy
- behavioral_risk_score
- independent_rule_groups
- label_source
- fraud_label_reason
- risk_signal_reason
- transaction_id
- customer_hash
- merchant_hash
- device_id
- reference_number
- authorization_code
- merchant_code
- terminal_code
- pan_card
- masked_card
- PAN_TARJETA
- TARJETA

## Optional columns removed to avoid overfitting
- merchant_name
- transaction_datetime

## Warnings and checks
- WARNING: is_fraud contains a single class. Model training will be affected.
- CONFIRMATION: response_code was NOT used to generate is_fraud (label_source contains no response_code_proxy).

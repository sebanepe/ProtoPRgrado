# Preprocessing Report

Source file: C:\Users\seban\AppData\Local\Temp\pytest-of-seban\pytest-65\test_feature_set_generation_re0\sample.csv

## Rows
- original_rows: 2
- cleaned_rows: 2
- feature_set_rows: 2

## Rows removed
- total_removed: 0
- note: detailed removal reasons are available in preprocessing logs if any.

## Distributions
### is_fraud distribution
- 1: 1
- 0: 1

### behavioral_risk_score summary
- count: 2.0
- mean: 0.4
- std: 0.5656854249492381
- min: 0.0
- 25%: 0.2
- 50%: 0.4
- 75%: 0.6000000000000001
- max: 0.8

### independent_rule_groups distribution
- 3: 1
- 0: 1

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
- ALERT: response_code_proxy was used for labeling in this dataset.

from backend.app.ml.preprocessing import preprocess_dataframe, save_processed
import pandas as pd, time, os, json
in_path = r"data/uploads/20260528000338_TRX abril2026.csv"
print('READING', in_path)
df = pd.read_csv(in_path)
cleaned, summary = preprocess_dataframe(df)
ts = int(time.time()*1000)
out_dir = os.path.join(os.getcwd(),'data','processed')
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, f"cleaned_dataset_manual_12_{ts}.csv")
save_processed(cleaned, out_path)
training_path = os.path.join(out_dir, f"training_dataset_manual_12_{ts}.csv")
try:
    cleaned_nosensitive = cleaned.copy()
    sensitive = ["pan_card", "masked_card", "tarjeta", "pan_tarjeta", "numero_cuenta", "documento_identidad"]
    cols_to_drop = [c for c in cleaned_nosensitive.columns if c.lower() in sensitive]
    cleaned_nosensitive = cleaned_nosensitive.drop(columns=cols_to_drop, errors="ignore")
    save_processed(cleaned_nosensitive, training_path)
except Exception as e:
    training_path = None
print('OUT_PATH', out_path)
print('TRAINING_PATH', training_path)
print('SUMMARY', json.dumps(summary)[:1000])

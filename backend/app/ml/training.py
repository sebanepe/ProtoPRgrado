import os
import pandas as pd
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest
from backend.app.ml.evaluation import compute_metrics
from typing import Dict, Tuple


def load_processed(input_path: str) -> pd.DataFrame:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Processed file not found: {input_path}")
    df = pd.read_csv(input_path)
    if "is_fraud" not in df.columns:
        raise ValueError("Processed file missing 'is_fraud' column")
    return df


def prepare_X_y(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    y = df["is_fraud"].astype(int)
    X = df.drop(columns=["is_fraud"]) if "is_fraud" in df.columns else df.copy()
    return X, y


def train_models(X_train, y_train) -> Dict:
    models = {}
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    models["logistic_regression"] = lr

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    models["random_forest"] = rf

    # IsolationForest: unsupervised - set contamination from positive rate if available
    contamination = float(y_train.mean()) if len(y_train) > 0 else 0.01
    contamination = contamination if contamination > 0 else 0.01
    iso = IsolationForest(random_state=42, contamination=contamination)
    iso.fit(X_train)
    models["isolation_forest"] = iso

    return models


def evaluate_model(name: str, model, X_test, y_test) -> Dict:
    # get predictions and scores
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
        y_pred = (y_score >= 0.5).astype(int)
    elif name == "isolation_forest":
        # isolation: predict -> -1 outlier, 1 inlier
        y_pred_raw = model.predict(X_test)
        y_pred = (y_pred_raw == -1).astype(int)
        # decision_function: the lower, the more abnormal -> invert sign so higher = more anomalous
        try:
            y_score = -model.decision_function(X_test)
        except Exception:
            y_score = y_pred
    else:
        y_pred = model.predict(X_test)
        y_score = None

    metrics = compute_metrics(y_test, y_pred, y_score)
    return metrics


def save_model(model, save_dir: str, model_name: str) -> str:
    os.makedirs(save_dir, exist_ok=True)
    version = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    filename = f"{model_name}_{version}.pkl"
    path = os.path.join(save_dir, filename)
    joblib.dump(model, path)
    return path, version

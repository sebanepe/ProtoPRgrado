from __future__ import annotations

import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from backend.app.ml.unsupervised_feature_builder import default_models_dir, default_processed_dir, normalize_run_token


AUTOENCODER_DEPENDENCY_ERROR = "AUTOENCODER_DEPENDENCY_NOT_AVAILABLE"
AUTOENCODER_WARNING = "Autoencoder anomalies are not confirmed fraud."
METHODOLOGY_WARNING_ES = (
    "Las anomalías detectadas por el autoencoder se basan en error de reconstrucción "
    "y no representan fraude confirmado. No se generó is_fraud. No se generó confirmed_fraud."
)

FORBIDDEN_COLUMNS = {
    "is_fraud",
    "confirmed_fraud",
    "target_is_fraud",
    "analyst_label",
    "human_label",
    "review_status",
    "reviewed_by",
    "reviewed_at",
    "rule_code",
    "rule_name",
    "alert_reason",
    "risk_score",
    "behavioral_risk_score",
    "anomaly_flag",
    "anomaly_score",
    "anomaly_percentile",
    "anomaly_model_name",
    "autoencoder_anomaly_flag",
    "autoencoder_anomaly_score",
    "reconstruction_error",
    "pan_tarjeta",
    "tarjeta",
    "pan_card",
    "raw_card",
    "masked_card",
    "authorization_code",
    "reference_number",
    "response_description",
}

FORBIDDEN_OUTPUT_COLUMNS = {
    "is_fraud",
    "confirmed_fraud",
    "target_is_fraud",
    "analyst_label",
    "human_label",
    "review_status",
    "reviewed_by",
    "reviewed_at",
    "rule_code",
    "rule_name",
    "alert_reason",
    "risk_score",
    "behavioral_risk_score",
    "anomaly_flag",
    "anomaly_score",
    "anomaly_percentile",
    "anomaly_model_name",
    "pan_tarjeta",
    "tarjeta",
    "pan_card",
    "raw_card",
    "masked_card",
    "authorization_code",
    "reference_number",
    "response_description",
}

CONTEXT_COLUMNS = {
    "source_run",
    "transaction_id",
    "customer_hash",
    "transaction_datetime",
    "amount",
    "country_code",
    "merchant_rubro_proxy",
    "pos_entry_mode",
    "has_pinblock",
    "card_presence_type",
}

CONTEXT_OUTPUT_COLUMNS = [
    "source_run",
    "transaction_id",
    "customer_hash",
    "transaction_datetime",
    "amount",
    "country_code",
    "merchant_rubro_proxy",
]


class AutoencoderDependencyError(RuntimeError):
    """Raised when PyTorch is required but not installed."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_torch():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except Exception as exc:
        raise AutoencoderDependencyError(AUTOENCODER_DEPENDENCY_ERROR) from exc
    return torch, nn, DataLoader, TensorDataset


def is_torch_available() -> bool:
    try:
        _require_torch()
        return True
    except AutoencoderDependencyError:
        return False


class AutoencoderTabular:
    def __init__(self, input_dim: int, latent_dim: int = 16):
        torch, nn, _, _ = _require_torch()
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.module = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, self.input_dim),
        )
        self._torch = torch

    def __call__(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def train(self, mode: bool = True):
        self.module.train(mode)
        return self

    def eval(self):
        self.module.eval()
        return self

    def parameters(self):
        return self.module.parameters()

    def state_dict(self):
        return self.module.state_dict()

    def to(self, device: str):
        self.module.to(device)
        return self


def _is_identifier_column(column: str) -> bool:
    key = column.lower()
    return key.endswith("_id") or key in {"id"} or "identifier" in key


def _is_forbidden_or_context(column: str) -> bool:
    key = column.lower()
    if key in FORBIDDEN_COLUMNS or key in CONTEXT_COLUMNS:
        return True
    if _is_identifier_column(key):
        return True
    if "review" in key or "fraud" in key or key.startswith("rule_"):
        return True
    return False


def prepare_autoencoder_matrix(df: pd.DataFrame):
    context_df = pd.DataFrame(index=df.index)
    for column in CONTEXT_OUTPUT_COLUMNS:
        if column in df.columns:
            context_df[column] = df[column]

    numeric_columns = [
        column
        for column in df.select_dtypes(include=[np.number, "bool"]).columns
        if not _is_forbidden_or_context(column)
    ]
    if not numeric_columns:
        raise ValueError("No numeric feature columns available for autoencoder training")

    matrix = df[numeric_columns].replace([np.inf, -np.inf], np.nan)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    imputed = imputer.fit_transform(matrix)
    X_scaled = scaler.fit_transform(imputed).astype(np.float32)
    scaler_bundle = {"imputer": imputer, "scaler": scaler}
    return X_scaled, scaler_bundle, numeric_columns, context_df


def compute_reconstruction_errors(model: AutoencoderTabular, X_scaled: np.ndarray, contamination: float, device: Optional[str] = None) -> Dict[str, Any]:
    torch, _, _, _ = _require_torch()
    device_name = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device_name)
    model.eval()
    tensor = torch.tensor(X_scaled, dtype=torch.float32, device=device_name)
    with torch.no_grad():
        reconstructed = model(tensor)
        errors = torch.mean((reconstructed - tensor) ** 2, dim=1).detach().cpu().numpy()

    percentile = 100.0 * (1.0 - float(contamination))
    threshold = float(np.percentile(errors, percentile))
    flags = (errors >= threshold).astype(int)
    if len(flags) and flags.sum() == 0:
        flags[int(np.argmax(errors))] = 1

    max_error = float(np.max(errors)) if len(errors) else 0.0
    min_error = float(np.min(errors)) if len(errors) else 0.0
    denominator = max(max_error - min_error, 1e-12)
    scores = (errors - min_error) / denominator
    ranks = pd.Series(errors).rank(method="first", ascending=False).astype(int).to_numpy()
    return {
        "reconstruction_error": errors.astype(float),
        "autoencoder_anomaly_score": scores.astype(float),
        "autoencoder_anomaly_flag": flags.astype(int),
        "anomaly_rank": ranks.astype(int),
        "threshold": threshold,
    }


def train_autoencoder_model(
    input_path: str,
    source_run: str,
    epochs: int = 30,
    batch_size: int = 512,
    learning_rate: float = 0.001,
    latent_dim: int = 16,
    contamination: float = 0.01,
    sample_size: Optional[int] = None,
    random_state: int = 42,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    if not 0 < float(contamination) < 1:
        raise ValueError("contamination must be between 0 and 1")
    if epochs < 1:
        raise ValueError("epochs must be >= 1")

    torch, nn, DataLoader, TensorDataset = _require_torch()
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)

    feature_frame = pd.read_csv(input_path)
    if feature_frame.empty:
        raise ValueError("No rows available for autoencoder training")
    if sample_size is not None and sample_size > 0 and sample_size < len(feature_frame):
        feature_frame = feature_frame.sample(n=sample_size, random_state=random_state).reset_index(drop=True)

    X_scaled, scaler_bundle, feature_columns, context_df = prepare_autoencoder_matrix(feature_frame)
    device_name = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoencoderTabular(input_dim=X_scaled.shape[1], latent_dim=latent_dim).to(device_name)
    dataset = TensorDataset(torch.tensor(X_scaled, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=max(1, int(batch_size)), shuffle=True)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    losses: list[float] = []

    for _ in range(int(epochs)):
        model.train()
        epoch_losses = []
        for (batch,) in loader:
            batch = batch.to(device_name)
            optimizer.zero_grad()
            reconstructed = model(batch)
            loss = criterion(reconstructed, batch)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu().item()))
        losses.append(float(np.mean(epoch_losses)) if epoch_losses else 0.0)

    score_parts = compute_reconstruction_errors(model, X_scaled, contamination=contamination, device=device_name)
    score_frame = context_df.copy()
    score_frame["source_run"] = str(source_run)
    score_frame["reconstruction_error"] = score_parts["reconstruction_error"]
    score_frame["autoencoder_anomaly_score"] = score_parts["autoencoder_anomaly_score"]
    score_frame["autoencoder_anomaly_flag"] = score_parts["autoencoder_anomaly_flag"]
    score_frame["anomaly_rank"] = score_parts["anomaly_rank"]
    score_frame = score_frame.sort_values(["anomaly_rank"]).reset_index(drop=True)

    return {
        "model": model,
        "scaler": scaler_bundle,
        "score_frame": score_frame,
        "feature_columns": feature_columns,
        "threshold": score_parts["threshold"],
        "loss_history": losses,
        "device": device_name,
    }


def _write_report(report_path: Path, metadata: Dict[str, Any]) -> None:
    lines = [
        "# Autoencoder PyTorch Report",
        "",
        f"- source_run: {metadata['source_run']}",
        f"- total_records: {metadata['total_records']}",
        "- algoritmo: autoencoder_pytorch",
        "- framework: PyTorch",
        f"- epochs: {metadata['epochs']}",
        f"- batch_size: {metadata['batch_size']}",
        f"- latent_dim: {metadata['latent_dim']}",
        f"- learning_rate: {metadata['learning_rate']}",
        f"- contamination: {metadata['contamination']}",
        f"- reconstruction_threshold: {metadata['threshold']}",
        f"- anomaly_count: {metadata['anomaly_count']}",
        f"- anomaly_rate: {metadata['anomaly_rate']}",
        f"- scores_file: {metadata['scores_file']}",
        f"- model_file: {metadata['model_file']}",
        f"- metadata_file: {metadata['metadata_file']}",
        "",
        "## Feature columns",
        "",
        ", ".join(metadata["feature_columns"]),
        "",
        "## Advertencia metodologica",
        "",
        METHODOLOGY_WARNING_ES,
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def save_autoencoder_artifacts(
    *,
    source_run: str,
    trained: Dict[str, Any],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    latent_dim: int,
    contamination: float,
    input_path: str,
    output_dir: str | os.PathLike[str] | None = None,
    models_dir: str | os.PathLike[str] | None = None,
) -> Dict[str, Any]:
    torch, _, _, _ = _require_torch()
    output_directory = Path(output_dir) if output_dir else default_processed_dir()
    models_directory = Path(models_dir) if models_dir else default_models_dir()
    output_directory.mkdir(parents=True, exist_ok=True)
    models_directory.mkdir(parents=True, exist_ok=True)

    run_token = normalize_run_token(source_run)
    score_file = output_directory / f"autoencoder_scores_run_{run_token}.csv"
    report_file = output_directory / f"autoencoder_report_run_{run_token}.md"
    model_file = models_directory / f"autoencoder_model_run_{run_token}.pt"
    scaler_file = models_directory / f"autoencoder_scaler_run_{run_token}.pkl"
    metadata_file = models_directory / f"autoencoder_model_run_{run_token}_metadata.json"

    score_frame = trained["score_frame"].copy()
    for forbidden in FORBIDDEN_OUTPUT_COLUMNS:
        score_frame = score_frame.drop(columns=[forbidden], errors="ignore")
    score_frame.to_csv(score_file, index=False)

    torch.save(
        {
            "state_dict": trained["model"].state_dict(),
            "input_dim": len(trained["feature_columns"]),
            "latent_dim": latent_dim,
            "feature_columns": trained["feature_columns"],
        },
        model_file,
    )
    joblib.dump(trained["scaler"], scaler_file)

    total_records = int(len(score_frame))
    anomaly_count = int(score_frame["autoencoder_anomaly_flag"].sum()) if total_records else 0
    anomaly_rate = float(anomaly_count / total_records) if total_records else 0.0
    metadata = {
        "source_run": str(source_run),
        "run_token": run_token,
        "model_family": "UNSUPERVISED",
        "algorithm": "autoencoder_pytorch",
        "framework": "pytorch",
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "latent_dim": int(latent_dim),
        "learning_rate": float(learning_rate),
        "contamination": float(contamination),
        "threshold": float(trained["threshold"]),
        "total_records": total_records,
        "anomaly_count": anomaly_count,
        "anomaly_rate": anomaly_rate,
        "feature_columns": trained["feature_columns"],
        "model_file": model_file.name,
        "scaler_file": scaler_file.name,
        "scores_file": score_file.name,
        "report_file": report_file.name,
        "metadata_file": metadata_file.name,
        "feature_file": str(input_path),
        "created_at": _utc_now(),
        "warnings": [AUTOENCODER_WARNING, METHODOLOGY_WARNING_ES],
    }
    metadata_file.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_report(report_file, metadata)

    return {
        "status": "COMPLETED",
        "model_family": "UNSUPERVISED",
        "algorithm": "autoencoder_pytorch",
        "source_run": str(source_run),
        "run_token": run_token,
        "total_records": total_records,
        "anomaly_count": anomaly_count,
        "anomaly_rate": anomaly_rate,
        "threshold": float(trained["threshold"]),
        "scores_file": score_file.name,
        "report_file": report_file.name,
        "model_file": model_file.name,
        "scaler_file": scaler_file.name,
        "metadata_file": metadata_file.name,
        "feature_file": str(input_path),
        "paths": {
            "scores_file": str(score_file),
            "report_file": str(report_file),
            "model_file": str(model_file),
            "scaler_file": str(scaler_file),
            "metadata_file": str(metadata_file),
        },
        "metadata": metadata,
        "warnings": metadata["warnings"],
    }

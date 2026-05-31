from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from sqlalchemy.orm import Session

from backend.app.ml.autoencoder_anomaly import AUTOENCODER_DEPENDENCY_ERROR, AutoencoderDependencyError
from backend.app.ml.train_autoencoder_anomaly import train_autoencoder_anomaly
from backend.app.services.artifact_registry_service import normalize_run_token


FORBIDDEN_RESPONSE_COLUMNS = {"is_fraud", "confirmed_fraud", "target_is_fraud", "pan_tarjeta", "tarjeta", "pan_card", "raw_card"}


class AutoencoderAnomalyService:
    def __init__(self, processed_dir: Optional[str] = None, models_dir: Optional[str] = None):
        self.processed_dir = Path(processed_dir or os.path.join("data", "processed"))
        self.models_dir = Path(models_dir or os.path.join("data", "models"))

    def train(
        self,
        *,
        source_run: str,
        epochs: int,
        batch_size: int,
        latent_dim: int,
        learning_rate: float,
        contamination: float,
        sample_size: Optional[int],
        db: Optional[Session] = None,
    ) -> dict[str, Any]:
        try:
            return train_autoencoder_anomaly(
                source_run=source_run,
                epochs=epochs,
                batch_size=batch_size,
                latent_dim=latent_dim,
                learning_rate=learning_rate,
                contamination=contamination,
                sample_size=sample_size,
                output_dir=self.processed_dir,
                models_dir=self.models_dir,
                db=db,
            )
        except AutoencoderDependencyError:
            return {
                "status": AUTOENCODER_DEPENDENCY_ERROR,
                "model_family": "UNSUPERVISED",
                "algorithm": "autoencoder_pytorch",
                "source_run": source_run,
                "warnings": ["PyTorch is required only for Autoencoder training."],
            }

    def get_metrics(self, source_run: str) -> dict[str, Any]:
        metadata = self._load_metadata(source_run)
        if not metadata:
            raise FileNotFoundError(f"Autoencoder metadata not found for source_run: {source_run}")
        return {
            "source_run": metadata.get("source_run", source_run),
            "algorithm": metadata.get("algorithm", "autoencoder_pytorch"),
            "total_records": metadata.get("total_records", 0),
            "anomaly_count": metadata.get("anomaly_count", 0),
            "anomaly_rate": metadata.get("anomaly_rate", 0.0),
            "threshold": metadata.get("threshold"),
            "contamination": metadata.get("contamination"),
            "created_at": metadata.get("created_at"),
            "status": "AVAILABLE",
            "files": {
                "scores_file": metadata.get("scores_file"),
                "report_file": metadata.get("report_file"),
                "model_file": metadata.get("model_file"),
                "metadata_file": metadata.get("metadata_file"),
                "scaler_file": metadata.get("scaler_file"),
            },
            "warnings": metadata.get("warnings", []),
        }

    def get_scores(self, source_run: str, page: int = 1, page_size: int = 50, anomaly_flag: Optional[int] = None) -> dict[str, Any]:
        score_file = self._find_scores_file(source_run)
        if not score_file or not score_file.exists():
            raise FileNotFoundError(f"Autoencoder scores not found for source_run: {source_run}")
        df = pd.read_csv(score_file, dtype={"merchant_rubro_proxy": str})
        if anomaly_flag is not None:
            df = df[df["autoencoder_anomaly_flag"] == anomaly_flag]
        total_items = len(df)
        total_pages = math.ceil(total_items / page_size) if page_size > 0 else 0
        start_idx = (max(page, 1) - 1) * page_size
        page_df = df.iloc[start_idx : start_idx + page_size]
        columns = [column for column in page_df.columns if column.lower() not in FORBIDDEN_RESPONSE_COLUMNS]
        page_df = page_df[columns].where(pd.notnull(page_df[columns]), None)
        return {
            "source_run": source_run,
            "algorithm": "autoencoder_pytorch",
            "page": page,
            "page_size": page_size,
            "total_items": total_items,
            "total_pages": total_pages,
            "items": page_df.to_dict(orient="records"),
        }

    def get_report(self, source_run: str) -> dict[str, str]:
        report_file = self._find_report_file(source_run)
        if not report_file or not report_file.exists():
            raise FileNotFoundError(f"Autoencoder report not found for source_run: {source_run}")
        return {"source_run": source_run, "algorithm": "autoencoder_pytorch", "report": report_file.read_text(encoding="utf-8")}

    def get_model_metadata(self, source_run: str) -> dict[str, Any]:
        metadata = self._load_metadata(source_run)
        if not metadata:
            raise FileNotFoundError(f"Autoencoder metadata not found for source_run: {source_run}")
        return {"source_run": source_run, "algorithm": "autoencoder_pytorch", "metadata": metadata}

    def _token(self, source_run: str) -> str:
        return normalize_run_token(source_run)

    def _find_scores_file(self, source_run: str) -> Optional[Path]:
        token = self._token(source_run)
        path = self.processed_dir / f"autoencoder_scores_run_{token}.csv"
        return path if path.exists() else None

    def _find_report_file(self, source_run: str) -> Optional[Path]:
        token = self._token(source_run)
        path = self.processed_dir / f"autoencoder_report_run_{token}.md"
        return path if path.exists() else None

    def _find_metadata_file(self, source_run: str) -> Optional[Path]:
        token = self._token(source_run)
        path = self.models_dir / f"autoencoder_model_run_{token}_metadata.json"
        return path if path.exists() else None

    def _load_metadata(self, source_run: str) -> dict[str, Any]:
        metadata_file = self._find_metadata_file(source_run)
        if not metadata_file:
            return {}
        try:
            payload = json.loads(metadata_file.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}

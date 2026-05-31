from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Optional

from sqlalchemy.orm import Session

from backend.app.database import SessionLocal
from backend.app.ml.autoencoder_anomaly import (
    AUTOENCODER_DEPENDENCY_ERROR,
    AutoencoderDependencyError,
    _require_torch,
    save_autoencoder_artifacts,
    train_autoencoder_model,
)
from backend.app.ml.unsupervised_feature_builder import build_unsupervised_features, default_models_dir, default_processed_dir
from backend.app.services import artifact_registry_service as artifacts
from backend.app.services import model_registry_service


def _resolve_feature_file_from_registry(db: Optional[Session], source_run: str) -> Optional[Path]:
    if db is None:
        return None
    try:
        artifact = artifacts.get_artifact_by_type(db, source_run, artifacts.ARTIFACT_UNSUPERVISED_FEATURE_SET)
    except Exception:
        return None
    if artifact and artifact.status == "AVAILABLE" and artifact.file_path:
        path = Path(artifact.file_path)
        if path.exists():
            return path
    return None


def _resolve_input_file(
    *,
    source_run: str,
    input_path: Optional[str],
    processed_dir: Path,
    db: Optional[Session],
    warnings: list[str],
) -> Path:
    if input_path:
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(f"Input feature file not found: {path}")
        return path

    registry_path = _resolve_feature_file_from_registry(db, source_run)
    if registry_path:
        return registry_path

    run_token = artifacts.normalize_run_token(source_run)
    feature_path = processed_dir / f"unsupervised_feature_set_run_{run_token}.csv"
    if feature_path.exists():
        warnings.append("Feature set resolved by default path fallback.")
        return feature_path

    preprocessed_path = processed_dir / f"preprocessed_run_{run_token}.csv"
    if not preprocessed_path.exists():
        raise FileNotFoundError(f"Unsupervised feature set and preprocessed input not found for {source_run}")

    warnings.append("Feature set was missing; it was built from the preprocessed file.")
    built_path, _ = build_unsupervised_features(str(preprocessed_path), source_run, processed_dir)
    return Path(built_path)


def _register_outputs(db: Session, source_run: str, result: dict[str, Any]) -> dict[str, Any]:
    run_token = artifacts.normalize_run_token(source_run)
    paths = result["paths"]
    common = {"algorithm": "autoencoder_pytorch", "model_family": "UNSUPERVISED"}
    registered = {}
    specs = [
        ("scores", artifacts.ARTIFACT_ANOMALY_SCORES_CSV, artifacts.PHASE_C3, paths["scores_file"], {"artifact_role": "autoencoder_scores", **common}),
        ("report", artifacts.ARTIFACT_ANOMALY_REPORT, artifacts.PHASE_C3, paths["report_file"], {"artifact_role": "autoencoder_report", **common}),
        ("model", artifacts.ARTIFACT_MODEL_ARTIFACT, artifacts.PHASE_C3, paths["model_file"], {"framework": "pytorch", **common}),
        ("metadata", artifacts.ARTIFACT_MODEL_METADATA, artifacts.PHASE_C3, paths["metadata_file"], common),
        ("scaler", artifacts.ARTIFACT_MODEL_SCALER, artifacts.PHASE_C3, paths["scaler_file"], {"artifact_role": "scaler", **common}),
    ]
    for key, artifact_type, phase, file_path, metadata in specs:
        artifact = artifacts.register_or_update_artifact(
            db,
            artifact_type=artifact_type,
            phase=phase,
            source_run=source_run,
            run_token=run_token,
            file_path=file_path,
            metadata=metadata,
        )
        registered[key] = artifacts.artifact_to_dict(artifact)

    metrics = {
        "total_records": result["total_records"],
        "anomaly_count": result["anomaly_count"],
        "anomaly_rate": result["anomaly_rate"],
        "reconstruction_threshold": result["threshold"],
        "contamination": result["metadata"]["contamination"],
        "epochs": result["metadata"]["epochs"],
        "batch_size": result["metadata"]["batch_size"],
        "latent_dim": result["metadata"]["latent_dim"],
        "learning_rate": result["metadata"]["learning_rate"],
    }
    model = model_registry_service.register_autoencoder_model(
        db,
        source_run=source_run,
        model_file=paths["model_file"],
        metadata_file=paths["metadata_file"],
        scores_file=paths["scores_file"],
        feature_file=result["feature_file"],
        report_file=paths["report_file"],
        metrics=metrics,
    )
    registered["model_registry"] = model_registry_service.model_registry_to_dict(model)
    return registered


def train_autoencoder_anomaly(
    *,
    source_run: str,
    input_path: Optional[str] = None,
    epochs: int = 30,
    batch_size: int = 512,
    latent_dim: int = 16,
    learning_rate: float = 0.001,
    contamination: float = 0.01,
    sample_size: Optional[int] = None,
    device: Optional[str] = None,
    output_dir: str | os.PathLike[str] | None = None,
    models_dir: str | os.PathLike[str] | None = None,
    db: Optional[Session] = None,
    register_db: bool = True,
) -> dict[str, Any]:
    warnings: list[str] = []
    processed_dir = Path(output_dir) if output_dir else default_processed_dir()
    models_path = Path(models_dir) if models_dir else default_models_dir()
    processed_dir.mkdir(parents=True, exist_ok=True)
    models_path.mkdir(parents=True, exist_ok=True)
    _require_torch()
    close_db = False
    if db is None and register_db:
        try:
            db = SessionLocal()
            close_db = True
        except Exception as exc:
            warnings.append(f"DB registration unavailable: {exc}")
            db = None

    try:
        feature_path = _resolve_input_file(
            source_run=source_run,
            input_path=input_path,
            processed_dir=processed_dir,
            db=db,
            warnings=warnings,
        )
        trained = train_autoencoder_model(
            input_path=str(feature_path),
            source_run=source_run,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            latent_dim=latent_dim,
            contamination=contamination,
            sample_size=sample_size,
            device=device,
        )
        result = save_autoencoder_artifacts(
            source_run=source_run,
            trained=trained,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            latent_dim=latent_dim,
            contamination=contamination,
            input_path=str(feature_path),
            output_dir=processed_dir,
            models_dir=models_path,
        )
        result["warnings"] = list(dict.fromkeys(result.get("warnings", []) + warnings))
        if db is not None and register_db:
            try:
                result["registry"] = _register_outputs(db, source_run, result)
            except Exception as exc:
                result.setdefault("warnings", []).append(f"DB registration warning: {exc}")

        try:
            from backend.app.ml.validate_autoencoder_outputs import validate_autoencoder_outputs

            result["validation"] = validate_autoencoder_outputs(
                score_file=result["paths"]["scores_file"],
                metadata_file=result["paths"]["metadata_file"],
            )
        except Exception as exc:
            result.setdefault("warnings", []).append(f"Validator warning: {exc}")
        return result
    finally:
        if close_db and db is not None:
            db.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PyTorch tabular autoencoder anomaly model")
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--input", default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--contamination", type=float, default=0.01)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--models-dir", default=None)
    args = parser.parse_args()

    try:
        result = train_autoencoder_anomaly(
            source_run=args.source_run,
            input_path=args.input,
            epochs=args.epochs,
            batch_size=args.batch_size,
            latent_dim=args.latent_dim,
            learning_rate=args.learning_rate,
            contamination=args.contamination,
            sample_size=args.sample_size,
            device=args.device,
            output_dir=args.output_dir,
            models_dir=args.models_dir,
        )
    except AutoencoderDependencyError:
        result = {"status": AUTOENCODER_DEPENDENCY_ERROR, "algorithm": "autoencoder_pytorch", "warnings": ["PyTorch is required only for Autoencoder training."]}
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()

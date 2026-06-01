"""
Service for managing unsupervised anomaly detection operations.
Handles reading anomaly scores, metrics, reports, and model metadata.
"""
import os
import json
import re
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import subprocess
import math


class AnomalyService:
    """Handles unsupervised anomaly detection data operations."""

    def __init__(
        self,
        processed_dir: Optional[str] = None,
        models_dir: Optional[str] = None,
    ):
        self.processed_dir = Path(processed_dir or os.path.join("data", "processed"))
        self.models_dir = Path(models_dir or os.path.join("data", "models"))
        self._csv_cache: Dict[str, Tuple[Tuple[int, int], pd.DataFrame]] = {}

    def _read_csv_cached(self, path: Path) -> pd.DataFrame:
        signature = (path.stat().st_mtime_ns, path.stat().st_size)
        cache_key = str(path.resolve())
        cached = self._csv_cache.get(cache_key)
        if cached and cached[0] == signature:
            return cached[1].copy(deep=False)

        df = pd.read_csv(path, dtype={"merchant_rubro_proxy": str}, low_memory=False)
        if len(self._csv_cache) >= 8:
            self._csv_cache.pop(next(iter(self._csv_cache)))
        self._csv_cache[cache_key] = (signature, df)
        return df.copy(deep=False)

    def list_anomaly_runs(self) -> List[Dict[str, Any]]:
        """
        Detect anomaly runs by looking for score files, report files, and model metadata.
        Returns list of runs with metadata.
        """
        runs = {}

        # Pattern: anomaly_scores_run_N.csv or anomaly_scores_preprocessed_run_N.csv
        for score_file in self.processed_dir.glob("anomaly_scores_*.csv"):
            run_id = self._extract_run_id_from_filename(score_file.name)
            if run_id not in runs:
                runs[run_id] = {
                    "anomaly_run_id": run_id,
                    "source_run": None,
                    "score_file": score_file.name,
                    "report_file": None,
                    "model_metadata_file": None,
                    "model_name": None,
                    "algorithm": None,
                    "anomaly_count": 0,
                    "anomaly_rate": 0.0,
                    "created_at": None,
                }

        # Pattern: anomaly_report_run_N.md or anomaly_report_preprocessed_run_N.md
        for report_file in self.processed_dir.glob("anomaly_report_*.md"):
            run_id = self._extract_run_id_from_filename(report_file.name)
            if run_id in runs:
                runs[run_id]["report_file"] = report_file.name

        # Pattern: isolation_forest_run_N_metadata.json
        for metadata_file in self.models_dir.glob("isolation_forest_*_metadata.json"):
            run_id = self._extract_run_id_from_filename(metadata_file.name)
            if run_id in runs:
                runs[run_id]["model_metadata_file"] = metadata_file.name

        # Populate metadata from actual files
        for run_id, run_info in runs.items():
            if run_info["score_file"]:
                try:
                    score_path = self.processed_dir / run_info["score_file"]
                    df_all = self._read_csv_cached(score_path)
                    anomaly_count = (df_all["anomaly_flag"] == 1).sum()
                    total_count = len(df_all)
                    anomaly_rate = anomaly_count / total_count if total_count > 0 else 0.0

                    run_info["anomaly_count"] = int(anomaly_count)
                    run_info["anomaly_rate"] = float(anomaly_rate)
                    run_info["created_at"] = score_path.stat().st_mtime
                except Exception:
                    pass

            if run_info["model_metadata_file"]:
                try:
                    metadata_path = self.models_dir / run_info["model_metadata_file"]
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    run_info["model_name"] = metadata.get("model_type", "isolation_forest")
                    run_info["algorithm"] = metadata.get("algorithm", "IsolationForest")
                    run_info["source_run"] = metadata.get("source_run")
                except Exception:
                    pass

        return sorted(runs.values(), key=lambda x: x.get("created_at") or 0, reverse=True)

    def get_anomaly_scores(
        self,
        run_id: str,
        page: int = 1,
        page_size: int = 50,
        anomaly_flag: Optional[int] = None,
        country_code: Optional[str] = None,
        pos_entry_mode: Optional[str] = None,
        merchant_rubro_proxy: Optional[str] = None,
        customer_hash: Optional[str] = None,
        min_score: Optional[float] = None,
        max_score: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Read anomaly scores from CSV with pagination and optional filters.
        """
        score_file = self._find_anomaly_scores_file(run_id)
        if not score_file or not score_file.exists():
            raise FileNotFoundError(f"Anomaly scores file not found for run: {run_id}")

        df = self._read_csv_cached(score_file)

        # Apply filters
        if anomaly_flag is not None:
            df = df[df["anomaly_flag"] == anomaly_flag]

        if country_code:
            df = df[df["country_code"] == country_code]

        if pos_entry_mode:
            df = df[df["pos_entry_mode"] == pos_entry_mode]

        if merchant_rubro_proxy:
            df = df[df["merchant_rubro_proxy"] == merchant_rubro_proxy]

        if customer_hash:
            df = df[df["customer_hash"] == customer_hash]

        if min_score is not None:
            df = df[df["anomaly_score"] >= min_score]

        if max_score is not None:
            df = df[df["anomaly_score"] <= max_score]

        total_items = len(df)
        total_pages = math.ceil(total_items / page_size) if page_size > 0 else 0

        # Validate page
        if page < 1:
            page = 1
        if page > total_pages and total_pages > 0:
            # Return empty items but correct metadata
            return {
                "run_id": run_id,
                "page": page,
                "page_size": page_size,
                "total_items": total_items,
                "total_pages": total_pages,
                "items": [],
            }

        # Paginate
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        page_df = df.iloc[start_idx:end_idx]

        # Exclude forbidden columns
        forbidden_cols = {"is_fraud", "confirmed_fraud", "target_is_fraud"}
        cols_to_return = [col for col in page_df.columns if col not in forbidden_cols]
        page_df = page_df[cols_to_return]

        # Sanitize NaN/NaT to None
        page_df = page_df.where(pd.notnull(page_df), None)

        items = page_df.to_dict(orient="records")

        return {
            "run_id": run_id,
            "page": page,
            "page_size": page_size,
            "total_items": total_items,
            "total_pages": total_pages,
            "items": items,
        }

    def get_top_anomalies(
        self,
        run_id: str,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """
        Get top anomalies ordered by anomaly_rank (ascending).
        """
        score_file = self._find_anomaly_scores_file(run_id)
        if not score_file or not score_file.exists():
            raise FileNotFoundError(f"Anomaly scores file not found for run: {run_id}")

        df = self._read_csv_cached(score_file)

        # Filter only anomalies
        df_anomalies = df[df["anomaly_flag"] == 1]

        # Sort by anomaly_rank (ascending)
        if "anomaly_rank" in df_anomalies.columns:
            df_anomalies = df_anomalies.sort_values("anomaly_rank", ascending=True)

        # Limit
        df_top = df_anomalies.head(limit)

        # Exclude forbidden columns
        forbidden_cols = {"is_fraud", "confirmed_fraud", "target_is_fraud"}
        cols_to_return = [col for col in df_top.columns if col not in forbidden_cols]
        df_top = df_top[cols_to_return]

        # Sanitize
        df_top = df_top.where(pd.notnull(df_top), None)

        items = df_top.to_dict(orient="records")

        return {
            "run_id": run_id,
            "limit": limit,
            "count": len(items),
            "items": items,
        }

    def get_anomaly_metrics(self, run_id: str) -> Dict[str, Any]:
        """
        Calculate anomaly metrics including distribution by country, pos_entry_mode, mcc, hour.
        """
        score_file = self._find_anomaly_scores_file(run_id)
        if not score_file or not score_file.exists():
            raise FileNotFoundError(f"Anomaly scores file not found for run: {run_id}")

        df = self._read_csv_cached(score_file)

        total_transactions = len(df)
        anomaly_count = (df["anomaly_flag"] == 1).sum()
        anomaly_rate = anomaly_count / total_transactions if total_transactions > 0 else 0.0

        # Get model info
        metadata = self._load_model_metadata(run_id)
        model_name = metadata.get("model_type", "isolation_forest") if metadata else "isolation_forest"
        algorithm = metadata.get("algorithm", "IsolationForest") if metadata else "IsolationForest"
        contamination = metadata.get("contamination", 0.01) if metadata else 0.01

        # Distributions
        anomalies_by_country = (
            df[df["anomaly_flag"] == 1]["country_code"]
            .value_counts()
            .to_dict()
        )
        anomalies_by_pos_entry_mode = (
            df[df["anomaly_flag"] == 1]["pos_entry_mode"]
            .value_counts()
            .to_dict()
        )
        anomalies_by_mcc = (
            df[df["anomaly_flag"] == 1]["merchant_rubro_proxy"]
            .value_counts()
            .head(20)
            .to_dict()
        )

        # Anomalies by hour (if transaction_datetime exists)
        anomalies_by_hour = {}
        if "transaction_datetime" in df.columns:
            try:
                df["_hour"] = pd.to_datetime(df["transaction_datetime"]).dt.hour
                anomalies_by_hour = (
                    df[df["anomaly_flag"] == 1]["_hour"]
                    .value_counts()
                    .sort_index()
                    .to_dict()
                )
            except Exception:
                pass

        # Top customers by anomaly count
        top_customers = (
            df[df["anomaly_flag"] == 1]["customer_hash"]
            .value_counts()
            .head(10)
            .to_dict()
        )

        return {
            "run_id": run_id,
            "total_transactions": int(total_transactions),
            "anomaly_count": int(anomaly_count),
            "anomaly_rate": float(anomaly_rate),
            "model_name": model_name,
            "algorithm": algorithm,
            "contamination": float(contamination),
            "anomalies_by_country": anomalies_by_country,
            "anomalies_by_pos_entry_mode": anomalies_by_pos_entry_mode,
            "anomalies_by_mcc": anomalies_by_mcc,
            "anomalies_by_hour": anomalies_by_hour,
            "top_customers_by_anomaly_count": list(
                [{"customer_hash": k, "count": v} for k, v in top_customers.items()]
            ),
        }

    def get_anomaly_report(self, run_id: str) -> Dict[str, str]:
        """
        Read anomaly report markdown.
        """
        report_file = self._find_anomaly_report_file(run_id)
        if not report_file or not report_file.exists():
            raise FileNotFoundError(f"Anomaly report file not found for run: {run_id}")

        report_content = report_file.read_text(encoding="utf-8")
        metadata = self._load_model_metadata(run_id)
        if metadata:
            repaired_report_content = self._repair_anomaly_report_content(report_content, metadata)
            if repaired_report_content != report_content:
                report_file.write_text(repaired_report_content, encoding="utf-8")
                report_content = repaired_report_content

        return {
            "run_id": run_id,
            "report": report_content,
        }

    def get_model_metadata(self, run_id: str) -> Dict[str, Any]:
        """
        Read model metadata JSON.
        """
        metadata = self._load_model_metadata(run_id)
        if not metadata:
            raise FileNotFoundError(f"Model metadata file not found for run: {run_id}")

        return {
            "run_id": run_id,
            "metadata": metadata,
        }

    def train_anomaly_model(
        self,
        source_run: str,
        model: str = "isolation_forest",
        contamination: float = 0.01,
        sample_size: Optional[int] = None,
        max_categories: int = 50,
        n_estimators: int = 200,
    ) -> Dict[str, Any]:
        """
        Execute unsupervised anomaly model training.
        Runs the CLI command synchronously.
        """
        # Find the preprocessed input file
        input_patterns = [
            self.processed_dir / f"{source_run}.csv",
            self.processed_dir / f"preprocessed_{source_run}.csv",
        ]

        input_path = None
        for pattern in input_patterns:
            if pattern.exists():
                input_path = pattern
                break

        if not input_path:
            raise FileNotFoundError(f"Preprocessed input file not found for run: {source_run}")

        # Build command
        cmd = [
            "python",
            "-m",
            "backend.app.ml.train_unsupervised_anomaly",
            "--input",
            str(input_path),
            "--source-run",
            source_run,
            "--model",
            model,
            "--contamination",
            str(contamination),
        ]

        if sample_size is not None:
            cmd.extend(["--sample-size", str(sample_size)])

        cmd.extend(["--max-categories", str(max_categories)])
        cmd.extend(["--n-estimators", str(n_estimators)])

        # Execute
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"Training failed: {result.stderr}"
                )

            # Read generated artifacts to populate response
            run_id = self._extract_run_id_from_source(source_run)
            score_file = self._find_anomaly_scores_file(run_id)
            report_file = self._find_anomaly_report_file(run_id)
            metadata_file = self._find_model_metadata_file(run_id)

            anomaly_count = 0
            anomaly_rate = 0.0

            if score_file and score_file.exists():
                try:
                    df = self._read_csv_cached(score_file)
                    anomaly_count = int((df["anomaly_flag"] == 1).sum())
                    anomaly_rate = anomaly_count / len(df) if len(df) > 0 else 0.0
                except Exception:
                    pass

            return {
                "status": "COMPLETED",
                "source_run": source_run,
                "anomaly_run_id": run_id,
                "score_file": score_file.name if score_file else None,
                "report_file": report_file.name if report_file else None,
                "metadata_file": metadata_file.name if metadata_file else None,
                "anomaly_count": anomaly_count,
                "anomaly_rate": float(anomaly_rate),
            }

        except subprocess.TimeoutExpired:
            raise RuntimeError("Training timed out after 1 hour")
        except Exception as e:
            raise RuntimeError(f"Training error: {str(e)}")

    # Private helper methods

    def _extract_run_id_from_filename(self, filename: str) -> str:
        """
        Extract run_id from filenames like:
        - anomaly_scores_run_26.csv -> run_26
        - anomaly_scores_preprocessed_run_26.csv -> run_26
        - isolation_forest_run_26_metadata.json -> run_26
        """
        # Try pattern: _run_N
        match = re.search(r"_run_(\d+)", filename)
        if match:
            return f"run_{match.group(1)}"

        # Try pattern: _preprocessed_run_N
        match = re.search(r"_preprocessed_run_(\d+)", filename)
        if match:
            return f"run_{match.group(1)}"

        # Try pattern: isolation_forest_preprocessed_run_N
        match = re.search(r"isolation_forest_preprocessed_run_(\d+)", filename)
        if match:
            return f"run_{match.group(1)}"

        return filename.replace(".csv", "").replace(".md", "").replace(".json", "")

    def _extract_run_id_from_source(self, source_run: str) -> str:
        """
        Extract run_id from source_run like:
        - preprocessed_run_26 -> run_26
        - run_26 -> run_26
        """
        if "run_" in source_run:
            match = re.search(r"run_(\d+)", source_run)
            if match:
                return f"run_{match.group(1)}"
        return source_run

    def _find_anomaly_scores_file(self, run_id: str) -> Optional[Path]:
        """Find anomaly scores file for a run."""
        # Pattern: anomaly_scores_run_N.csv
        matches = list(self.processed_dir.glob(f"anomaly_scores_{run_id}.csv"))
        if matches:
            return matches[0]

        # Pattern: anomaly_scores_preprocessed_run_N.csv
        match = re.search(r"run_(\d+)", run_id)
        if match:
            num = match.group(1)
            matches = list(
                self.processed_dir.glob(f"anomaly_scores_preprocessed_run_{num}.csv")
            )
            if matches:
                return matches[0]

        return None

    def _find_anomaly_report_file(self, run_id: str) -> Optional[Path]:
        """Find anomaly report file for a run."""
        matches = list(self.processed_dir.glob(f"anomaly_report_{run_id}.md"))
        if matches:
            return matches[0]

        match = re.search(r"run_(\d+)", run_id)
        if match:
            num = match.group(1)
            matches = list(
                self.processed_dir.glob(f"anomaly_report_preprocessed_run_{num}.md")
            )
            if matches:
                return matches[0]

        return None

    def _find_model_metadata_file(self, run_id: str) -> Optional[Path]:
        """Find model metadata file for a run."""
        match = re.search(r"run_(\d+)", run_id)
        if match:
            num = match.group(1)
            matches = list(
                self.models_dir.glob(f"isolation_forest_*run_{num}_metadata.json")
            )
            if matches:
                return matches[0]

        return None

    def _load_model_metadata(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Load model metadata from JSON file."""
        metadata_file = self._find_model_metadata_file(run_id)
        if not metadata_file or not metadata_file.exists():
            return None

        try:
            return json.loads(metadata_file.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _repair_anomaly_report_content(self, report_content: str, metadata: Dict[str, Any]) -> str:
        """Normalize report metadata lines from the source model metadata."""
        source_run = metadata.get("source_run")
        if not source_run or str(source_run).strip().lower() == "none":
            source_run = metadata.get("source_run_token")

        source_run_token = metadata.get("source_run_token")
        if source_run_token is None or str(source_run_token).strip().lower() == "none":
            source_run_token = source_run
        if isinstance(source_run_token, str) and source_run_token.isdigit():
            source_run_token = int(source_run_token)

        repaired_report_content = re.sub(
            r"^- source_run: .*$",
            f"- source_run: {source_run}",
            report_content,
            count=1,
            flags=re.MULTILINE,
        )
        repaired_report_content = re.sub(
            r"^- source_run_token: .*$",
            f"- source_run_token: {source_run_token}",
            repaired_report_content,
            count=1,
            flags=re.MULTILINE,
        )
        return repaired_report_content

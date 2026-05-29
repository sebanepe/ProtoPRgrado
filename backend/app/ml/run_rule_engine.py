from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from backend.app.services.rule_engine_service import generate_alerts_from_preprocessed_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase B rule engine over a preprocessed CSV")
    parser.add_argument("--input", required=True, help="Path to preprocessed_run_N.csv")
    parser.add_argument("--source-run", required=True, help="Source run token, for example preprocessed_run_26")
    parser.add_argument("--output-dir", default=None, help="Directory where alerts and report will be written")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent
    started = time.perf_counter()
    result = generate_alerts_from_preprocessed_csv(str(input_path), output_dir=str(output_dir), config={"source_run": args.source_run})
    summary = result["summary"]
    validation = result["validation"]
    alerts_path = Path(result["alerts_path"])
    report_path = Path(result["report_path"])

    detailed_rows = int(summary.get("alerts_generated", 0))
    grouped_rows = int(summary.get("grouped_alerts_generated", 0))
    reduction_pct = summary.get("grouping_reduction_pct", 0.0)
    alerts_df = pd.read_csv(alerts_path)
    summary_df = pd.read_csv(result["summary_path"])

    elapsed_seconds = time.perf_counter() - started
    extra_summary = {
        "alerts_by_rule": alerts_df["rule_code"].value_counts().to_dict() if not alerts_df.empty else {},
        "alerts_by_risk_level": alerts_df["risk_level"].value_counts().to_dict() if not alerts_df.empty else {},
        "alerts_by_mcc": alerts_df["merchant_rubro_proxy"].fillna("UNKNOWN").astype(str).value_counts().to_dict() if not alerts_df.empty else {},
        "alerts_by_country": alerts_df["country_code"].fillna("UNKNOWN").astype(str).value_counts().to_dict() if not alerts_df.empty else {},
        "top_customer_hashes": alerts_df["customer_hash"].fillna("UNKNOWN").astype(str).value_counts().head(20).to_dict() if not alerts_df.empty else {},
    }

    print(f"input_rows={summary.get('input_rows', 0)}")
    print(f"detailed_alerts={detailed_rows}")
    print(f"grouped_alerts={grouped_rows}")
    print(f"grouping_reduction_pct={reduction_pct}")
    print(f"alerts_path={alerts_path}")
    print(f"summary_path={result['summary_path']}")
    print(f"report_path={report_path}")
    print(f"elapsed_seconds={elapsed_seconds:.2f}")
    print(f"alerts_by_rule={json.dumps(extra_summary['alerts_by_rule'], ensure_ascii=False)}")
    print(f"alerts_by_risk_level={json.dumps(extra_summary['alerts_by_risk_level'], ensure_ascii=False)}")
    print(f"alerts_by_mcc={json.dumps(extra_summary['alerts_by_mcc'], ensure_ascii=False)}")
    print(f"alerts_by_country={json.dumps(extra_summary['alerts_by_country'], ensure_ascii=False)}")
    print(f"top_customer_hashes={json.dumps(extra_summary['top_customer_hashes'], ensure_ascii=False)}")
    print(f"validation_verdict={validation.get('verdict')}")


if __name__ == "__main__":
    main()
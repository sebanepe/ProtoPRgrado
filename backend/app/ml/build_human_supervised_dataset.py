from __future__ import annotations

import argparse
import json

from backend.app.ml.supervised_dataset_builder import build_human_supervised_alert_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the human supervised alert dataset")
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    result = build_human_supervised_alert_dataset(args.source_run, output_dir=args.output_dir)
    print(result.get("verdict"))
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

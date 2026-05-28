import argparse
import os
import pandas as pd
from typing import Dict, List

from backend.app.ml import preprocessing

CRITICAL_RAW = [
    "PAIS",
    "POS_ENTRY_MODE",
    "TIENE_PINBLOCK",
    "CODIGO_ESTABLECIMIENTO",
    "CODIGO_TERMINAL",
    "ESTABLECIMIENTO",
    "TARJETA",
    "PAN_TARJETA",
]

CRITICAL_MAP = {
    "PAIS": "country_code",
    "POS_ENTRY_MODE": "pos_entry_mode",
    "TIENE_PINBLOCK": "has_pinblock",
    "CODIGO_ESTABLECIMIENTO": "merchant_code",
    "CODIGO_TERMINAL": "terminal_code",
    "ESTABLECIMIENTO": "merchant_name",
    "TARJETA": "masked_card",
    "PAN_TARJETA": "pan_card",
}


def _sample_values(df: pd.DataFrame, col: str, n: int = 5) -> List[str]:
    if col not in df.columns:
        return []
    vals = df[col].head(n).tolist()
    return ["" if pd.isna(v) else str(v) for v in vals]


def _report_section(title: str, lines: List[str]) -> str:
    out = [f"## {title}"]
    out.extend(lines)
    return "\n".join(out) + "\n"


def _detect_columns(raw_cols: List[str]) -> Dict[str, bool]:
    present = {}
    normalized = {c: preprocessing.normalize_column_names(pd.DataFrame(columns=[c])).columns[0] for c in raw_cols}
    norm_set = set(normalized.values())
    for raw in CRITICAL_RAW:
        raw_norm = preprocessing.normalize_column_names(pd.DataFrame(columns=[raw])).columns[0]
        present[raw] = raw_norm in norm_set
    return present


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    try:
        return preprocessing.normalize_column_names(df)
    except Exception:
        return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--preprocessed", required=False)
    parser.add_argument("--sample-size", type=int, default=1000,
                        help="Number of rows to sample from input/preprocessed files")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise SystemExit(f"input not found: {args.input}")

    raw = pd.read_csv(args.input, nrows=args.sample_size)
    raw_cols = list(raw.columns)
    norm = _normalize_df(raw)
    norm_cols = list(norm.columns)

    present = _detect_columns(raw_cols)

    report_lines = []
    report_lines.append(_report_section("Raw columns", [", ".join(raw_cols)]))
    report_lines.append(_report_section("Normalized columns", [", ".join(norm_cols)]))

    det_lines = []
    for raw_name in CRITICAL_RAW:
        det_lines.append(f"- {raw_name}: {'FOUND' if present.get(raw_name) else 'MISSING'}")
    report_lines.append(_report_section("Critical column detection", det_lines))

    value_lines = []
    for raw_name in CRITICAL_RAW:
        raw_norm = preprocessing.normalize_column_names(pd.DataFrame(columns=[raw_name])).columns[0]
        vals = _sample_values(norm, raw_norm)
        value_lines.append(f"- {raw_name} ({raw_norm}): {vals}")
    report_lines.append(_report_section("Sample raw values (normalized names)", value_lines))

    pre_lines = []
    pre = None
    if args.preprocessed:
        if not os.path.exists(args.preprocessed):
            pre_lines.append(f"preprocessed not found: {args.preprocessed}")
        else:
            pre = pd.read_csv(args.preprocessed, nrows=args.sample_size)
            pre_cols = list(pre.columns)
            pre_lines.append("Columns: " + ", ".join(pre_cols))
            for raw_name, target in CRITICAL_MAP.items():
                vals = _sample_values(pre, target)
                pre_lines.append(f"- {raw_name} -> {target}: {vals}")
    report_lines.append(_report_section("Preprocessed sample values", pre_lines))

    loss_lines = []
    for raw_name, target in CRITICAL_MAP.items():
        raw_norm = preprocessing.normalize_column_names(pd.DataFrame(columns=[raw_name])).columns[0]
        raw_present = raw_norm in norm_cols
        pre_present = False
        if pre is not None:
            pre_present = target in pre.columns
        if raw_present and not pre_present:
            loss_lines.append(f"- {raw_name} present in raw but missing in preprocessed: likely lost during ingestion or preprocessing.")
    if not loss_lines:
        loss_lines.append("- No obvious losses detected from raw->preprocessed sample.")
    report_lines.append(_report_section("Loss hints", loss_lines))

    report_lines.append(_report_section("Build training dataset hints", [
        "- If a column exists in preprocessed but not in feature_set, build_training_dataset may be dropping it.",
        "- Provide the feature_set path to validate with validate_feature_set.py.",
    ]))

    out_dir = os.path.join(os.getcwd(), "data", "processed")
    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, "debug_pipeline_mapping_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Debug pipeline mapping report\n\n")
        f.write("\n".join(report_lines))

    print(report_path)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Audit whether a CSV is suitable for TF-IDF + TruncatedSVD(LSA)."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import pandas as pd


MODEL_TARGET = "TFIDF_TRUNCATED_SVD_LSA"
REQUIRED_COLUMNS = [
    "chunk_id_out",
    "file_name_out",
    "page_out",
    "chunk_text",
    "lsa_tokens_str",
    "token_count",
    "char_len",
    "cut_reason",
    "forced_slice",
    "tokenizer_name",
    "chunk_profile",
    "model_target",
]
NOISE_TOKENS = {"pptx", "docx", "confidential", "copyright", "page", "nan", "none"}


def empty_ratio(series: pd.Series) -> float:
    if len(series) == 0:
        return 0.0
    empty = series.isna() | (series.astype(str).str.strip() == "")
    return float(empty.mean())


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, max(0, math.ceil((pct / 100.0) * len(values)) - 1))
    return float(values[idx])


def numeric_list(df: pd.DataFrame, col: str) -> list[float]:
    if col not in df.columns:
        return []
    return [float(v) for v in pd.to_numeric(df[col], errors="coerce").fillna(0).tolist()]


def audit_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    failures: list[str] = []
    warnings: list[str] = []
    present = {col: col in df.columns for col in REQUIRED_COLUMNS}
    for col, ok in present.items():
        if not ok:
            if col in {"lsa_tokens_str", "chunk_text", "token_count"}:
                failures.append(f"required column missing: {col}")
            else:
                warnings.append(f"required column missing: {col}")

    metrics: dict[str, Any] = {"rows": int(len(df))}
    if "lsa_tokens_str" in df.columns:
        metrics["empty_lsa_tokens_ratio"] = empty_ratio(df["lsa_tokens_str"])
        if metrics["empty_lsa_tokens_ratio"] > 0.05:
            failures.append("more than 5% rows have empty lsa_tokens_str")
    if "chunk_text" in df.columns:
        metrics["empty_chunk_text_ratio"] = empty_ratio(df["chunk_text"])
        if metrics["empty_chunk_text_ratio"] > 0.05:
            failures.append("more than 5% rows have empty chunk_text")
    if "model_target" in df.columns:
        invalid_ratio = float((df["model_target"].astype(str) != MODEL_TARGET).mean()) if len(df) else 0.0
        metrics["invalid_model_target_ratio"] = invalid_ratio
        if invalid_ratio > 0:
            failures.append(f"model_target must equal {MODEL_TARGET}")
    if "chunk_id_out" in df.columns and len(df):
        duplicate_ratio = float(df["chunk_id_out"].duplicated().mean())
        metrics["duplicate_chunk_id_out_ratio"] = duplicate_ratio
        if duplicate_ratio > 0.01:
            failures.append("duplicate chunk_id_out ratio > 1%")

    token_counts = numeric_list(df, "token_count")
    char_lens = numeric_list(df, "char_len")
    metrics["median_token_count"] = percentile(token_counts, 50)
    metrics["p95_char_len"] = percentile(char_lens, 95)
    if token_counts:
        if metrics["median_token_count"] < 5:
            warnings.append("median token_count < 5")
        if metrics["median_token_count"] > 120:
            warnings.append("median token_count > 120")
    if char_lens and metrics["p95_char_len"] > 1800:
        warnings.append("p95 char_len > 1800")
    if "forced_slice" in df.columns:
        forced = df["forced_slice"].astype(str).str.lower().isin({"true", "1", "yes"})
        metrics["forced_slice_cut_rate"] = float(forced.mean()) if len(df) else 0.0
        if metrics["forced_slice_cut_rate"] > 0.25:
            warnings.append("forced_slice cut_rate > 0.25")

    for col in ("file_name_out", "page_out"):
        if col not in df.columns:
            warnings.append(f"{col} is missing")
        elif empty_ratio(df[col]) > 0.5:
            warnings.append(f"{col} is mostly empty")

    counter: Counter[str] = Counter()
    if "lsa_tokens_str" in df.columns:
        for value in df["lsa_tokens_str"].fillna(""):
            counter.update(str(value).split())
    top_tokens = [token for token, _ in counter.most_common(50)]
    metrics["top_tokens"] = top_tokens[:20]
    if any(token.lower() in NOISE_TOKENS for token in top_tokens):
        warnings.append("top tokens contain obvious document noise")
    total_tokens = sum(counter.values())
    one_char = sum(count for token, count in counter.items() if len(token) == 1)
    numeric_only = sum(count for token, count in counter.items() if token.isdigit())
    metrics["one_character_token_ratio"] = one_char / max(total_tokens, 1)
    metrics["numeric_only_token_ratio"] = numeric_only / max(total_tokens, 1)
    if metrics["one_character_token_ratio"] > 0.20:
        warnings.append("too many one-character tokens")
    if metrics["numeric_only_token_ratio"] > 0.20:
        warnings.append("too many numeric-only tokens")

    score = 100
    score -= 30 * len(failures)
    score -= 8 * len(warnings)
    score = max(0, min(100, score))
    verdict = "FAIL" if failures else ("WARN" if warnings else "PASS")
    return {
        "verdict": verdict,
        "score": score,
        "failures": failures,
        "warnings": warnings,
        "metrics": metrics,
        "required_columns_present": present,
        "recommended_text_col_for_lsa": "lsa_tokens_str",
        "recommended_display_col": "chunk_text",
        "recommended_file_col": "file_name_out",
        "recommended_page_col": "page_out",
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# LSA CSV Audit",
        "",
        f"- Verdict: {report['verdict']}",
        f"- Score: {report['score']}",
        f"- Recommended LSA text column: `{report['recommended_text_col_for_lsa']}`",
        f"- Recommended display column: `{report['recommended_display_col']}`",
        f"- Recommended file column: `{report['recommended_file_col']}`",
        f"- Recommended page column: `{report['recommended_page_col']}`",
        "",
        "## Failures",
        "",
    ]
    if report["failures"]:
        lines.extend(f"- {item}" for item in report["failures"])
    else:
        lines.append("- None")
    lines.extend(["", "## Warnings", ""])
    if report["warnings"]:
        lines.extend(f"- {item}" for item in report["warnings"])
    else:
        lines.append("- None")
    lines.extend(["", "## Metrics", "", "```json", json.dumps(report["metrics"], ensure_ascii=False, indent=2), "```", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--text-col", default="lsa_tokens_str")
    parser.add_argument("--display-col", default="chunk_text")
    parser.add_argument("--file-col", default="file_name_out")
    parser.add_argument("--page-col", default="page_out")
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--output-md")
    parser.add_argument("--encoding", default="utf-8-sig")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    df = pd.read_csv(args.input, encoding=args.encoding)
    report = audit_dataframe(df)
    Path(args.output_report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.output_md:
        write_markdown(report, Path(args.output_md))
    print(json.dumps({"verdict": report["verdict"], "score": report["score"]}, ensure_ascii=False))
    return 1 if report["verdict"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())

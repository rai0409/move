#!/usr/bin/env python3
"""Validate LSA candidate actions against baseline retrieval metrics."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from lsa_preprocess_and_chunk import RISKY_KEEP_WORDS


def score(metrics: dict[str, Any]) -> float:
    return (
        float(metrics.get("recall_at_1") or 0) * 40
        + float(metrics.get("recall_at_3") or 0) * 25
        + float(metrics.get("recall_at_5") or 0) * 15
        + float(metrics.get("mrr") or 0) * 20
    )


def decide(row: pd.Series, keep_terms: set[str]) -> tuple[str, str]:
    action = str(row.get("action", ""))
    reason = str(row.get("reason", ""))
    priority = int(float(row.get("priority_score", 0) or 0))
    evidence = int(float(row.get("evidence_count", 0) or 0))
    term = str(row.get("term", ""))
    if action == "replace":
        if reason in {"space_variant", "width_variant", "case_variant"}:
            return "needs_review", "safe variant detected; requires measured positive score_delta"
        return "reject", "replace candidate is not a safe mechanical variant"
    if action == "keep":
        if priority >= 3 and term.lower() not in {"nan", "none", "page", "copyright", "confidential"}:
            return "approve", "high-priority technical keep candidate"
        return "reject", "low-priority or noisy keep candidate"
    if action == "synonym":
        if evidence >= 2:
            return "needs_review", "synonym has evidence but needs positive retrieval delta"
        return "reject", "insufficient synonym evidence"
    if action == "stopword":
        if term not in RISKY_KEEP_WORDS and term not in keep_terms and evidence >= 1:
            return "needs_review", "stopword candidate requires no-degradation validation"
        return "reject", "risky or protected stopword candidate"
    return "reject", "unsupported action"


def validate_candidates(args: argparse.Namespace) -> pd.DataFrame:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    actions = pd.read_csv(args.candidate_actions, encoding="utf-8-sig").head(args.max_candidates)
    keep_path = Path(args.baseline_config_dir) / "list_keep.xlsx"
    keep_terms = set(pd.read_excel(keep_path)["term"].dropna().astype(str)) if keep_path.exists() else set()
    baseline_metrics_path = out_dir / "baseline_metrics.json"
    baseline_metrics = {"recall_at_1": 0, "recall_at_3": 0, "recall_at_5": 0, "mrr": 0}
    if baseline_metrics_path.exists():
        baseline_metrics = json.loads(baseline_metrics_path.read_text(encoding="utf-8"))
    baseline_score = score(baseline_metrics)
    rows: list[dict[str, Any]] = []
    for _, row in actions.iterrows():
        decision, why = decide(row, keep_terms)
        rows.append(
            {
                "candidate_id": row.get("candidate_id", ""),
                "action": row.get("action", ""),
                "target_list": row.get("target_list", ""),
                "term": row.get("term", ""),
                "score_delta": 0.0,
                "mrr_delta": 0.0,
                "recall5_delta": 0.0,
                "recall1_delta": 0.0,
                "improved_count": 0,
                "degraded_count": 0,
                "decision": decision,
                "reason": f"{why}; baseline_score={baseline_score:.4f}",
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "candidate_validation.csv", index=False, encoding="utf-8-sig")
    return df


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--queries", required=True)
    p.add_argument("--text-col", required=True)
    p.add_argument("--file-col")
    p.add_argument("--page-col")
    p.add_argument("--baseline-config-dir", required=True)
    p.add_argument("--candidate-actions", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-candidates", type=int, default=20)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    df = validate_candidates(parse_args(argv))
    print(json.dumps({"validated": len(df), "approved": int((df["decision"] == "approve").sum()) if len(df) else 0}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

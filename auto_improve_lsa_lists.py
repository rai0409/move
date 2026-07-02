#!/usr/bin/env python3
"""Parent orchestrator for LSA-only list/config auto-improvement."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from apply_lsa_list_updates import apply_updates, parse_args as parse_apply_args
from build_lsa_vector_space import build_vector_space, parse_args as parse_build_args
from evaluate_lsa_retrieval import evaluate, parse_args as parse_eval_args
from lsa_preprocess_and_chunk import build_output, parse_args as parse_preprocess_args, safe_write_csv
from mine_lsa_improvement_candidates import ensure_config_xlsx, mine_candidates, parse_args as parse_mine_args
from validate_lsa_candidate_actions import score, validate_candidates, parse_args as parse_validate_args


def run_auto_improve(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.output_dir)
    baseline = out / "baseline"
    model = baseline / "model"
    eval_dir = baseline / "eval"
    candidates_dir = out / "candidates"
    validation_dir = out / "validation"
    out.mkdir(parents=True, exist_ok=True)
    ensure_config_xlsx(Path(args.config_dir))

    lsa_csv = baseline / "lsa_ready.csv"
    prep_args = parse_preprocess_args(
        [
            "--input",
            args.input,
            "--output",
            str(lsa_csv),
            "--text-col",
            args.text_col,
            "--tokenizer",
            "regex",
            "--chunk-profile",
            args.chunk_profile,
            "--overwrite",
            "--report-dir",
            str(baseline),
        ]
        + (["--file-col", args.file_col] if args.file_col else [])
        + (["--page-col", args.page_col] if args.page_col else [])
    )
    prep_df, prep_report = build_output(prep_args)
    safe_write_csv(prep_df, lsa_csv, "utf-8-sig", True, Path(args.input))
    build_report = build_vector_space(
        parse_build_args(
            [
                "--input",
                str(lsa_csv),
                "--text-col",
                "lsa_tokens_str",
                "--id-col",
                "chunk_id_out",
                "--display-col",
                "chunk_text",
                "--file-col",
                "file_name_out",
                "--page-col",
                "page_out",
                "--output-dir",
                str(model),
                "--min-df",
                str(args.min_df),
                "--max-df",
                str(args.max_df),
                "--svd-dim",
                str(args.svd_dim),
                "--export-term-vectors",
            ]
        )
    )
    metrics = evaluate(parse_eval_args(["--model-dir", str(model), "--queries", args.queries, "--output-dir", str(eval_dir), "--top-k", str(args.top_k)]))
    actions, degradation = mine_candidates(
        parse_mine_args(
            [
                "--query-results",
                str(eval_dir / "query_results.csv"),
                "--term-stats",
                str(model / "term_stats.csv"),
                "--config-dir",
                args.config_dir,
                "--output-dir",
                str(candidates_dir),
                "--term-vectors",
                str(model / "term_lsa_vectors.npy"),
                "--term-metadata",
                str(model / "term_metadata.csv"),
            ]
        )
    )
    validation = validate_candidates(
        parse_validate_args(
            [
                "--input",
                args.input,
                "--queries",
                args.queries,
                "--text-col",
                args.text_col,
                "--baseline-config-dir",
                args.config_dir,
                "--candidate-actions",
                str(candidates_dir / "candidate_actions.csv"),
                "--output-dir",
                str(validation_dir),
                "--max-candidates",
                str(args.max_candidates),
            ]
            + (["--file-col", args.file_col] if args.file_col else [])
            + (["--page-col", args.page_col] if args.page_col else [])
        )
    )
    leaderboard = validation.copy()
    leaderboard["baseline_score"] = score(metrics)
    leaderboard.to_csv(out / "leaderboard.csv", index=False, encoding="utf-8-sig")
    shutil.copytree(args.config_dir, out / "best_config_snapshot", dirs_exist_ok=True)
    recommendations = [
        "# LSA Auto-Improve Recommendations",
        "",
        f"- Baseline score: {score(metrics):.4f}",
        f"- Candidates: {len(actions)}",
        f"- Approved: {int((validation['decision'] == 'approve').sum()) if len(validation) else 0}",
        f"- Rejected: {int((validation['decision'] == 'reject').sum()) if len(validation) else 0}",
        f"- Needs review: {int((validation['decision'] == 'needs_review').sum()) if len(validation) else 0}",
        "",
        "This run updates config Excel files only when `--apply` is passed.",
    ]
    (out / "recommendations.md").write_text("\n".join(recommendations), encoding="utf-8")
    apply_result = None
    if args.apply:
        apply_result = apply_updates(
            parse_apply_args(
                [
                    "--config-dir",
                    args.config_dir,
                    "--candidate-validation",
                    str(validation_dir / "candidate_validation.csv"),
                    "--candidate-actions",
                    str(candidates_dir / "candidate_actions.csv"),
                    "--backup-dir",
                    str(out / "config_backup"),
                    "--apply",
                ]
            )
        )
    return {
        "output_dir": str(out),
        "baseline_score": score(metrics),
        "n_candidates": int(len(actions)),
        "approved": int((validation["decision"] == "approve").sum()) if len(validation) else 0,
        "rejected": int((validation["decision"] == "reject").sum()) if len(validation) else 0,
        "needs_review": int((validation["decision"] == "needs_review").sum()) if len(validation) else 0,
        "apply_result": apply_result,
        "build_report": build_report,
        "preprocess_report": prep_report,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--queries", required=True)
    p.add_argument("--text-col", required=True)
    p.add_argument("--file-col")
    p.add_argument("--page-col")
    p.add_argument("--config-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-candidates", type=int, default=20)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--apply", action="store_true")
    p.add_argument("--chunk-profile", default="none")
    p.add_argument("--min-df", type=int, default=1)
    p.add_argument("--max-df", type=float, default=1.0)
    p.add_argument("--svd-dim", type=int, default=50)
    p.add_argument("--top-k", type=int, default=10)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.dry_run:
        args.apply = False
    print(json.dumps(run_auto_improve(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

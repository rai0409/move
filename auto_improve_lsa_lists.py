#!/usr/bin/env python3
"""Parent orchestrator for LSA list/config auto-improvement using existing make_ce_v1 backend."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from apply_lsa_list_updates import apply_updates, parse_args as parse_apply_args
from evaluate_lsa_retrieval import evaluate, parse_args as parse_eval_args
from lsa_preprocess_and_chunk import build_output, parse_args as parse_preprocess_args, safe_write_csv
from make_ce_backend import build_make_ce_vectors, parse_args as parse_make_ce_args
from mine_lsa_improvement_candidates import ensure_config_xlsx, mine_candidates, parse_args as parse_mine_args
from validate_lsa_candidate_actions import score, validate_candidates, parse_args as parse_validate_args


def run_auto_improve(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.output_dir)
    baseline = out / "baseline"
    make_ce_report_dir = baseline / "make_ce"
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

    make_ce_result = build_make_ce_vectors(
        parse_make_ce_args(
            [
                "--lsa-ready",
                str(lsa_csv),
                "--base-dir",
                args.base_dir,
                "--craw-name",
                args.craw_name,
                "--make-ce-script",
                args.make_ce_script,
                "--output-dir",
                str(make_ce_report_dir),
                "--n-clusters",
                str(args.n_clusters),
                "--chunk-id-col",
                "chunk_id_out",
                "--file-col",
                "file_name_out",
                "--page-col",
                "page_out",
                "--text-col",
                "chunk_text",
            ]
            + (["--python-exe", args.python_exe] if args.python_exe else [])
            + (["--skip-space"] if args.skip_space else [])
            + (["--skip-vectors"] if args.skip_vectors else [])
        )
    )

    eval_argv = [
        "--backend",
        "make_ce_direct",
        "--model-dir",
        make_ce_result["model_dir"],
        "--queries",
        args.queries,
        "--output-dir",
        str(eval_dir),
        "--top-k",
        str(args.top_k),
        "--file-col",
        "name_org",
        "--page-col",
        "pageno",
        "--display-col",
        "text",
        "--id-col",
        "chunk_id",
        "--vector-text-col",
        args.vector_text_col,
    ]

    if args.use_mean_tfidf:
        eval_argv.append("--use-mean-tfidf")
    if args.allow_dense_mean:
        eval_argv.append("--allow-dense-mean")

    metrics = evaluate(parse_eval_args(eval_argv))

    mine_argv = [
        "--query-results",
        str(eval_dir / "query_results.csv"),
        "--config-dir",
        args.config_dir,
        "--output-dir",
        str(candidates_dir),
    ]

    term_stats_path = Path(make_ce_result["model_dir"]) / "step1" / "term_stats.csv"
    if term_stats_path.exists():
        mine_argv.extend(["--term-stats", str(term_stats_path)])

    actions, degradation = mine_candidates(parse_mine_args(mine_argv))

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
        "- Backend: make_ce_v1.py",
        f"- craw_name: {args.craw_name}",
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
        "backend": "make_ce",
        "craw_name": args.craw_name,
        "baseline_score": score(metrics),
        "n_candidates": int(len(actions)),
        "approved": int((validation["decision"] == "approve").sum()) if len(validation) else 0,
        "rejected": int((validation["decision"] == "reject").sum()) if len(validation) else 0,
        "needs_review": int((validation["decision"] == "needs_review").sum()) if len(validation) else 0,
        "apply_result": apply_result,
        "make_ce_result": make_ce_result,
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

    p.add_argument("--base-dir", default=r"C:\project\document_viewer\_data_assets")
    p.add_argument("--craw-name", required=True)
    p.add_argument("--make-ce-script", default="make_ce_v1.py")
    p.add_argument("--python-exe")
    p.add_argument("--n-clusters", type=int, default=0)

    p.add_argument("--vector-text-col", default="auto")
    p.add_argument("--use-mean-tfidf", action="store_true")
    p.add_argument("--allow-dense-mean", action="store_true")

    p.add_argument("--skip-space", action="store_true")
    p.add_argument("--skip-vectors", action="store_true")

    p.add_argument("--max-candidates", type=int, default=20)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--apply", action="store_true")
    p.add_argument("--chunk-profile", default="none")
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
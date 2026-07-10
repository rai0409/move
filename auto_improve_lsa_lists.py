#!/usr/bin/env python3
"""Parent orchestrator for LSA list/config auto-improvement using existing make_ce_v1 backend."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from apply_lsa_list_updates import apply_candidate_actions_for_eval, apply_updates, parse_args as parse_apply_args
from evaluate_lsa_retrieval import evaluate, parse_args as parse_eval_args
from lsa_preprocess_and_chunk import PRODUCTION_TOKENIZERS, build_output, parse_args as parse_preprocess_args, safe_write_csv
from make_ce_backend import build_make_ce_vectors, parse_args as parse_make_ce_args
from mine_lsa_improvement_candidates import ensure_config_xlsx, mine_candidates, parse_args as parse_mine_args
from validate_lsa_candidate_actions import critical_query_regressions, query_rank_deltas, score, validate_candidates, parse_args as parse_validate_args


MINIMAL_TUNING_CANDIDATES = [
    {"tuning_candidate_id": name, "tokenizer": name, "chunk_profile": "current", "morphology_profile": "content_lemma", "min_df": None, "max_df": None, "svd_dim": None, "phase": 1}
    for name in ["mecab_ipadic_utf8_v102", "sudachi_small_b", "sudachi_small_c", "sudachi_core_b", "sudachi_core_c"]
]

CHUNK_PROFILE_PARAMS = {
    "none": {"chunk_size": None, "overlap": None},
    "small": {"chunk_size": 450, "overlap": 60},
    "medium": {"chunk_size": 750, "overlap": 100},
    "large": {"chunk_size": 1050, "overlap": 150},
    "auto": {"chunk_size": "auto", "overlap": "auto"},
    "current": {"chunk_size": 750, "overlap": 100},
}


def run_retrieval_measurement(
    args: argparse.Namespace,
    *,
    config_dir: str,
    run_dir: Path,
    label: str,
    tuning_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    lsa_csv = run_dir / "lsa_ready.csv"
    make_ce_report_dir = run_dir / "make_ce"
    eval_dir = run_dir / "eval"

    tokenizer_name = str((tuning_params or {}).get("tokenizer") or getattr(args, "tokenizer", "mecab_ipadic_utf8_v102"))
    chunk_profile = str((tuning_params or {}).get("chunk_profile") or args.chunk_profile)
    eval_top_k = max(int(args.top_k), 20)

    print(
        f"[auto_improve] {label} preprocess input="
        f"{args.input} output={lsa_csv} text_col={args.text_col} config_dir={config_dir} "
        f"tokenizer={tokenizer_name} chunk_profile={chunk_profile}"
    )
    prep_args = parse_preprocess_args(
        [
            "--input",
            args.input,
            "--output",
            str(lsa_csv),
            "--text-col",
            args.text_col,
            "--tokenizer",
            tokenizer_name,
            "--chunk-profile",
            chunk_profile,
            "--config-dir",
            config_dir,
            "--morphology-profile",
            str((tuning_params or {}).get("morphology_profile") or args.morphology_profile),
            "--overwrite",
            "--report-dir",
            str(run_dir),
        ]
        + (["--file-col", args.file_col] if args.file_col else [])
        + (["--page-col", args.page_col] if args.page_col else [])
        + (["--dictionary-path", args.dictionary_path] if args.dictionary_path and tokenizer_name == "mecab_ipadic_utf8_v102" else [])
        + (["--mecab-options", args.mecab_options] if args.mecab_options else [])
    )
    prep_df, prep_report = build_output(prep_args)
    safe_write_csv(prep_df, lsa_csv, "utf-8-sig", True, Path(args.input))

    print(f"[auto_improve] {label} make_ce input={lsa_csv} text_col=lsa_tokens_str display_col=chunk_text output_dir={make_ce_report_dir}")
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
                "lsa_tokens_str",
                "--display-text-col",
                "chunk_text",
                "--docid-col",
                "docid",
                "--preprocessing-profile",
                str(run_dir / "preprocessing_profile.json"),
            ]
            + (["--python-exe", args.python_exe] if args.python_exe else [])
            + (["--skip-space"] if args.skip_space else [])
            + (["--skip-vectors"] if args.skip_vectors else [])
        )
    )

    print(
        f"[auto_improve] {label} evaluate model_dir="
        f"{make_ce_result['model_dir']} queries={args.queries} vector_text_col={args.vector_text_col} output_dir={eval_dir}"
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
        str(eval_top_k),
        "--file-col",
        "name_org",
        "--page-col",
        "pageno",
        "--display-col",
        "chunk_text",
        "--id-col",
        "chunk_id",
        "--docid-col",
        "docid",
        "--vector-text-col",
        args.vector_text_col,
        "--chunk-profile",
        chunk_profile,
        "--svd-dim",
        "100",
        "--query-preprocessing-profile",
        str(run_dir / "preprocessing_profile.json"),
    ]
    if args.queries_encoding:
        eval_argv.extend(["--queries-encoding", args.queries_encoding])
    if args.use_mean_tfidf:
        eval_argv.append("--use-mean-tfidf")
    if args.allow_dense_mean:
        eval_argv.append("--allow-dense-mean")

    metrics = evaluate(parse_eval_args(eval_argv))
    return {
        "lsa_csv": str(lsa_csv),
        "make_ce_result": make_ce_result,
        "metrics": metrics,
        "metrics_path": str(eval_dir / "metrics.json"),
        "results_path": str(eval_dir / "query_results.csv"),
        "ranked_results_path": str(eval_dir / "ranked_results.csv"),
        "eval_dir": str(eval_dir),
        "preprocess_report": prep_report,
        "measurement_params": {
            "tokenizer": tokenizer_name,
            "chunk_profile": chunk_profile,
            "top_k": eval_top_k,
        },
    }


def metric_delta_dict(before: dict[str, Any] | None, after: dict[str, Any] | None) -> dict[str, float | None]:
    keys = ["recall_at_1", "recall_at_3", "recall_at_5", "recall_at_10", "recall_at_20", "mrr", "mean_rank", "no_hit_query_count"]
    out: dict[str, float | None] = {}
    before = before or {}
    after = after or {}
    for key in keys:
        try:
            b = float(before.get(key)) if before.get(key) is not None else None
            a = float(after.get(key)) if after.get(key) is not None else None
        except Exception:
            b = None
            a = None
        out[key] = None if b is None or a is None else a - b
    out["composite_score"] = score(after) - score(before)
    return out


def decide_tuning_candidate(delta: dict[str, float | None], critical_query_regression: bool, measured: bool,
                            evaluated_query_count: int = 0, min_evaluated_queries: int = 1) -> tuple[str, str]:
    if not measured:
        return "needs_measurement", "candidate retrieval metrics were not measured"
    if critical_query_regression:
        return "reject", "critical query regression detected"
    if evaluated_query_count < min_evaluated_queries:
        return "needs_more_evidence", f"evaluated queries {evaluated_query_count} below minimum {min_evaluated_queries}"
    r1 = delta.get("recall_at_1")
    r5 = delta.get("recall_at_5")
    r20 = delta.get("recall_at_20")
    mrr = delta.get("mrr")
    composite = delta.get("composite_score")
    if r1 is not None and r1 < 0:
        return "reject", f"recall_at_1 dropped: {r1:.4f}"
    if r5 is not None and r5 < 0:
        return "reject", f"recall_at_5 dropped: {r5:.4f}"
    if r20 is not None and r20 < 0:
        return "reject", f"recall_at_20 dropped: {r20:.4f}"
    if mrr is not None and mrr < 0:
        return "reject", f"mrr dropped: {mrr:.4f}"
    if (r1 or 0) >= 0.01 or (mrr or 0) >= 0.01:
        return "approve", f"measured threshold met: recall@1 delta={r1 or 0:.4f}, mrr delta={mrr or 0:.4f}"
    return "needs_more_evidence", "no degradation, but recall@1 and MRR gains are below 0.01"


def run_param_tuning_candidates(
    args: argparse.Namespace,
    *,
    baseline_measurement: dict[str, Any],
    baseline_config_dir: str,
    out_dir: Path,
) -> dict[str, Any] | None:
    if not args.enable_param_tuning:
        return None

    tuning_dir = out_dir / "tuning"
    tuning_dir.mkdir(parents=True, exist_ok=True)
    candidates = MINIMAL_TUNING_CANDIDATES[: max(0, int(args.max_tuning_candidates))]
    rows: list[dict[str, Any]] = []

    for candidate in candidates:
        candidate_id = candidate["tuning_candidate_id"]
        candidate_dir = tuning_dir / candidate_id
        candidate_config_dir = candidate_dir / "candidate_config"
        blockers: list[str] = []

        if candidate_config_dir.exists():
            shutil.rmtree(candidate_config_dir)
        shutil.copytree(baseline_config_dir, candidate_config_dir)

        active_params: dict[str, Any] = {"tokenizer": candidate["tokenizer"]}
        inactive_params = {
            "min_df": candidate["min_df"], "max_df": candidate["max_df"], "svd_dim": candidate["svd_dim"],
        }
        blockers.append(
            "make_ce backend does not expose min_df/max_df/svd_dim controls; these params are recorded but inactive in this path"
        )

        metrics = None
        candidate_metrics_path = None
        candidate_results_path = None
        measurement_status = "not_run"
        try:
            measurement = run_retrieval_measurement(
                args, config_dir=str(candidate_config_dir), run_dir=candidate_dir / "measurement",
                label=f"tuning:{candidate_id}", tuning_params=candidate,
            )
            metrics = measurement["metrics"]
            candidate_metrics_path = measurement["metrics_path"]
            candidate_results_path = measurement["results_path"]
            baseline_lineage_path = Path(baseline_measurement["eval_dir"]) / "evaluation_lineage.json"
            candidate_lineage_path = Path(measurement["eval_dir"]) / "evaluation_lineage.json"
            baseline_lineage = json.loads(baseline_lineage_path.read_text(encoding="utf-8"))
            candidate_lineage = json.loads(candidate_lineage_path.read_text(encoding="utf-8"))
            if (
                baseline_lineage.get("query_csv_fingerprint") != candidate_lineage.get("query_csv_fingerprint")
                or baseline_lineage.get("profiles_match") is not True
                or candidate_lineage.get("profiles_match") is not True
            ):
                raise RuntimeError("query fingerprint or document/query preprocessing profile mismatch")
            measurement_status = "measured"
        except Exception as exc:
            blockers.append(f"candidate measurement failed: {type(exc).__name__}: {exc}")
            metrics = None
            candidate_metrics_path = None
            candidate_results_path = None
            measurement_status = "insufficient_evidence"

        delta = metric_delta_dict(baseline_measurement.get("metrics"), metrics)
        measured = metrics is not None
        regressions = critical_query_regressions(
            baseline_measurement.get("results_path"), candidate_results_path, max_rank_drop=5
        ) if candidate_results_path else []
        ranks = query_rank_deltas(baseline_measurement.get("results_path"), candidate_results_path)
        decision, reason = decide_tuning_candidate(
            delta, bool(regressions), measured,
            int((metrics or {}).get("evaluated_query_count") or 0), args.min_evaluated_queries,
        )
        validation_mode = "measured" if measured else "insufficient_evidence"
        params = {**candidate, **CHUNK_PROFILE_PARAMS.get(str(candidate["chunk_profile"]), {})}

        row = {
            "validation_mode": validation_mode,
            "tuning_candidate_id": candidate_id,
            "params": params,
            "active_params": active_params,
            "inactive_params": inactive_params,
            "baseline_config_dir": baseline_config_dir,
            "candidate_config_dir": str(candidate_config_dir),
            "baseline_metrics_path": baseline_measurement["metrics_path"],
            "candidate_metrics_path": candidate_metrics_path,
            "baseline_results_path": baseline_measurement["results_path"],
            "candidate_results_path": candidate_results_path,
            "vector_text_col": args.vector_text_col,
            "query_path": args.queries,
            "baseline_metrics": baseline_measurement.get("metrics"),
            "candidate_metrics": metrics,
            "metric_delta": delta,
            "recall_at_1_delta": delta.get("recall_at_1"),
            "recall_at_5_delta": delta.get("recall_at_5"),
            "recall_at_20_delta": delta.get("recall_at_20"),
            "mrr_delta": delta.get("mrr"),
            "composite_score_delta": delta.get("composite_score"),
            "mean_rank_delta": delta.get("mean_rank"),
            "no_hit_count_delta": delta.get("no_hit_query_count"),
            **ranks,
            "critical_query_regression": bool(regressions),
            "critical_query_regressions": regressions,
            "decision": decision,
            "reason": reason,
            "measurement_status": measurement_status,
            "environment_blockers": blockers,
        }
        candidate_dir.mkdir(parents=True, exist_ok=True)
        (candidate_dir / "tuning_validation_result.json").write_text(
            json.dumps(row, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        rows.append(row)

    summary = {
        "validation_mode": "measured" if any(r["validation_mode"] == "measured" for r in rows) else "insufficient_evidence",
        "tuning_profile": args.tuning_profile,
        "candidate_count": len(rows),
        "candidates": rows,
    }
    (tuning_dir / "tuning_validation_result.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "tuning_candidate_id": r["tuning_candidate_id"],
                "validation_mode": r["validation_mode"],
                "measurement_status": r["measurement_status"],
                "tokenizer": r["params"]["tokenizer"],
                "chunk_profile": r["params"]["chunk_profile"],
                "chunk_size": r["params"]["chunk_size"],
                "overlap": r["params"]["overlap"],
                "min_df": r["params"]["min_df"],
                "max_df": r["params"]["max_df"],
                "svd_dim": r["params"]["svd_dim"],
                "recall_at_1_delta": r["recall_at_1_delta"],
                "recall_at_5_delta": r["recall_at_5_delta"],
                "recall_at_20_delta": r["recall_at_20_delta"],
                "mrr_delta": r["mrr_delta"],
                "composite_score_delta": r["composite_score_delta"],
                "mean_rank_delta": r["mean_rank_delta"],
                "no_hit_count_delta": r["no_hit_count_delta"],
                "improved_query_count": r["improved_query_count"],
                "regressed_query_count": r["regressed_query_count"],
                "unchanged_query_count": r["unchanged_query_count"],
                "worst_rank_drop": r["worst_rank_drop"],
                "best_rank_gain": r["best_rank_gain"],
                "decision": r["decision"],
                "reason": r["reason"],
            }
            for r in rows
        ]
    ).to_csv(tuning_dir / "tuning_leaderboard.csv", index=False, encoding="utf-8-sig")
    return summary


def run_auto_improve(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.output_dir)
    baseline = out / "baseline"
    make_ce_report_dir = baseline / "make_ce"
    eval_dir = baseline / "eval"
    candidates_dir = out / "candidates"
    validation_dir = out / "validation"

    out.mkdir(parents=True, exist_ok=True)
    ensure_config_xlsx(Path(args.config_dir))

    baseline_measurement = run_retrieval_measurement(args, config_dir=args.config_dir, run_dir=baseline, label="baseline")
    tuning_result = run_param_tuning_candidates(
        args,
        baseline_measurement=baseline_measurement,
        baseline_config_dir=args.config_dir,
        out_dir=out,
    )
    lsa_csv = Path(baseline_measurement["lsa_csv"])
    make_ce_result = baseline_measurement["make_ce_result"]
    metrics = baseline_measurement["metrics"]
    prep_report = baseline_measurement["preprocess_report"]
    print(
        "[auto_improve] mine query_results="
        f"{eval_dir / 'ranked_results.csv'} config_dir={args.config_dir} output_dir={candidates_dir}"
    )

    mine_argv = [
        "--query-results",
        str(eval_dir / "ranked_results.csv"),
        "--config-dir",
        args.config_dir,
        "--output-dir",
        str(candidates_dir),
    ]

    term_stats_path = Path(make_ce_result["model_dir"]) / "step1" / "term_stats.csv"
    if term_stats_path.exists():
        mine_argv.extend(["--term-stats", str(term_stats_path)])

    actions, degradation = mine_candidates(parse_mine_args(mine_argv))

    candidate_config_dir = out / "candidate_config"
    if candidate_config_dir.exists():
        shutil.rmtree(candidate_config_dir)
    shutil.copytree(args.config_dir, candidate_config_dir)
    candidate_config_result = apply_candidate_actions_for_eval(
        candidate_config_dir,
        candidates_dir / "candidate_actions.csv",
        max_candidates=args.max_candidates,
    )

    candidate_measurement = None
    candidate_eval_blockers: list[str] = []
    if len(actions):
        try:
            candidate_measurement = run_retrieval_measurement(
                args,
                config_dir=str(candidate_config_dir),
                run_dir=out / "candidate_measurement",
                label="candidate",
            )
        except Exception as exc:
            candidate_eval_blockers.append(f"candidate measurement failed: {type(exc).__name__}: {exc}")
            (validation_dir / "candidate_measurement_error.json").parent.mkdir(parents=True, exist_ok=True)
            (validation_dir / "candidate_measurement_error.json").write_text(
                json.dumps(
                    {
                        "candidate_config_path": str(candidate_config_dir),
                        "required_inputs": {
                            "input": args.input,
                            "queries": args.queries,
                            "vector_text_col": args.vector_text_col,
                            "make_ce_script": args.make_ce_script,
                            "base_dir": args.base_dir,
                        },
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

    validate_argv = [
        "--input",
        args.input,
        "--queries",
        args.queries,
        "--text-col",
        args.text_col,
        "--baseline-config-dir",
        args.config_dir,
        "--candidate-config-dir",
        str(candidate_config_dir),
        "--candidate-actions",
        str(candidates_dir / "candidate_actions.csv"),
        "--output-dir",
        str(validation_dir),
        "--max-candidates",
        str(args.max_candidates),
        "--baseline-metrics",
        baseline_measurement["metrics_path"],
        "--baseline-results",
        baseline_measurement["results_path"],
        "--vector-text-col",
        args.vector_text_col,
        "--min-evaluated-queries",
        str(args.min_evaluated_queries),
    ]
    if candidate_measurement:
        validate_argv.extend(
            [
                "--candidate-metrics",
                candidate_measurement["metrics_path"],
                "--candidate-results",
                candidate_measurement["results_path"],
            ]
        )
    validation = validate_candidates(
        parse_validate_args(
            validate_argv
            + (["--file-col", args.file_col] if args.file_col else [])
            + (["--page-col", args.page_col] if args.page_col else [])
        )
    )

    lineage = {
        "input_csv": args.input,
        "input_text_col": args.text_col,
        "queries_csv": args.queries,
        "config_dir": args.config_dir,
        "lsa_ready_csv": str(lsa_csv),
        "lsa_ready_text_col_for_make_ce": "lsa_tokens_str",
        "lsa_ready_display_col_for_make_ce": "chunk_text",
        "make_ce_df0_csv": make_ce_result.get("df0_csv"),
        "make_ce_model_dir": make_ce_result.get("model_dir"),
        "evaluate_vector_text_col": args.vector_text_col,
        "evaluate_output_dir": str(eval_dir),
        "candidates_dir": str(candidates_dir),
        "validation_dir": str(validation_dir),
        "baseline_metrics_path": baseline_measurement["metrics_path"],
        "baseline_results_path": baseline_measurement["results_path"],
        "candidate_config_dir": str(candidate_config_dir),
        "candidate_config_result": candidate_config_result,
        "candidate_metrics_path": candidate_measurement["metrics_path"] if candidate_measurement else None,
        "candidate_results_path": candidate_measurement["results_path"] if candidate_measurement else None,
        "candidate_eval_blockers": candidate_eval_blockers,
        "param_tuning_result_path": str(out / "tuning" / "tuning_validation_result.json") if tuning_result else None,
        "preprocess_report": prep_report,
    }
    (baseline / "lineage.json").write_text(json.dumps(lineage, ensure_ascii=False, indent=2), encoding="utf-8")

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
        "param_tuning_result": tuning_result,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)

    p.add_argument("--input", required=True)
    p.add_argument("--queries", required=True)
    p.add_argument("--queries-encoding")
    p.add_argument("--text-col", required=True)
    p.add_argument("--file-col")
    p.add_argument("--page-col")

    p.add_argument("--config-dir", required=True)
    p.add_argument("--output-dir", required=True)

    p.add_argument("--base-dir", default=r"C:\project\document_viewer\_data_assets")
    p.add_argument("--craw-name", default="my_project")
    p.add_argument("--make-ce-script", default="make_ce_v1.py")
    p.add_argument("--python-exe")
    p.add_argument("--n-clusters", type=int, default=0)

    p.add_argument("--vector-text-col", default="text")
    p.add_argument("--use-mean-tfidf", action="store_true")
    p.add_argument("--allow-dense-mean", action="store_true")

    p.add_argument("--skip-space", action="store_true")
    p.add_argument("--skip-vectors", action="store_true")

    p.add_argument("--max-candidates", type=int, default=20)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--apply", action="store_true")
    p.add_argument("--chunk-profile", default="current")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--tokenizer", choices=["regex", *sorted(PRODUCTION_TOKENIZERS)], default="mecab_ipadic_utf8_v102",
                   help="regex is compatibility/test only and is never included in tuning candidates")
    p.add_argument("--dictionary-path")
    p.add_argument("--mecab-options", default="")
    p.add_argument("--morphology-profile", choices=["noun_only", "content_surface", "content_lemma", "surface_plus_lemma"], default="content_lemma")
    p.add_argument("--enable-param-tuning", action="store_true")
    p.add_argument("--tuning-profile", choices=["minimal"], default="minimal")
    p.add_argument("--max-tuning-candidates", type=int, default=5)
    p.add_argument("--min-evaluated-queries", type=int, default=1)

    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.dry_run:
        args.apply = False
    print(json.dumps(run_auto_improve(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

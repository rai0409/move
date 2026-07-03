#!/usr/bin/env python3
"""Run multi-cycle auto-improvement using existing make_ce_v1 backend."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from apply_lsa_list_updates import apply_updates, parse_args as parse_apply_args
from auto_improve_lsa_lists import parse_args as parse_auto_args, run_auto_improve


def copy_config(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    if src.exists():
        shutil.copytree(src, dst)
    else:
        dst.mkdir(parents=True, exist_ok=True)


def backup_and_replace_config(src: Path, dst: Path, backup_dir: Path) -> None:
    backup_dir.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        shutil.copytree(dst, backup_dir / "config_backup", dirs_exist_ok=True)

    dst.mkdir(parents=True, exist_ok=True)

    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


def run_one_cycle_pipeline(args: argparse.Namespace, config_dir: Path, output_dir: Path) -> dict[str, Any]:
    argv = [
        "--input",
        args.input,
        "--queries",
        args.queries,
        "--text-col",
        args.text_col,
        "--config-dir",
        str(config_dir),
        "--output-dir",
        str(output_dir),
        "--max-candidates",
        str(args.max_candidates_per_cycle),
        "--dry-run",
        "--base-dir",
        args.base_dir,
        "--craw-name",
        args.craw_name,
        "--make-ce-script",
        args.make_ce_script,
        "--n-clusters",
        str(args.n_clusters),
        "--top-k",
        str(args.top_k),
        "--chunk-profile",
        args.chunk_profile,
        "--vector-text-col",
        args.vector_text_col,
    ]

    if args.file_col:
        argv.extend(["--file-col", args.file_col])
    if args.page_col:
        argv.extend(["--page-col", args.page_col])
    if args.python_exe:
        argv.extend(["--python-exe", args.python_exe])
    if args.skip_space:
        argv.append("--skip-space")
    if args.skip_vectors:
        argv.append("--skip-vectors")
    if args.use_mean_tfidf:
        argv.append("--use-mean-tfidf")
    if args.allow_dense_mean:
        argv.append("--allow-dense-mean")

    return run_auto_improve(parse_auto_args(argv))


def read_metrics(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "baseline" / "eval" / "metrics.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def selected_approvals(cycle_dir: Path, max_approved: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    actions_path = cycle_dir / "before" / "candidates" / "candidate_actions.csv"
    validation_path = cycle_dir / "before" / "validation" / "candidate_validation.csv"

    actions = pd.read_csv(actions_path, encoding="utf-8-sig") if actions_path.exists() else pd.DataFrame()
    validation = pd.read_csv(validation_path, encoding="utf-8-sig") if validation_path.exists() else pd.DataFrame()

    if actions.empty or validation.empty:
        return actions.head(0), validation.head(0)

    if "candidate_id" not in actions.columns or "candidate_id" not in validation.columns:
        return actions.head(0), validation.head(0)

    merge_cols = [c for c in ["candidate_id", "priority_score", "evidence_count"] if c in actions.columns]
    merged = validation.merge(actions[merge_cols], on="candidate_id", how="left")

    approved = merged[merged["decision"] == "approve"].copy()
    if approved.empty:
        return actions.head(0), validation.head(0)

    approved["score_delta"] = pd.to_numeric(approved.get("score_delta", 0), errors="coerce").fillna(0)
    approved["priority_score"] = pd.to_numeric(approved.get("priority_score", 0), errors="coerce").fillna(0)

    approved = approved.sort_values(["score_delta", "priority_score"], ascending=[False, False]).head(max_approved)
    selected_ids = set(approved["candidate_id"].astype(str))

    return (
        actions[actions["candidate_id"].astype(str).isin(selected_ids)].copy(),
        validation[validation["candidate_id"].astype(str).isin(selected_ids)].copy(),
    )


def write_selected_files(actions: pd.DataFrame, validation: pd.DataFrame, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    actions_path = out_dir / "selected_candidate_actions.csv"
    validation_path = out_dir / "selected_candidate_validation.csv"

    actions.to_csv(actions_path, index=False, encoding="utf-8-sig")
    validation.to_csv(validation_path, index=False, encoding="utf-8-sig")

    return actions_path, validation_path


def apply_selected_to_temp(
    config_dir: Path,
    actions: pd.DataFrame,
    validation: pd.DataFrame,
    temp_config: Path,
    cycle_dir: Path,
) -> None:
    copy_config(config_dir, temp_config)

    actions_path, validation_path = write_selected_files(actions, validation, cycle_dir / "selected")

    apply_updates(
        parse_apply_args(
            [
                "--config-dir",
                str(temp_config),
                "--candidate-validation",
                str(validation_path),
                "--candidate-actions",
                str(actions_path),
                "--backup-dir",
                str(cycle_dir / "temp_config_backup"),
                "--apply",
            ]
        )
    )


def run_cycles(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    working_config = out / "working_config"
    copy_config(Path(args.config_dir), working_config)

    cycle_rows: list[dict[str, Any]] = []
    approved_history: list[pd.DataFrame] = []
    rejected_history: list[pd.DataFrame] = []

    best_run_dir: Path | None = None
    best_score = 0.0
    accepted = 0
    rolled_back = 0
    stopped = 0

    for cycle in range(1, args.cycles + 1):
        cycle_dir = out / f"cycle_{cycle:03d}"
        snapshot = cycle_dir / "config_snapshot"
        copy_config(working_config, snapshot)

        before_run = cycle_dir / "before"
        before = run_one_cycle_pipeline(args, working_config, before_run)
        before_metrics = read_metrics(before_run)
        score_before = float(before["baseline_score"])

        best_run_dir = before_run
        best_score = score_before

        all_validation_path = before_run / "validation" / "candidate_validation.csv"
        if all_validation_path.exists():
            all_validation = pd.read_csv(all_validation_path, encoding="utf-8-sig")
            rejected_history.append(all_validation[all_validation["decision"] != "approve"].copy())

        actions, validation = selected_approvals(cycle_dir, args.max_approved_per_cycle)

        if actions.empty or validation.empty:
            cycle_rows.append(
                {
                    "cycle": cycle,
                    "decision": "stopped_no_approved",
                    "score_before": score_before,
                    "score_after": score_before,
                    "score_delta": 0.0,
                    "recall5_before": before_metrics.get("recall_at_5"),
                    "recall5_after": before_metrics.get("recall_at_5"),
                    "approved_applied": 0,
                    "reason": "no approved candidates",
                }
            )
            stopped += 1
            if args.stop_if_no_approved:
                break
            continue

        approved_history.append(validation.copy())

        temp_config = cycle_dir / "temp_config"
        apply_selected_to_temp(working_config, actions, validation, temp_config, cycle_dir)

        after_run = cycle_dir / "after"
        after = run_one_cycle_pipeline(args, temp_config, after_run)
        after_metrics = read_metrics(after_run)

        score_after = float(after["baseline_score"])
        score_delta = score_after - score_before

        recall5_before = float(before_metrics.get("recall_at_5") or 0)
        recall5_after = float(after_metrics.get("recall_at_5") or 0)
        recall_drop = args.rollback_if_score_drops and recall5_after < recall5_before

        if score_after >= score_before + args.min_score_delta and not recall_drop:
            copy_config(temp_config, working_config)
            decision = "accepted"
            accepted += 1
            best_run_dir = after_run
            best_score = score_after
        else:
            copy_config(snapshot, working_config)
            decision = "rolled_back"
            rolled_back += 1

        cycle_rows.append(
            {
                "cycle": cycle,
                "decision": decision,
                "score_before": score_before,
                "score_after": score_after,
                "score_delta": score_delta,
                "recall5_before": recall5_before,
                "recall5_after": recall5_after,
                "approved_applied": len(validation),
                "reason": "accepted score improvement"
                if decision == "accepted"
                else "score improvement below threshold or recall_at_5 dropped",
            }
        )

        if decision == "rolled_back" or score_delta < args.min_score_delta:
            break

    final_dir = out / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(cycle_rows).to_csv(out / "cycle_summary.csv", index=False, encoding="utf-8-sig")

    if approved_history:
        pd.concat(approved_history, ignore_index=True).to_csv(out / "approved_history.csv", index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame().to_csv(out / "approved_history.csv", index=False, encoding="utf-8-sig")

    if rejected_history:
        pd.concat(rejected_history, ignore_index=True).to_csv(out / "rejected_history.csv", index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame().to_csv(out / "rejected_history.csv", index=False, encoding="utf-8-sig")

    copy_config(working_config, final_dir / "best_config")

    final_metrics = read_metrics(best_run_dir) if best_run_dir else {}
    final_metrics["composite_score"] = best_score
    final_metrics["backend"] = "make_ce"
    final_metrics["craw_name"] = args.craw_name

    (final_dir / "final_metrics.json").write_text(
        json.dumps(final_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if best_run_dir and (best_run_dir / "baseline" / "lsa_ready.csv").exists():
        shutil.copy2(best_run_dir / "baseline" / "lsa_ready.csv", final_dir / "best_lsa_ready.csv")

    (final_dir / "recommendations.md").write_text(
        "\n".join(
            [
                "# LSA Multi-Cycle Recommendations",
                "",
                "- Backend: make_ce_v1.py",
                f"- craw_name: {args.craw_name}",
                f"- Final score: {best_score:.4f}",
                f"- Accepted cycles: {accepted}",
                f"- Rolled back cycles: {rolled_back}",
                f"- Stopped cycles: {stopped}",
                "",
                "Only accepted config changes are present in `best_config/`.",
            ]
        ),
        encoding="utf-8",
    )

    if args.apply:
        backup_and_replace_config(final_dir / "best_config", Path(args.config_dir), out / "original_config_backup")

    return {
        "output_dir": str(out),
        "backend": "make_ce",
        "craw_name": args.craw_name,
        "final_score": best_score,
        "accepted": accepted,
        "rolled_back": rolled_back,
        "stopped": stopped,
        "cycles_run": len(cycle_rows),
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
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--chunk-profile", default="none")
    p.add_argument("--vector-text-col", default="auto")

    p.add_argument("--use-mean-tfidf", action="store_true")
    p.add_argument("--allow-dense-mean", action="store_true")

    p.add_argument("--skip-space", action="store_true")
    p.add_argument("--skip-vectors", action="store_true")

    p.add_argument("--cycles", type=int, default=5)
    p.add_argument("--max-candidates-per-cycle", type=int, default=20)
    p.add_argument("--max-approved-per-cycle", type=int, default=5)
    p.add_argument("--min-score-delta", type=float, default=0.01)
    p.add_argument("--stop-if-no-approved", action="store_true")
    p.add_argument("--rollback-if-score-drops", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--apply", action="store_true")

    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.dry_run:
        args.apply = False
    print(json.dumps(run_cycles(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
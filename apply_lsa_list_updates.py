#!/usr/bin/env python3
"""Apply approved LSA candidate actions to config Excel files."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from mine_lsa_improvement_candidates import SCHEMAS, ensure_config_xlsx


def backup_configs(config_dir: Path, backup_dir: Path) -> None:
    backup_dir.mkdir(parents=True, exist_ok=True)
    for filename in SCHEMAS:
        src = config_dir / filename
        if src.exists():
            shutil.copy2(src, backup_dir / filename)


def append_unique(path: Path, row: dict[str, Any], key_cols: list[str]) -> None:
    cols = SCHEMAS[path.name]
    df = pd.read_excel(path) if path.exists() else pd.DataFrame(columns=cols)
    for col in cols:
        if col not in df.columns:
            df[col] = ""
    mask = pd.Series([False] * len(df))
    if len(df):
        mask = pd.Series([True] * len(df))
        for col in key_cols:
            mask &= df[col].astype(str) == str(row.get(col, ""))
    if len(df) and mask.any():
        for col, value in row.items():
            if col in df.columns:
                df.loc[mask, col] = value
    else:
        df = pd.concat([df, pd.DataFrame([{col: row.get(col, "") for col in cols}])], ignore_index=True)
    df[cols].to_excel(path, index=False)


def action_to_row(action: pd.Series, validation: pd.Series) -> tuple[str, dict[str, Any], list[str]]:
    common = {
        "enabled": True,
        "priority": action.get("priority_score", ""),
        "source": "auto_improve_lsa",
        "evidence_count": action.get("evidence_count", ""),
        "status": "approved",
        "note": action.get("note", ""),
    }
    if action["action"] == "replace":
        return (
            "df_replace.xlsx",
            {
                **common,
                "pattern": action.get("pattern", action.get("term", "")),
                "replacement": action.get("replacement", ""),
                "match_type": "literal",
                "stage": "preprocess",
                "improved_count": validation.get("improved_count", 0),
                "degraded_count": validation.get("degraded_count", 0),
                "last_score_delta": validation.get("score_delta", 0),
            },
            ["pattern", "replacement"],
        )
    if action["action"] == "keep":
        return ("list_keep.xlsx", {**common, "term": action.get("term", ""), "term_type": "technical"}, ["term"])
    if action["action"] == "stopword":
        return (
            "list_stopword.xlsx",
            {
                **common,
                "term": action.get("term", ""),
                "frequency": action.get("evidence_count", ""),
                "doc_freq_ratio": "",
                "total_miss_count": 0,
                "mean_rank_when_degraded": "",
                "mean_rank_baseline": "",
                "rank_delta": "",
            },
            ["term"],
        )
    return (
        "list_synonym.xlsx",
        {
            **common,
            "canonical": action.get("canonical", ""),
            "variant": action.get("variant", action.get("term", "")),
            "direction": action.get("direction", "query_to_doc"),
            "improved_count": validation.get("improved_count", 0),
            "degraded_count": validation.get("degraded_count", 0),
            "last_score_delta": validation.get("score_delta", 0),
        },
        ["canonical", "variant", "direction"],
    )


def apply_updates(args: argparse.Namespace) -> dict[str, Any]:
    config_dir = Path(args.config_dir)
    ensure_config_xlsx(config_dir)
    actions = pd.read_csv(args.candidate_actions, encoding="utf-8-sig")
    validation = pd.read_csv(args.candidate_validation, encoding="utf-8-sig")
    approved = validation[validation["decision"] == "approve"]
    if not args.apply:
        return {"dry_run": True, "approved_to_apply": int(len(approved)), "updated": 0}
    backup_configs(config_dir, Path(args.backup_dir))
    by_id = {str(r["candidate_id"]): r for _, r in actions.iterrows()}
    updated = 0
    for _, val in approved.iterrows():
        action = by_id.get(str(val["candidate_id"]))
        if action is None:
            continue
        filename, row, keys = action_to_row(action, val)
        append_unique(config_dir / filename, row, keys)
        updated += 1
    return {"dry_run": False, "approved_to_apply": int(len(approved)), "updated": updated}


def apply_candidate_actions_for_eval(
    config_dir: Path,
    candidate_actions: str | Path,
    *,
    max_candidates: int,
) -> dict[str, Any]:
    """Apply raw candidate actions to an isolated candidate config for A/B evaluation.

    This intentionally does not update the user's source config and does not imply
    approval. It only materializes a candidate config so retrieval can be measured.
    """
    ensure_config_xlsx(config_dir)
    actions = pd.read_csv(candidate_actions, encoding="utf-8-sig").head(max_candidates)
    updated = 0
    empty_validation = pd.Series(dtype=object)
    for _, action in actions.iterrows():
        filename, row, keys = action_to_row(action, empty_validation)
        append_unique(config_dir / filename, row, keys)
        updated += 1
    return {"candidate_config_dir": str(config_dir), "candidate_actions_applied": int(updated)}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config-dir", required=True)
    p.add_argument("--candidate-validation", required=True)
    p.add_argument("--candidate-actions", required=True)
    p.add_argument("--backup-dir", required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--apply", action="store_true")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.dry_run:
        args.apply = False
    print(json.dumps(apply_updates(args), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

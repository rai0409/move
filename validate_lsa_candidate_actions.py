#!/usr/bin/env python3
"""Validate LSA candidate actions against baseline retrieval metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from lsa_preprocess_and_chunk import RISKY_KEEP_WORDS


def score(metrics: dict[str, Any]) -> float:
    # Composite score favors top ranks, but includes recall@20 for tuning runs.
    return (
        float(metrics.get("recall_at_1") or 0) * 35
        + float(metrics.get("recall_at_5") or 0) * 25
        + float(metrics.get("recall_at_20") or 0) * 20
        + float(metrics.get("mrr") or 0) * 20
    )


METRIC_KEYS = ["recall_at_1", "recall_at_3", "recall_at_5", "recall_at_10", "recall_at_20", "mrr", "mean_rank", "no_hit_query_count"]
RECOMMENDED_MEASURED_METRICS = ["recall_at_1", "recall_at_5", "recall_at_20", "mrr"]


def numeric_metric(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except Exception:
        return None


def load_json_if_exists(path_value: str | None) -> dict[str, Any] | None:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def measurable_metrics(metrics: dict[str, Any] | None) -> bool:
    if not metrics:
        return False
    return str(metrics.get("validation_mode", "")).startswith("measured") and all(
        numeric_metric(metrics.get(key)) is not None for key in RECOMMENDED_MEASURED_METRICS
    )


def missing_metric_pairs(before: dict[str, Any] | None, after: dict[str, Any] | None) -> list[str]:
    missing: list[str] = []
    before = before or {}
    after = after or {}
    for key in RECOMMENDED_MEASURED_METRICS:
        if numeric_metric(before.get(key)) is None or numeric_metric(after.get(key)) is None:
            missing.append(key)
    return missing


def metric_delta(before: dict[str, Any] | None, after: dict[str, Any] | None) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    before = before or {}
    after = after or {}
    for key in METRIC_KEYS:
        b = numeric_metric(before.get(key))
        a = numeric_metric(after.get(key))
        out[key] = None if b is None or a is None else a - b
    out["composite_score"] = score(after) - score(before)
    return out


def load_results_proxy(path_value: str | None) -> dict[str, Any] | None:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.exists():
        return None
    df = pd.read_csv(path, encoding="utf-8-sig")
    if df.empty or "query" not in df.columns:
        return {"available": False, "reason": "empty or missing query column"}
    if "rank" in df.columns:
        ranked = df.copy()
        ranked["rank"] = pd.to_numeric(ranked["rank"], errors="coerce")
        top = ranked.sort_values("rank").groupby("query", as_index=False).head(1)
    else:
        top = df.groupby("query", as_index=False).head(1)
    proxy: dict[str, Any] = {
        "available": True,
        "n_queries": int(top["query"].nunique()),
    }
    for col in ["score", "jaccard"]:
        if col in top.columns:
            values = pd.to_numeric(top[col], errors="coerce").dropna()
            proxy[f"mean_top1_{col}"] = float(values.mean()) if len(values) else None
    return proxy


def proxy_delta(before: dict[str, Any] | None, after: dict[str, Any] | None) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    before = before or {}
    after = after or {}
    for key in ["mean_top1_score", "mean_top1_jaccard"]:
        b = numeric_metric(before.get(key))
        a = numeric_metric(after.get(key))
        out[key] = None if b is None or a is None else a - b
    return out


def first_hit_ranks(path_value: str | None) -> dict[str, int | None]:
    if not path_value:
        return {}
    path = Path(path_value)
    if not path.exists():
        return {}
    df = pd.read_csv(path, encoding="utf-8-sig")
    if df.empty or "query" not in df.columns:
        return {}
    if "best_correct_rank" in df.columns:
        keys = df["query_index"].astype(str) + ":" + df["query"].astype(str) if "query_index" in df.columns else df["query"].astype(str)
        values = pd.to_numeric(df["best_correct_rank"], errors="coerce")
        return {key: None if pd.isna(rank) else int(rank) for key, rank in zip(keys, values)}
    if "rank" not in df.columns or "is_hit" not in df.columns:
        return {}
    df = df.copy()
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    ranks: dict[str, int | None] = {}
    for query, group in df.groupby("query"):
        hits = group[group["is_hit"] == True].sort_values("rank")  # noqa: E712
        ranks[str(query)] = None if hits.empty or pd.isna(hits.iloc[0]["rank"]) else int(hits.iloc[0]["rank"])
    return ranks


def query_rank_deltas(before_results: str | None, after_results: str | None) -> dict[str, Any]:
    before, after = first_hit_ranks(before_results), first_hit_ranks(after_results)
    improved: list[dict[str, Any]] = []
    regressed: list[dict[str, Any]] = []
    unchanged: list[dict[str, Any]] = []
    for query in sorted(set(before) & set(after)):
        b, a = before[query], after[query]
        b_score, a_score = (21 if b is None else b), (21 if a is None else a)
        row = {"query": query, "before_rank": b, "after_rank": a, "rank_delta": a_score - b_score}
        if a_score < b_score:
            improved.append(row)
        elif a_score > b_score:
            regressed.append(row)
        else:
            unchanged.append(row)
    return {
        "improved_queries": improved, "regressed_queries": regressed, "unchanged_queries": unchanged,
        "improved_query_count": len(improved), "regressed_query_count": len(regressed),
        "unchanged_query_count": len(unchanged),
        "worst_rank_drop": max((r["rank_delta"] for r in regressed), default=0),
        "best_rank_gain": max((-r["rank_delta"] for r in improved), default=0),
    }


def critical_query_regressions(
    before_results: str | None,
    after_results: str | None,
    *,
    max_rank_drop: int,
) -> list[dict[str, Any]]:
    before = first_hit_ranks(before_results)
    after = first_hit_ranks(after_results)
    regressions: list[dict[str, Any]] = []
    for query, before_rank in before.items():
        after_rank = after.get(query)
        if before_rank is None:
            continue
        if after_rank is None:
            regressions.append({"query": query, "before_rank": before_rank, "after_rank": None, "reason": "lost_hit"})
        elif after_rank - before_rank >= max_rank_drop:
            regressions.append(
                {
                    "query": query,
                    "before_rank": before_rank,
                    "after_rank": after_rank,
                    "reason": f"rank_drop_ge_{max_rank_drop}",
                }
            )
    return regressions


def static_decide(row: pd.Series, keep_terms: set[str]) -> tuple[str, str]:
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
            return "static_pass", "high-priority technical keep candidate"
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


def measured_decision(
    before_metrics: dict[str, Any] | None,
    after_metrics: dict[str, Any] | None,
    *,
    min_score_delta: float,
    rollback_if_score_drops: bool,
    critical_regressions: list[dict[str, Any]],
    min_evaluated_queries: int,
) -> tuple[str, str, dict[str, float | None]]:
    deltas = metric_delta(before_metrics, after_metrics)
    before_score = score(before_metrics or {})
    after_score = score(after_metrics or {})
    recall1_delta = deltas.get("recall_at_1")
    recall5_delta = deltas.get("recall_at_5")
    recall20_delta = deltas.get("recall_at_20")
    mrr_delta = deltas.get("mrr")
    if critical_regressions:
        return "reject", f"critical query regression detected: {len(critical_regressions)} queries", deltas
    before_count = int((before_metrics or {}).get("evaluated_query_count") or 0)
    after_count = int((after_metrics or {}).get("evaluated_query_count") or 0)
    if min(before_count, after_count) < min_evaluated_queries or before_count != after_count:
        return "insufficient_evidence", f"evaluated query count invalid: baseline={before_count}, candidate={after_count}, minimum={min_evaluated_queries}", deltas
    if rollback_if_score_drops and recall1_delta is not None and recall1_delta < 0:
        return "reject", f"measured recall_at_1 dropped: {recall1_delta:.4f}", deltas
    if rollback_if_score_drops and recall5_delta is not None and recall5_delta < 0:
        return "reject", f"measured recall_at_5 dropped: {recall5_delta:.4f}", deltas
    if rollback_if_score_drops and recall20_delta is not None and recall20_delta < 0:
        return "reject", f"measured recall_at_20 dropped: {recall20_delta:.4f}", deltas
    if rollback_if_score_drops and mrr_delta is not None and mrr_delta < 0:
        return "reject", f"measured mrr dropped: {mrr_delta:.4f}", deltas
    if (recall1_delta or 0) >= min_score_delta or (mrr_delta or 0) >= min_score_delta:
        return "approve", f"measured improvement met threshold: recall@1 delta={recall1_delta or 0:.4f}, mrr delta={mrr_delta or 0:.4f}", deltas
    return "needs_more_evidence", f"no degradation but improvement below threshold {min_score_delta:.4f}", deltas


def build_validation_result(args: argparse.Namespace, blockers: list[str]) -> dict[str, Any]:
    before_metrics = load_json_if_exists(args.baseline_metrics)
    after_metrics = load_json_if_exists(args.candidate_metrics)
    before_proxy = load_results_proxy(args.baseline_results)
    after_proxy = load_results_proxy(args.candidate_results)
    regressions = critical_query_regressions(
        args.baseline_results,
        args.candidate_results,
        max_rank_drop=args.critical_rank_drop,
    )

    has_before_after_metrics = measurable_metrics(before_metrics) and measurable_metrics(after_metrics)
    has_before_after_proxy = bool(before_proxy and before_proxy.get("available") and after_proxy and after_proxy.get("available"))
    missing_metrics = missing_metric_pairs(before_metrics, after_metrics) if before_metrics or after_metrics else []
    rank_comparison = query_rank_deltas(args.baseline_results, args.candidate_results)

    def lineage_for(result_path: str | None) -> dict[str, Any] | None:
        if not result_path:
            return None
        path = Path(result_path).parent / "evaluation_lineage.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    before_lineage, after_lineage = lineage_for(args.baseline_results), lineage_for(args.candidate_results)
    lineage_match = bool(
        before_lineage and after_lineage
        and before_lineage.get("query_csv_fingerprint") == after_lineage.get("query_csv_fingerprint")
        and before_lineage.get("profiles_match") is True and after_lineage.get("profiles_match") is True
    )

    if has_before_after_metrics and lineage_match:
        mode = "measured"
    elif before_metrics or after_metrics or before_proxy or after_proxy:
        mode = "insufficient_evidence"
        if not has_before_after_metrics:
            blockers.append("target-based before/after metrics are not both available")
        if has_before_after_proxy:
            blockers.append("only proxy query-result diagnostics are available; not safe for automatic approval")
        if not lineage_match:
            blockers.append("baseline/candidate query fingerprint or preprocessing profile validation does not match")
    else:
        mode = "static_only"
        blockers.append("no baseline/candidate metrics or result diagnostics were provided")

    return {
        "validation_mode": mode,
        "baseline_config_path": args.baseline_config_dir,
        "candidate_config_path": args.candidate_config_dir,
        "query_path": args.queries,
        "vector_text_col": args.vector_text_col,
        "before_metrics": before_metrics,
        "after_metrics": after_metrics,
        "metric_delta": metric_delta(before_metrics, after_metrics) if before_metrics or after_metrics else {},
        "missing_metrics": missing_metrics,
        "before_proxy_metrics": before_proxy,
        "after_proxy_metrics": after_proxy,
        "proxy_metric_delta": proxy_delta(before_proxy, after_proxy) if before_proxy or after_proxy else {},
        "critical_query_regression": bool(regressions),
        "critical_query_regressions": regressions,
        "query_rank_comparison": rank_comparison,
        "baseline_evaluation_lineage": before_lineage,
        "candidate_evaluation_lineage": after_lineage,
        "evaluation_lineage_match": lineage_match,
        "environment_blockers": blockers,
    }


def validate_candidates(args: argparse.Namespace) -> pd.DataFrame:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    actions = pd.read_csv(args.candidate_actions, encoding="utf-8-sig").head(args.max_candidates)
    keep_path = Path(args.baseline_config_dir) / "list_keep.xlsx"
    keep_terms = set(pd.read_excel(keep_path)["term"].dropna().astype(str)) if keep_path.exists() else set()

    blockers: list[str] = []
    validation_result = build_validation_result(args, blockers)
    before_metrics = validation_result["before_metrics"]
    after_metrics = validation_result["after_metrics"]
    mode = validation_result["validation_mode"]

    measured_row_decision = None
    measured_row_reason = None
    deltas: dict[str, float | None] = {}
    if mode == "measured":
        measured_row_decision, measured_row_reason, deltas = measured_decision(
            before_metrics,
            after_metrics,
            min_score_delta=args.min_score_delta,
            rollback_if_score_drops=args.rollback_if_score_drops,
            critical_regressions=validation_result["critical_query_regressions"],
            min_evaluated_queries=args.min_evaluated_queries,
        )
        validation_result["decision"] = measured_row_decision
        validation_result["reason"] = measured_row_reason
    else:
        validation_result["decision"] = "needs_real_eval" if mode == "static_only" else "insufficient_evidence"
        validation_result["reason"] = "; ".join(blockers)

    rows: list[dict[str, Any]] = []
    for _, row in actions.iterrows():
        static_decision, static_reason = static_decide(row, keep_terms)
        if mode == "measured":
            decision = measured_row_decision or "reject"
            why = measured_row_reason or "measured validation unavailable"
        elif static_decision == "reject":
            decision = "reject"
            why = static_reason
        elif mode == "insufficient_evidence":
            decision = "insufficient_evidence"
            why = f"{static_reason}; measured before/after metrics unavailable"
        else:
            decision = "needs_real_eval"
            why = f"{static_reason}; static check only, no measured retrieval validation"
        rows.append(
            {
                "candidate_id": row.get("candidate_id", ""),
                "action": row.get("action", ""),
                "target_list": row.get("target_list", ""),
                "term": row.get("term", ""),
                "validation_mode": mode,
                "static_decision": static_decision,
                "score_delta": deltas.get("composite_score", 0.0) if mode == "measured" else 0.0,
                "mrr_delta": deltas.get("mrr", 0.0) if mode == "measured" else 0.0,
                "recall5_delta": deltas.get("recall_at_5", 0.0) if mode == "measured" else 0.0,
                "recall1_delta": deltas.get("recall_at_1", 0.0) if mode == "measured" else 0.0,
                "recall20_delta": deltas.get("recall_at_20", 0.0) if mode == "measured" else 0.0,
                "mean_rank_delta": deltas.get("mean_rank") if mode == "measured" else None,
                "no_hit_count_delta": deltas.get("no_hit_query_count") if mode == "measured" else None,
                "improved_count": validation_result["query_rank_comparison"]["improved_query_count"],
                "degraded_count": validation_result["query_rank_comparison"]["regressed_query_count"],
                "unchanged_count": validation_result["query_rank_comparison"]["unchanged_query_count"],
                "worst_rank_drop": validation_result["query_rank_comparison"]["worst_rank_drop"],
                "best_rank_gain": validation_result["query_rank_comparison"]["best_rank_gain"],
                "decision": decision,
                "reason": why,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "candidate_validation.csv", index=False, encoding="utf-8-sig")
    validation_result["candidate_count"] = int(len(df))
    validation_result["decision_counts"] = df["decision"].value_counts().to_dict() if len(df) else {}
    (out_dir / "candidate_validation_result.json").write_text(
        json.dumps(validation_result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        "[validate_lsa] mode="
        f"{mode} baseline_config={args.baseline_config_dir} candidate_config={args.candidate_config_dir} "
        f"queries={args.queries} vector_text_col={args.vector_text_col} output_dir={out_dir}"
    )
    return df


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--queries", required=True)
    p.add_argument("--text-col", required=True)
    p.add_argument("--file-col")
    p.add_argument("--page-col")
    p.add_argument("--baseline-config-dir", required=True)
    p.add_argument("--candidate-config-dir")
    p.add_argument("--candidate-actions", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-candidates", type=int, default=20)
    p.add_argument("--baseline-metrics")
    p.add_argument("--candidate-metrics")
    p.add_argument("--baseline-results")
    p.add_argument("--candidate-results")
    p.add_argument("--vector-text-col", default="text")
    p.add_argument("--min-score-delta", type=float, default=0.01)
    p.add_argument("--min-evaluated-queries", type=int, default=1)
    p.add_argument("--critical-rank-drop", type=int, default=5)
    p.add_argument("--rollback-if-score-drops", dest="rollback_if_score_drops", action="store_true", default=True)
    p.add_argument("--no-rollback-if-score-drops", dest="rollback_if_score_drops", action="store_false")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    df = validate_candidates(parse_args(argv))
    print(
        json.dumps(
            {
                "validated": len(df),
                "approved": int((df["decision"] == "approve").sum()) if len(df) else 0,
                "needs_real_eval": int((df["decision"] == "needs_real_eval").sum()) if len(df) else 0,
                "insufficient_evidence": int((df["decision"] == "insufficient_evidence").sum()) if len(df) else 0,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

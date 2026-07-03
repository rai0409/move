#!/usr/bin/env python3
"""Mine mechanical config-list improvement candidates from retrieval logs."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from lsa_preprocess_and_chunk import RISKY_KEEP_WORDS


SCHEMAS = {
    "df_replace.xlsx": [
        "pattern",
        "replacement",
        "match_type",
        "stage",
        "enabled",
        "priority",
        "source",
        "evidence_count",
        "improved_count",
        "degraded_count",
        "last_score_delta",
        "status",
        "note",
    ],
    "list_keep.xlsx": ["term", "term_type", "enabled", "priority", "source", "evidence_count", "status", "note"],
    "list_stopword.xlsx": [
        "term",
        "enabled",
        "priority",
        "source",
        "frequency",
        "doc_freq_ratio",
        "total_miss_count",
        "mean_rank_when_degraded",
        "mean_rank_baseline",
        "rank_delta",
        "status",
        "note",
    ],
    "list_synonym.xlsx": [
        "canonical",
        "variant",
        "direction",
        "enabled",
        "priority",
        "source",
        "evidence_count",
        "improved_count",
        "degraded_count",
        "last_score_delta",
        "status",
        "note",
    ],
}


ACTION_COLUMNS = [
    "candidate_id",
    "action",
    "target_list",
    "term",
    "pattern",
    "replacement",
    "canonical",
    "variant",
    "direction",
    "reason",
    "priority_score",
    "evidence_count",
    "status",
    "note",
]


def ensure_config_xlsx(config_dir: Path) -> dict[str, pd.DataFrame]:
    config_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, pd.DataFrame] = {}

    for filename, columns in SCHEMAS.items():
        path = config_dir / filename
        if path.exists():
            df = pd.read_excel(path)
            for col in columns:
                if col not in df.columns:
                    df[col] = ""
            df = df[columns]
        else:
            df = pd.DataFrame(columns=columns)
            df.to_excel(path, index=False)
        out[filename] = df

    return out


def technical_term(term: str) -> bool:
    return bool(
        re.search(r"[A-Z]", term)
        or re.search(r"\d", term)
        or re.search(r"[A-Za-z][\u3040-\u30ff\u3400-\u9fff]|[\u3040-\u30ff\u3400-\u9fff][A-Za-z]", term)
    )


def priority(total_miss: int, affected: int, term: str) -> int:
    score = 1
    if total_miss >= 2:
        score += 1
    if affected >= 2:
        score += 1
    if technical_term(term):
        score += 1
    if len(term) >= 6:
        score += 1
    return max(1, min(5, score))


def spaced_variants(term: str) -> set[str]:
    collapsed = re.sub(r"[\s_\-]+", "", term)
    variants = {
        collapsed,
        term.replace("_", " "),
        term.replace("-", " "),
        term.lower(),
        term.upper(),
    }
    return {v for v in variants if v and v != term}


def hit_text_spacing_variants(term: str, hit_texts: Sequence[str]) -> set[str]:
    collapsed = re.sub(r"[\s_\-]+", "", term).lower()
    variants: set[str] = set()

    for text in hit_texts:
        parts = str(text).split()
        for i in range(len(parts)):
            joined = ""
            display: list[str] = []
            for part in parts[i : i + 4]:
                joined += re.sub(r"[\s_\-]+", "", part)
                display.append(part)
                if joined.lower() == collapsed and " ".join(display) != term:
                    variants.add(" ".join(display))
    return variants


def existing_terms(configs: dict[str, pd.DataFrame], filename: str, col: str) -> set[str]:
    df = configs[filename]
    if col not in df:
        return set()
    return set(df[col].dropna().astype(str))


def valid_neighbor_term(term: str, stopwords: set[str]) -> bool:
    if not term or term in stopwords or term in RISKY_KEEP_WORDS:
        return False
    if len(term) <= 1 or term.isdigit():
        return False
    if str(term).lower() in {"nan", "none", "page", "copyright", "confidential"}:
        return False
    return True


def load_term_stats(path_value: str | None) -> pd.DataFrame:
    if not path_value:
        return pd.DataFrame(columns=["term", "frequency", "doc_frequency", "doc_freq_ratio"])

    path = Path(path_value)
    if not path.exists():
        return pd.DataFrame(columns=["term", "frequency", "doc_frequency", "doc_freq_ratio"])

    df = pd.read_csv(path, encoding="utf-8-sig")
    for col in ["term", "frequency", "doc_frequency", "doc_freq_ratio"]:
        if col not in df.columns:
            df[col] = 0 if col != "term" else ""
    return df[["term", "frequency", "doc_frequency", "doc_freq_ratio"]]


def load_term_neighbors(args: argparse.Namespace) -> tuple[pd.DataFrame | None, np.ndarray | None, dict[str, int]]:
    if not args.term_vectors or not args.term_metadata:
        return None, None, {}

    vector_path = Path(args.term_vectors)
    metadata_path = Path(args.term_metadata)

    if not vector_path.exists() or not metadata_path.exists():
        return None, None, {}

    metadata = pd.read_csv(metadata_path, encoding="utf-8-sig")
    vectors = np.load(vector_path)
    index = {str(row["term"]): int(row["term_index"]) for _, row in metadata.iterrows()}
    return metadata, vectors, index


def append_lsa_neighbor_synonyms(
    actions: list[dict[str, Any]],
    cid_start: int,
    args: argparse.Namespace,
    results: pd.DataFrame,
    miss_counter: Counter[str],
    affected_queries: defaultdict[str, set[str]],
    stopwords: set[str],
) -> int:
    metadata, vectors, index = load_term_neighbors(args)
    if metadata is None or vectors is None or not index:
        return cid_start

    cid = cid_start
    hit_text_by_term: defaultdict[str, str] = defaultdict(str)

    for _, row in results.iterrows():
        hit_text = str(row.get("hit_text", ""))
        for term in str(row.get("missing_terms", "")).split():
            hit_text_by_term[term] += " " + hit_text

    for term, miss_count in miss_counter.most_common():
        if term not in index:
            continue

        base_idx = index[term]
        sims = vectors @ vectors[base_idx]
        order = np.argsort(-sims)
        emitted = 0

        for neighbor_idx in order:
            neighbor_idx = int(neighbor_idx)
            if neighbor_idx == base_idx:
                continue

            sim = float(sims[neighbor_idx])
            if sim < args.min_term_similarity:
                break

            neighbor = str(metadata.iloc[neighbor_idx]["term"])
            if not valid_neighbor_term(neighbor, stopwords):
                continue

            target_boost = neighbor in hit_text_by_term[term]
            pr = priority(miss_count + (1 if target_boost else 0), len(affected_queries[term]), term)

            actions.append(
                {
                    "candidate_id": f"C{cid:04d}",
                    "action": "synonym",
                    "target_list": "list_synonym.xlsx",
                    "term": term,
                    "pattern": "",
                    "replacement": "",
                    "canonical": neighbor,
                    "variant": term,
                    "direction": "query_to_doc",
                    "reason": "lsa_neighbor_missing_term",
                    "priority_score": max(pr, 3 if target_boost else pr),
                    "evidence_count": int(miss_count + (1 if target_boost else 0)),
                    "status": "candidate",
                    "note": f"term_similarity={sim:.4f}; target_hit_text_match={target_boost}",
                }
            )
            cid += 1
            emitted += 1

            if emitted >= args.max_term_neighbors:
                break

    return cid


def mine_candidates(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = ensure_config_xlsx(Path(args.config_dir))
    stopwords = existing_terms(configs, "list_stopword.xlsx", "term")
    keep_terms = existing_terms(configs, "list_keep.xlsx", "term")

    term_stats = load_term_stats(args.term_stats)
    term_stat_by_term = {str(r["term"]): r for _, r in term_stats.iterrows()}

    results = pd.read_csv(args.query_results, encoding="utf-8-sig")

    if "missing_terms" not in results.columns:
        results["missing_terms"] = ""
    if "hit_text" not in results.columns:
        results["hit_text"] = ""
    if "query" not in results.columns:
        results["query"] = ""

    miss_counter: Counter[str] = Counter()
    affected_queries: defaultdict[str, set[str]] = defaultdict(set)
    target_groups: defaultdict[tuple[str, str, str], list[str]] = defaultdict(list)

    for _, row in results.iterrows():
        for term in str(row.get("missing_terms", "")).split():
            miss_counter[term] += 1
            affected_queries[term].add(str(row.get("query", "")))
            target_groups[(term, str(row.get("target_doc", "")), str(row.get("page", "")))].append(str(row.get("hit_text", "")))

    actions: list[dict[str, Any]] = []
    cid = 1

    for term, count in miss_counter.most_common():
        if not term or term.lower() == "nan":
            continue

        affected = len(affected_queries[term])
        pr = priority(count, affected, term)

        hit_texts_for_term = [
            str(x)
            for x in results.loc[
                results["missing_terms"].fillna("").astype(str).str.contains(re.escape(term), regex=True),
                "hit_text",
            ]
        ]

        for variant in list(hit_text_spacing_variants(term, hit_texts_for_term)) + list(spaced_variants(term)):
            if variant in term_stat_by_term or any(variant in str(x) for x in results.get("hit_text", [])):
                reason = "space_variant" if re.sub(r"\s+", "", variant) == re.sub(r"\s+", "", term) else "case_variant"
                actions.append(
                    {
                        "candidate_id": f"C{cid:04d}",
                        "action": "replace",
                        "target_list": "df_replace.xlsx",
                        "term": term,
                        "pattern": term,
                        "replacement": variant,
                        "canonical": "",
                        "variant": "",
                        "direction": "",
                        "reason": reason,
                        "priority_score": pr,
                        "evidence_count": count,
                        "status": "candidate",
                        "note": "mechanical normalization variant",
                    }
                )
                cid += 1
                break

        if technical_term(term) and term not in stopwords:
            actions.append(
                {
                    "candidate_id": f"C{cid:04d}",
                    "action": "keep",
                    "target_list": "list_keep.xlsx",
                    "term": term,
                    "pattern": "",
                    "replacement": "",
                    "canonical": "",
                    "variant": "",
                    "direction": "",
                    "reason": "technical_missing_term",
                    "priority_score": pr,
                    "evidence_count": count,
                    "status": "candidate",
                    "note": "technical-looking query term repeatedly missing",
                }
            )
            cid += 1

        for (mterm, doc, page), hit_texts in target_groups.items():
            if mterm == term and doc and page and len(hit_texts) >= 2:
                hit_terms = [t for text in hit_texts for t in str(text).split() if len(t) >= 2 and t != term]
                if hit_terms:
                    canonical = Counter(hit_terms).most_common(1)[0][0]
                    actions.append(
                        {
                            "candidate_id": f"C{cid:04d}",
                            "action": "synonym",
                            "target_list": "list_synonym.xlsx",
                            "term": term,
                            "pattern": "",
                            "replacement": "",
                            "canonical": canonical,
                            "variant": term,
                            "direction": "query_to_doc",
                            "reason": "repeated_same_target",
                            "priority_score": pr,
                            "evidence_count": len(hit_texts),
                            "status": "candidate",
                            "note": "candidate synonym from repeated target miss",
                        }
                    )
                    cid += 1

    total_docs = max(float(term_stats["doc_frequency"].max()) if not term_stats.empty else 1.0, 1.0)

    for _, row in term_stats.iterrows():
        term = str(row.get("term", ""))
        ratio = float(row.get("doc_freq_ratio", row.get("doc_frequency", 0) / total_docs))
        if ratio >= 0.6 and miss_counter[term] == 0 and term not in keep_terms and term not in RISKY_KEEP_WORDS:
            actions.append(
                {
                    "candidate_id": f"C{cid:04d}",
                    "action": "stopword",
                    "target_list": "list_stopword.xlsx",
                    "term": term,
                    "pattern": "",
                    "replacement": "",
                    "canonical": "",
                    "variant": "",
                    "direction": "",
                    "reason": "high_document_frequency_low_missing",
                    "priority_score": 2,
                    "evidence_count": int(row.get("frequency", 0)),
                    "status": "candidate",
                    "note": "must be validated before apply",
                }
            )
            cid += 1

    cid = append_lsa_neighbor_synonyms(actions, cid, args, results, miss_counter, affected_queries, stopwords)

    actions_df = pd.DataFrame(actions, columns=ACTION_COLUMNS)

    degradation = []
    all_terms = sorted(set(miss_counter) | set(term_stats.get("term", pd.Series(dtype=str)).astype(str)))

    for term in all_terms:
        stats = term_stat_by_term.get(term, {})
        miss = miss_counter[term]
        affected = len(affected_queries[term])

        get_stat = getattr(stats, "get", lambda *_: 0)
        ratio = float(get_stat("doc_freq_ratio", 0) or 0)

        rec_action = "keep" if technical_term(term) and miss else ("stopword" if miss == 0 and ratio >= 0.6 else "do_nothing")

        degradation.append(
            {
                "term": term,
                "term_type": "technical" if technical_term(term) else "general",
                "frequency": int(get_stat("frequency", 0) or 0),
                "doc_frequency": int(get_stat("doc_frequency", 0) or 0),
                "query_frequency": int(miss),
                "total_miss_count": int(miss),
                "affected_query_count": int(affected),
                "mean_rank_when_missing": 0,
                "mean_rank_baseline": 0,
                "mean_rank_candidate": 0,
                "rank_delta": 0,
                "recall5_delta": 0,
                "mrr_delta": 0,
                "improved_count": 0,
                "degraded_count": 0,
                "priority_score": priority(miss, affected, term) if miss else 1,
                "recommended_action": rec_action,
                "recommended_list": "list_keep.xlsx" if rec_action == "keep" else ("list_stopword.xlsx" if rec_action == "stopword" else ""),
                "status": "candidate" if rec_action != "do_nothing" else "observed",
            }
        )

    degradation_df = pd.DataFrame(degradation)

    actions_df.to_csv(out_dir / "candidate_actions.csv", index=False, encoding="utf-8-sig")
    degradation_df.to_csv(out_dir / "term_degradation_report.csv", index=False, encoding="utf-8-sig")

    (out_dir / "recommendations.md").write_text(
        f"# LSA Candidate Recommendations\n\n- Candidates: {len(actions_df)}\n- Missing terms observed: {len(miss_counter)}\n",
        encoding="utf-8",
    )

    return actions_df, degradation_df


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--query-results", required=True)
    p.add_argument("--term-stats", default=None)
    p.add_argument("--config-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--term-vectors")
    p.add_argument("--term-metadata")
    p.add_argument("--max-term-neighbors", type=int, default=20)
    p.add_argument("--min-term-similarity", type=float, default=0.35)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    actions, _ = mine_candidates(parse_args(argv))
    print(json.dumps({"candidates": len(actions)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
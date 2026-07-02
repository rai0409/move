#!/usr/bin/env python3
"""Evaluate query retrieval against a built TF-IDF + TruncatedSVD(LSA) space."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize


TARGET_COLS = {"target_doc", "page", "expected_text_contains"}


def has_targets(df: pd.DataFrame) -> bool:
    return any(col in df.columns and df[col].notna().any() for col in TARGET_COLS)


def query_vector(query: str, vectorizer: Any, svd: Any) -> np.ndarray:
    tfidf = vectorizer.transform([query])
    if svd is not None:
        vec = svd.transform(tfidf)
    else:
        vec = tfidf.toarray()
    return normalize(vec, norm="l2")[0]


def token_jaccard(a: str, b: str) -> tuple[float, str, str]:
    qa = set(str(a).split())
    hb = set(str(b).split())
    inter = qa & hb
    union = qa | hb
    return (len(inter) / max(len(union), 1), " ".join(sorted(qa - hb)), " ".join(sorted(hb - qa)))


def is_hit(row: pd.Series, hit: pd.Series) -> tuple[bool, bool, bool]:
    doc_hit = True
    page_hit = True
    text_hit = True
    if "target_doc" in row and str(row.get("target_doc", "")).strip() and str(row["target_doc"]) != "nan":
        doc_hit = str(row["target_doc"]) == str(hit.get("file_name_out", ""))
    if "page" in row and str(row.get("page", "")).strip() and str(row["page"]) != "nan":
        page_hit = str(row["page"]) == str(hit.get("page_out", ""))
    if "expected_text_contains" in row and str(row.get("expected_text_contains", "")).strip() and str(row["expected_text_contains"]) != "nan":
        text_hit = str(row["expected_text_contains"]) in str(hit.get("chunk_text", ""))
    return doc_hit and page_hit and text_hit, doc_hit, page_hit


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    model_dir = Path(args.model_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vectorizer = joblib.load(model_dir / "tfidf_vectorizer.joblib")
    svd = joblib.load(model_dir / "truncated_svd.joblib")
    vectors = np.load(model_dir / "lsa_vectors.npy")
    metadata = pd.read_csv(model_dir / "metadata.csv", encoding="utf-8-sig")
    queries = pd.read_csv(args.queries, encoding=args.encoding)
    if "query" not in queries.columns:
        raise ValueError("queries CSV must contain query")
    target_mode = has_targets(queries)
    rows: list[dict[str, Any]] = []
    per_query_hits: list[int | None] = []
    for qi, qrow in queries.iterrows():
        query = str(qrow["query"])
        qvec = query_vector(query, vectorizer, svd)
        scores = vectors @ qvec
        order = np.argsort(-scores)[: args.top_k]
        first_hit_rank: int | None = None
        for rank, idx in enumerate(order, start=1):
            hit = metadata.iloc[int(idx)]
            jaccard, missing, extra = token_jaccard(query, hit.get("lsa_tokens_str", ""))
            hit_ok, doc_ok, page_ok = is_hit(qrow, hit) if target_mode else (False, False, False)
            if target_mode and hit_ok and first_hit_rank is None:
                first_hit_rank = rank
            rows.append(
                {
                    "query": query,
                    "hit_text": hit.get("chunk_text", ""),
                    "rank": rank,
                    "score": float(scores[int(idx)]),
                    "n_query_tokens": len(query.split()),
                    "n_hit_tokens": len(str(hit.get("lsa_tokens_str", "")).split()),
                    "jaccard": jaccard,
                    "missing_terms": missing,
                    "extra_terms": extra,
                    "target_doc": qrow.get("target_doc", ""),
                    "page": qrow.get("page", ""),
                    "hit_doc": hit.get("file_name_out", ""),
                    "hit_page": hit.get("page_out", ""),
                    "is_hit": bool(hit_ok),
                    "is_target_doc_hit": bool(doc_ok),
                    "is_page_hit": bool(page_ok),
                }
            )
        per_query_hits.append(first_hit_rank if target_mode else None)
    results = pd.DataFrame(rows)
    results.to_csv(out_dir / "query_results.csv", index=False, encoding="utf-8-sig")
    if target_mode:
        failed = results.groupby("query", as_index=False)["is_hit"].max()
        failed = failed[~failed["is_hit"]]
        failed.to_csv(out_dir / "failed_queries.csv", index=False, encoding="utf-8-sig")
        denom = max(len(per_query_hits), 1)
        metrics = {
            "n_queries": len(per_query_hits),
            "recall_at_1": sum(1 for r in per_query_hits if r is not None and r <= 1) / denom,
            "recall_at_3": sum(1 for r in per_query_hits if r is not None and r <= 3) / denom,
            "recall_at_5": sum(1 for r in per_query_hits if r is not None and r <= 5) / denom,
            "recall_at_10": sum(1 for r in per_query_hits if r is not None and r <= 10) / denom,
            "mrr": sum(0 if r is None else 1 / r for r in per_query_hits) / denom,
        }
    else:
        results.to_csv(out_dir / "public_query_diagnostics.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(columns=results.columns).to_csv(out_dir / "failed_queries.csv", index=False, encoding="utf-8-sig")
        metrics = {"n_queries": len(queries), "recall_at_1": None, "recall_at_3": None, "recall_at_5": None, "recall_at_10": None, "mrr": None}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--queries", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--encoding", default="utf-8-sig")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    print(json.dumps(evaluate(parse_args(argv)), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

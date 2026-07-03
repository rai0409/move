#!/usr/bin/env python3
"""
Evaluate retrieval for:

1. legacy_lsa
   Original generated model format:
     tfidf_vectorizer.joblib
     truncated_svd.joblib
     lsa_vectors.npy
     metadata.csv

2. make_ce_direct
   Existing make_ce_v1.py output format:
     step1/vectorizer.pkl
     step1/df.pkl or step1/df.csv
     step1/overall_mean_tfidf.pkl optional

   Important:
     This backend does NOT require a saved document-vector pkl.
     It rebuilds document TF-IDF at evaluation time from df text columns.

Confirmed make_ce metadata columns:
  file name : name_org
  page      : pageno
  text      : text
  id        : chunk_id
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd

try:
    from scipy import sparse
except Exception:  # pragma: no cover
    sparse = None

from sklearn.preprocessing import normalize


TARGET_COLS = {"target_doc", "page", "expected_text_contains"}


def has_targets(df: pd.DataFrame) -> bool:
    return any(col in df.columns and df[col].notna().any() for col in TARGET_COLS)


def load_pickle_any(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"file not found: {path}")
    try:
        return joblib.load(path)
    except Exception:
        with path.open("rb") as f:
            return pickle.load(f)


def read_metadata(path: Path, fmt: str = "auto", encoding: str = "utf-8-sig") -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"metadata not found: {path}")

    if fmt == "auto":
        if path.suffix.lower() == ".pkl":
            fmt = "pkl"
        elif path.suffix.lower() == ".csv":
            fmt = "csv"
        else:
            raise ValueError(f"cannot infer metadata format: {path}")

    if fmt == "pkl":
        obj = load_pickle_any(path)
        if not isinstance(obj, pd.DataFrame):
            raise TypeError(f"metadata pkl must be pandas DataFrame, got {type(obj)}")
        return obj

    if fmt == "csv":
        return pd.read_csv(path, encoding=encoding)

    raise ValueError(f"unsupported metadata format: {fmt}")


def autodetect_make_ce_paths(model_dir: Path, args: argparse.Namespace) -> dict[str, Path]:
    step1 = model_dir / "step1"
    step2 = model_dir / "step2"

    vectorizer_path = Path(args.vectorizer_path) if args.vectorizer_path else step1 / "vectorizer.pkl"

    if args.metadata_path:
        metadata_path = Path(args.metadata_path)
    elif (step1 / "df.pkl").exists():
        metadata_path = step1 / "df.pkl"
    elif (step1 / "df.csv").exists():
        metadata_path = step1 / "df.csv"
    elif (step2 / "df.pkl").exists():
        metadata_path = step2 / "df.pkl"
    elif (step2 / "df.csv").exists():
        metadata_path = step2 / "df.csv"
    else:
        metadata_path = step1 / "df.pkl"

    mean_path = Path(args.mean_tfidf_path) if args.mean_tfidf_path else step1 / "overall_mean_tfidf.pkl"

    return {
        "vectorizer": vectorizer_path,
        "metadata": metadata_path,
        "mean_tfidf": mean_path,
    }


def choose_vector_text_col(df: pd.DataFrame, requested: str) -> str:
    if requested and requested != "auto":
        if requested not in df.columns:
            raise ValueError(f"vector text column not found: {requested}; available={list(df.columns)}")
        return requested

    for col in ["CHASEN_TEXT_cleaned", "wakati_text", "text"]:
        if col in df.columns:
            return col

    raise ValueError(
        "no usable vector text column found. "
        "Expected one of CHASEN_TEXT_cleaned, wakati_text, text."
    )


def safe_normalize_matrix(x: Any) -> Any:
    if sparse is not None and sparse.issparse(x):
        return normalize(x, norm="l2", axis=1, copy=True)

    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[~np.isfinite(norms)] = 0
    norms[norms == 0] = 1.0
    return arr / norms


def safe_normalize_vector(x: Any) -> np.ndarray:
    if sparse is not None and sparse.issparse(x):
        x = x.toarray()

    arr = np.asarray(x, dtype=float).reshape(1, -1)
    norm = np.linalg.norm(arr)
    if not np.isfinite(norm) or norm == 0:
        norm = 1.0
    return (arr / norm)[0]


def maybe_subtract_mean(x: Any, mean: Any | None, allow_dense: bool) -> Any:
    if mean is None:
        return x

    if sparse is not None and sparse.issparse(mean):
        mean_arr = mean.toarray()
    else:
        mean_arr = np.asarray(mean)

    mean_arr = mean_arr.reshape(1, -1)

    if sparse is not None and sparse.issparse(x):
        if not allow_dense:
            raise RuntimeError(
                "Refusing to subtract dense mean from sparse matrix because it may consume huge memory. "
                "Run without --use-mean-tfidf, or add --allow-dense-mean if you intentionally accept densification."
            )
        return x.toarray() - mean_arr

    return np.asarray(x) - mean_arr


def token_jaccard(a: str, b: str) -> tuple[float, str, str]:
    qa = set(str(a).split())
    hb = set(str(b).split())
    inter = qa & hb
    union = qa | hb
    return (len(inter) / max(len(union), 1), " ".join(sorted(qa - hb)), " ".join(sorted(hb - qa)))


def compare_doc(expected: Any, actual: Any) -> bool:
    if pd.isna(expected):
        return True
    se = str(expected).strip()
    sa = str(actual).strip()
    if not se or se.lower() == "nan":
        return True
    return se == sa


def compare_page(expected: Any, actual: Any) -> bool:
    if pd.isna(expected):
        return True

    se = str(expected).strip()
    sa = str(actual).strip()

    if not se or se.lower() == "nan":
        return True

    try:
        return int(float(se)) == int(float(sa))
    except Exception:
        return se == sa


def is_hit_generic(
    qrow: pd.Series,
    hit: pd.Series,
    *,
    file_col: str,
    page_col: str,
    display_col: str,
) -> tuple[bool, bool, bool]:
    doc_hit = True
    page_hit = True
    text_hit = True

    if "target_doc" in qrow and str(qrow.get("target_doc", "")).strip() and str(qrow["target_doc"]) != "nan":
        doc_hit = compare_doc(qrow["target_doc"], hit.get(file_col, ""))

    if "page" in qrow and str(qrow.get("page", "")).strip() and str(qrow["page"]) != "nan":
        page_hit = compare_page(qrow["page"], hit.get(page_col, ""))

    if (
        "expected_text_contains" in qrow
        and str(qrow.get("expected_text_contains", "")).strip()
        and str(qrow["expected_text_contains"]) != "nan"
    ):
        text_hit = str(qrow["expected_text_contains"]) in str(hit.get(display_col, ""))

    return doc_hit and page_hit and text_hit, doc_hit, page_hit


def build_metrics(results: pd.DataFrame, queries: pd.DataFrame) -> dict[str, Any]:
    hits: list[int | None] = []

    for _, qrow in queries.iterrows():
        q = str(qrow["query"])
        qhits = results[results["query"] == q].copy()
        if qhits.empty:
            hits.append(None)
            continue

        qhits["rank"] = pd.to_numeric(qhits["rank"], errors="coerce")
        matched = qhits[qhits["is_hit"] == True].sort_values("rank")  # noqa: E712

        if matched.empty:
            hits.append(None)
        else:
            hits.append(int(matched.iloc[0]["rank"]))

    denom = max(len(queries), 1)
    return {
        "n_queries": len(queries),
        "recall_at_1": sum(1 for r in hits if r is not None and r <= 1) / denom,
        "recall_at_3": sum(1 for r in hits if r is not None and r <= 3) / denom,
        "recall_at_5": sum(1 for r in hits if r is not None and r <= 5) / denom,
        "recall_at_10": sum(1 for r in hits if r is not None and r <= 10) / denom,
        "mrr": sum(0 if r is None else 1 / r for r in hits) / denom,
    }


def write_outputs(rows: list[dict[str, Any]], queries: pd.DataFrame, out_dir: Path, target_mode: bool) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(rows)
    results.to_csv(out_dir / "query_results.csv", index=False, encoding="utf-8-sig")

    if target_mode:
        failed = results.groupby("query", as_index=False)["is_hit"].max()
        failed = failed[~failed["is_hit"]]
        failed.to_csv(out_dir / "failed_queries.csv", index=False, encoding="utf-8-sig")
        metrics = build_metrics(results, queries)
    else:
        results.to_csv(out_dir / "public_query_diagnostics.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(columns=results.columns).to_csv(out_dir / "failed_queries.csv", index=False, encoding="utf-8-sig")
        metrics = {
            "n_queries": len(queries),
            "recall_at_1": None,
            "recall_at_3": None,
            "recall_at_5": None,
            "recall_at_10": None,
            "mrr": None,
        }

    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def evaluate_legacy_lsa(args: argparse.Namespace) -> dict[str, Any]:
    model_dir = Path(args.model_dir)
    out_dir = Path(args.output_dir)

    vectorizer = joblib.load(model_dir / "tfidf_vectorizer.joblib")
    svd = joblib.load(model_dir / "truncated_svd.joblib")
    vectors = np.load(model_dir / "lsa_vectors.npy")
    metadata = pd.read_csv(model_dir / "metadata.csv", encoding="utf-8-sig")
    queries = pd.read_csv(args.queries, encoding=args.encoding)

    if "query" not in queries.columns:
        raise ValueError("queries CSV must contain query")

    target_mode = has_targets(queries)
    rows: list[dict[str, Any]] = []

    for _, qrow in queries.iterrows():
        query = str(qrow["query"])
        tfidf = vectorizer.transform([query])
        qvec = svd.transform(tfidf) if svd is not None else tfidf.toarray()
        qvec = normalize(qvec, norm="l2")[0]

        scores = vectors @ qvec
        order = np.argsort(-scores)[: args.top_k]

        for rank, idx in enumerate(order, start=1):
            hit = metadata.iloc[int(idx)]
            jaccard, missing, extra = token_jaccard(query, hit.get("lsa_tokens_str", ""))

            hit_ok, doc_ok, page_ok = (
                is_hit_generic(qrow, hit, file_col="file_name_out", page_col="page_out", display_col="chunk_text")
                if target_mode
                else (False, False, False)
            )

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
                    "hit_id": hit.get("chunk_id_out", ""),
                    "is_hit": bool(hit_ok),
                    "is_target_doc_hit": bool(doc_ok),
                    "is_page_hit": bool(page_ok),
                }
            )

    return write_outputs(rows, queries, out_dir, target_mode)


def evaluate_make_ce_direct(args: argparse.Namespace) -> dict[str, Any]:
    if not args.model_dir:
        raise ValueError("--model-dir is required for backend=make_ce_direct")

    model_dir = Path(args.model_dir)
    out_dir = Path(args.output_dir)

    queries = pd.read_csv(args.queries, encoding=args.encoding)
    if "query" not in queries.columns:
        raise ValueError("queries CSV must contain query")

    paths = autodetect_make_ce_paths(model_dir, args)

    if not paths["vectorizer"].exists():
        raise FileNotFoundError(f"vectorizer.pkl not found: {paths['vectorizer']}")
    if not paths["metadata"].exists():
        raise FileNotFoundError(f"df.pkl/df.csv not found: {paths['metadata']}")

    vectorizer = load_pickle_any(paths["vectorizer"])
    metadata = read_metadata(paths["metadata"], args.metadata_format, args.encoding)

    vector_text_col = choose_vector_text_col(metadata, args.vector_text_col)

    required_cols = [args.file_col, args.page_col, args.display_col, args.id_col, vector_text_col]
    missing = [c for c in required_cols if c not in metadata.columns]
    if missing:
        raise ValueError(f"missing metadata columns: {missing}; available={list(metadata.columns)}")

    docs_text = metadata[vector_text_col].fillna("").astype(str).tolist()
    doc_matrix = vectorizer.transform(docs_text)

    mean_tfidf = None
    if args.use_mean_tfidf:
        if not paths["mean_tfidf"].exists():
            raise FileNotFoundError(f"overall_mean_tfidf.pkl not found: {paths['mean_tfidf']}")
        mean_tfidf = load_pickle_any(paths["mean_tfidf"])
        doc_matrix = maybe_subtract_mean(doc_matrix, mean_tfidf, args.allow_dense_mean)

    doc_matrix = safe_normalize_matrix(doc_matrix)

    target_mode = has_targets(queries)
    rows: list[dict[str, Any]] = []

    for _, qrow in queries.iterrows():
        query = str(qrow["query"])

        qvec = vectorizer.transform([query])
        qvec = maybe_subtract_mean(qvec, mean_tfidf, args.allow_dense_mean)
        qvec = safe_normalize_vector(qvec)

        if sparse is not None and sparse.issparse(doc_matrix):
            scores = np.asarray(doc_matrix @ qvec.reshape(-1, 1)).ravel()
        else:
            scores = np.asarray(doc_matrix @ qvec).ravel()

        if not np.all(np.isfinite(scores)):
            scores = np.nan_to_num(scores, nan=-math.inf, posinf=math.inf, neginf=-math.inf)

        order = np.argsort(-scores)[: args.top_k]

        for rank, idx in enumerate(order, start=1):
            hit = metadata.iloc[int(idx)]
            hit_text = hit.get(args.display_col, "")
            vector_text = hit.get(vector_text_col, "")

            jaccard, missing, extra = token_jaccard(query, vector_text)

            hit_ok, doc_ok, page_ok = (
                is_hit_generic(qrow, hit, file_col=args.file_col, page_col=args.page_col, display_col=args.display_col)
                if target_mode
                else (False, False, False)
            )

            rows.append(
                {
                    "query": query,
                    "hit_text": hit_text,
                    "rank": rank,
                    "score": float(scores[int(idx)]),
                    "n_query_tokens": len(query.split()),
                    "n_hit_tokens": len(str(vector_text).split()),
                    "jaccard": jaccard,
                    "missing_terms": missing,
                    "extra_terms": extra,
                    "target_doc": qrow.get("target_doc", ""),
                    "page": qrow.get("page", ""),
                    "hit_doc": hit.get(args.file_col, ""),
                    "hit_page": hit.get(args.page_col, ""),
                    "hit_id": hit.get(args.id_col, ""),
                    "is_hit": bool(hit_ok),
                    "is_target_doc_hit": bool(doc_ok),
                    "is_page_hit": bool(page_ok),
                }
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "backend": "make_ce_direct",
        "model_dir": str(model_dir),
        "vectorizer_path": str(paths["vectorizer"]),
        "metadata_path": str(paths["metadata"]),
        "mean_tfidf_path": str(paths["mean_tfidf"]) if args.use_mean_tfidf else None,
        "use_mean_tfidf": bool(args.use_mean_tfidf),
        "vector_text_col": vector_text_col,
        "file_col": args.file_col,
        "page_col": args.page_col,
        "display_col": args.display_col,
        "id_col": args.id_col,
        "metadata_rows": int(len(metadata)),
        "doc_matrix_shape": list(doc_matrix.shape),
    }
    (out_dir / "evaluate_backend_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    return write_outputs(rows, queries, out_dir, target_mode)


def evaluate_results_csv(args: argparse.Namespace) -> dict[str, Any]:
    if not args.results_csv:
        raise ValueError("--results-csv is required for backend=results_csv")

    out_dir = Path(args.output_dir)
    queries = pd.read_csv(args.queries, encoding=args.encoding)
    results_raw = pd.read_csv(args.results_csv, encoding=args.encoding)

    if "query" not in queries.columns:
        raise ValueError("queries CSV must contain query")
    if "query" not in results_raw.columns:
        raise ValueError("results CSV must contain query")
    if "rank" not in results_raw.columns:
        raise ValueError("results CSV must contain rank")

    target_mode = has_targets(queries)
    rows: list[dict[str, Any]] = []

    for _, qrow in queries.iterrows():
        query = str(qrow["query"])
        qhits = results_raw[results_raw["query"].astype(str) == query].copy()
        qhits["rank"] = pd.to_numeric(qhits["rank"], errors="coerce")
        qhits = qhits.sort_values("rank").head(args.top_k)

        for _, hit in qhits.iterrows():
            hit_text = hit.get(args.display_col, "")
            jaccard, missing, extra = token_jaccard(query, hit_text)

            hit_ok, doc_ok, page_ok = (
                is_hit_generic(qrow, hit, file_col=args.file_col, page_col=args.page_col, display_col=args.display_col)
                if target_mode
                else (False, False, False)
            )

            rows.append(
                {
                    "query": query,
                    "hit_text": hit_text,
                    "rank": int(hit.get("rank", 0)),
                    "score": float(hit.get("score", 0) or 0),
                    "n_query_tokens": len(query.split()),
                    "n_hit_tokens": len(str(hit_text).split()),
                    "jaccard": jaccard,
                    "missing_terms": missing,
                    "extra_terms": extra,
                    "target_doc": qrow.get("target_doc", ""),
                    "page": qrow.get("page", ""),
                    "hit_doc": hit.get(args.file_col, ""),
                    "hit_page": hit.get(args.page_col, ""),
                    "hit_id": hit.get(args.id_col, ""),
                    "is_hit": bool(hit_ok),
                    "is_target_doc_hit": bool(doc_ok),
                    "is_page_hit": bool(page_ok),
                }
            )

    return write_outputs(rows, queries, out_dir, target_mode)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if args.backend == "legacy_lsa":
        return evaluate_legacy_lsa(args)
    if args.backend == "make_ce_direct":
        return evaluate_make_ce_direct(args)
    if args.backend == "results_csv":
        return evaluate_results_csv(args)
    raise ValueError(f"unsupported backend: {args.backend}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)

    p.add_argument("--backend", choices=["legacy_lsa", "make_ce_direct", "results_csv"], default="legacy_lsa")

    p.add_argument("--model-dir")
    p.add_argument("--queries", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--encoding", default="utf-8-sig")

    p.add_argument("--vectorizer-path")
    p.add_argument("--metadata-path")
    p.add_argument("--metadata-format", choices=["auto", "pkl", "csv"], default="auto")
    p.add_argument("--mean-tfidf-path")
    p.add_argument("--use-mean-tfidf", action="store_true")
    p.add_argument("--allow-dense-mean", action="store_true")

    p.add_argument("--results-csv")

    p.add_argument("--vector-text-col", default="auto")
    p.add_argument("--file-col", default="name_org")
    p.add_argument("--page-col", default="pageno")
    p.add_argument("--display-col", default="text")
    p.add_argument("--id-col", default="chunk_id")

    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    print(json.dumps(evaluate(parse_args(argv)), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
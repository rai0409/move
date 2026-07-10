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
   step2/vectorizer.pkl
     step2/df.pkl or step2/df.csv
     step2/overall_mean_tfidf.pkl optional

   step1 artifacts are used only for native make_ce assets that do not exist in step2:
     step1/word2index.pkl
     step1/SVD_tuple.pkl
     step1/all_denominator.pkl
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
import hashlib
import json
import math
import pickle
import re
import statistics
import time
import unicodedata
from datetime import datetime, timezone
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

from lsa_preprocess_and_chunk import apply_config_replacements, compile_patterns, normalize_text, tokenize


QUERY_SCHEMA_COLUMNS = ["query", "target_doc", "page", "docid", "expected_text_contains", "model3の順位", "コメント"]
GROUND_TRUTH_AUX_COLUMNS = ["page", "docid", "expected_text_contains"]
PROFILE_FINGERPRINT_KEYS = [
    "unicode_normalization", "lowercase_policy", "pos_policy", "lemma_policy", "stopwords", "keep_words",
    "protected_patterns", "token_join_policy", "use_noun_compounds", "synonyms", "replacements", "candidate",
]


def value_present(value: Any) -> bool:
    return pd.notna(value) and str(value).strip().lower() not in {"", "nan", "none"}


def row_has_target(row: pd.Series) -> bool:
    return value_present(row.get("target_doc")) and any(value_present(row.get(col)) for col in GROUND_TRUTH_AUX_COLUMNS)


def has_targets(df: pd.DataFrame) -> bool:
    return bool(not df.empty and df.apply(row_has_target, axis=1).any())


def validate_query_schema(df: pd.DataFrame) -> None:
    missing = [col for col in ["query", "target_doc"] if col not in df.columns]
    if missing:
        raise ValueError(f"queries CSV missing required columns: {missing}; available={list(df.columns)}")
    if not any(col in df.columns for col in GROUND_TRUTH_AUX_COLUMNS):
        raise ValueError("queries CSV must contain at least one of page, docid, expected_text_contains")


def evaluation_normalize(value: Any) -> str:
    if not value_present(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = re.sub(r"[A-Z]", lambda m: m.group(0).lower(), text)
    text = re.sub(r"[\s\u3000]+", "", text)
    return re.sub(r"[、。，．,.]", "", text)


def document_normalize(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).replace("\\", "/").strip().lower()
    return text.rsplit("/", 1)[-1]


def page_matches(expected: Any, actual: Any) -> bool:
    if not value_present(expected):
        return True
    if not value_present(actual):
        return False
    expected_text = unicodedata.normalize("NFKC", str(expected)).strip()
    try:
        actual_num = int(float(str(actual).strip()))
    except Exception:
        return expected_text == str(actual).strip()
    range_match = re.fullmatch(r"(\d+)\s*[-~〜～]\s*(\d+)", expected_text)
    if range_match:
        return int(range_match.group(1)) <= actual_num <= int(range_match.group(2))
    allowed = []
    for part in re.split(r"[,、/]", expected_text):
        try:
            allowed.append(int(float(part.strip())))
        except Exception:
            pass
    return actual_num in allowed if allowed else expected_text == str(actual).strip()


def match_hit(qrow: pd.Series, hit: pd.Series, *, file_col: str, page_col: str, display_col: str,
              docid_col: str, id_col: str) -> dict[str, Any]:
    has_gt = row_has_target(qrow)
    if not has_gt:
        return {"is_hit": False, "matched_by": "no_ground_truth", "doc_hit": False, "page_hit": False,
                "text_hit": False, "docid_hit": False}
    doc_hit = document_normalize(qrow.get("target_doc")) == document_normalize(hit.get(file_col, ""))
    page_required = value_present(qrow.get("page"))
    text_required = value_present(qrow.get("expected_text_contains"))
    docid_required = value_present(qrow.get("docid"))
    page_hit = page_matches(qrow.get("page"), hit.get(page_col, "")) if page_required else True
    expected_text = evaluation_normalize(qrow.get("expected_text_contains"))
    text_hit = expected_text in evaluation_normalize(hit.get(display_col, "")) if text_required else True
    actual_docid = hit.get(docid_col, "") or hit.get(id_col, "")
    docid_hit = evaluation_normalize(qrow.get("docid")) == evaluation_normalize(actual_docid) if docid_required else False

    if not doc_hit:
        matched_by, is_hit = "not_found", False
    elif page_required and text_required and page_hit and text_hit:
        matched_by, is_hit = "target_doc_page_text", True
    elif page_required and not text_required and page_hit:
        matched_by, is_hit = "target_doc_page", True
    elif text_required and not page_required and text_hit:
        matched_by, is_hit = "target_doc_text", True
    elif not page_required and not text_required and docid_required and docid_hit:
        matched_by, is_hit = "target_doc_docid", True
    else:
        matched_by, is_hit = "target_doc_only", False
    return {"is_hit": is_hit, "matched_by": matched_by, "doc_hit": doc_hit, "page_hit": page_hit,
            "text_hit": text_hit, "docid_hit": docid_hit}


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

    if args.vectorizer_path:
        vectorizer_path = Path(args.vectorizer_path)
    elif (step2 / "vectorizer.pkl").exists():
        vectorizer_path = step2 / "vectorizer.pkl"
    else:
        vectorizer_path = step2 / "vectorizer.pkl"

    if args.metadata_path:
        metadata_path = Path(args.metadata_path)
    elif (step2 / "df.pkl").exists():
        metadata_path = step2 / "df.pkl"
    elif (step2 / "df.csv").exists():
        metadata_path = step2 / "df.csv"
    else:
        metadata_path = step2 / "df.pkl"

    mean_path = Path(args.mean_tfidf_path) if args.mean_tfidf_path else step2 / "overall_mean_tfidf.pkl"

    native_make_ce_artifacts = {
        "word2index": step1 / "word2index.pkl",
        "svd_tuple": step1 / "SVD_tuple.pkl",
        "all_denominator": step1 / "all_denominator.pkl",
    }
    return {
        "vectorizer": vectorizer_path,
        "metadata": metadata_path,
        "mean_tfidf": mean_path,
        "native_word2index": native_make_ce_artifacts["word2index"],
        "native_svd_tuple": native_make_ce_artifacts["svd_tuple"],
        "native_all_denominator": native_make_ce_artifacts["all_denominator"],
    }


def choose_vector_text_col(df: pd.DataFrame, requested: str) -> str:
    if requested and requested != "auto":
        if requested not in df.columns:
            raise ValueError(f"vector text column not found: {requested}; available={list(df.columns)}")
        return requested

    for col in ["text", "CHASEN_TEXT_cleaned", "wakati_text"]:
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


def failure_category(qrow: pd.Series, qhits: pd.DataFrame, query_tokens: str) -> str:
    if not row_has_target(qrow):
        return "no_ground_truth"
    if not query_tokens.strip():
        return "empty_query_tokens"
    if qhits.empty or not bool(qhits["doc_hit"].any()):
        return "no_target_document_in_top20"
    target_hits = qhits[qhits["doc_hit"] == True]  # noqa: E712
    if value_present(qrow.get("page")) and not bool(target_hits["page_hit"].any()):
        return "target_document_wrong_page"
    if value_present(qrow.get("expected_text_contains")) and not bool(target_hits["text_hit"].any()):
        return "expected_text_not_contained"
    if target_hits["hit_doc"].nunique() > 1:
        return "ambiguous_multiple_documents"
    if value_present(qrow.get("docid")) and not bool(target_hits["docid_hit"].any()):
        return "target_metadata_mismatch"
    return "other"


def summarize_query_results(rows: list[dict[str, Any]], queries: pd.DataFrame) -> pd.DataFrame:
    ranked = pd.DataFrame(rows)
    summaries: list[dict[str, Any]] = []
    for query_index, qrow in queries.reset_index(drop=True).iterrows():
        qhits = ranked[ranked["query_index"] == query_index].sort_values("rank") if not ranked.empty else pd.DataFrame()
        correct = qhits[qhits["is_hit"] == True] if not qhits.empty else pd.DataFrame()  # noqa: E712
        matched = correct.iloc[0] if not correct.empty else None
        top = qhits.iloc[0] if not qhits.empty else None
        best_rank = int(matched["rank"]) if matched is not None else None
        has_gt = row_has_target(qrow)
        matched_by = str(matched["matched_by"]) if matched is not None else ("no_ground_truth" if not has_gt else "not_found")
        query_tokens = str(top.get("query_vector_text", "")) if top is not None else ""
        failure = "" if matched is not None else failure_category(qrow, qhits, query_tokens)
        summaries.append({
            "query_index": query_index, "query": str(qrow.get("query", "")), "target_doc": qrow.get("target_doc", ""),
            "expected_page": qrow.get("page", ""), "expected_docid": qrow.get("docid", ""),
            "expected_text_contains": qrow.get("expected_text_contains", ""), "best_correct_rank": best_rank,
            "recall_at_1_hit": bool(best_rank is not None and best_rank <= 1),
            "recall_at_5_hit": bool(best_rank is not None and best_rank <= 5),
            "recall_at_20_hit": bool(best_rank is not None and best_rank <= 20),
            "reciprocal_rank": 0.0 if best_rank is None else 1.0 / best_rank,
            "matched_result_doc": "" if matched is None else matched.get("hit_doc", ""),
            "matched_result_page": "" if matched is None else matched.get("hit_page", ""),
            "matched_result_docid": "" if matched is None else matched.get("hit_docid", ""),
            "matched_result_chunk_id": "" if matched is None else matched.get("hit_id", ""),
            "matched_by": matched_by, "failure_reason": failure,
            "top_1_doc": "" if top is None else top.get("hit_doc", ""),
            "top_1_page": "" if top is None else top.get("hit_page", ""),
            "top_1_docid": "" if top is None else top.get("hit_docid", ""),
            "top_1_chunk_id": "" if top is None else top.get("hit_id", ""),
            "top_1_chunk_text": "" if top is None else top.get("hit_text", ""),
            "top_1_score": None if top is None else top.get("score"),
            "query_vector_text": query_tokens, "ground_truth": has_gt,
            "target_doc_found": bool(not qhits.empty and qhits["doc_hit"].any()),
            "exact_page_found": bool(not qhits.empty and qhits["doc_hit"].any() and qhits.loc[qhits["doc_hit"] == True, "page_hit"].any()),  # noqa: E712
            "expected_text_found": bool(not qhits.empty and qhits["doc_hit"].any() and qhits.loc[qhits["doc_hit"] == True, "text_hit"].any()),  # noqa: E712
            "comment": qrow.get("コメント", ""), "model3_rank": qrow.get("model3の順位", ""),
        })
    return pd.DataFrame(summaries)


def build_metrics(summary: pd.DataFrame) -> dict[str, Any]:
    evaluated = summary[summary["ground_truth"] == True].copy()  # noqa: E712
    diagnostic_count = int((summary["ground_truth"] == False).sum())  # noqa: E712
    if evaluated.empty:
        return {"query_count": len(summary), "evaluated_query_count": 0, "diagnostic_only_query_count": diagnostic_count,
                "recall_at_1": None, "recall_at_5": None, "recall_at_20": None, "mrr": None,
                "recall_at_3": None, "recall_at_10": None,
                "mean_rank": None, "median_rank": None, "no_hit_query_count": 0, "hit_query_count": 0,
                "target_doc_hit_rate": None, "exact_page_hit_rate": None, "expected_text_hit_rate": None,
                "improved_query_count": 0, "regressed_query_count": 0, "unchanged_query_count": 0,
                "worst_rank_drop": None, "best_rank_gain": None, "validation_mode": "diagnostic_only"}
    ranks = pd.to_numeric(evaluated["best_correct_rank"], errors="coerce").dropna().astype(int).tolist()
    denom = len(evaluated)
    page_applicable = evaluated[evaluated["expected_page"].map(value_present)]
    text_applicable = evaluated[evaluated["expected_text_contains"].map(value_present)]
    metrics = {
        "query_count": len(summary), "evaluated_query_count": denom, "diagnostic_only_query_count": diagnostic_count,
        "recall_at_1": float(evaluated["recall_at_1_hit"].mean()),
        "recall_at_3": float((pd.to_numeric(evaluated["best_correct_rank"], errors="coerce") <= 3).fillna(False).mean()),
        "recall_at_5": float(evaluated["recall_at_5_hit"].mean()),
        "recall_at_10": float((pd.to_numeric(evaluated["best_correct_rank"], errors="coerce") <= 10).fillna(False).mean()),
        "recall_at_20": float(evaluated["recall_at_20_hit"].mean()),
        "mrr": float(evaluated["reciprocal_rank"].mean()),
        "mean_rank": float(statistics.mean(ranks)) if ranks else None,
        "median_rank": float(statistics.median(ranks)) if ranks else None,
        "no_hit_query_count": int(evaluated["best_correct_rank"].isna().sum()), "hit_query_count": len(ranks),
        "target_doc_hit_rate": float(evaluated["target_doc_found"].mean()),
        "exact_page_hit_rate": float(page_applicable["exact_page_found"].mean()) if not page_applicable.empty else None,
        "expected_text_hit_rate": float(text_applicable["expected_text_found"].mean()) if not text_applicable.empty else None,
        "improved_query_count": 0, "regressed_query_count": 0, "unchanged_query_count": 0,
        "worst_rank_drop": None, "best_rank_gain": None,
        "validation_mode": "measured" if diagnostic_count == 0 else "measured_plus_diagnostic_only",
    }
    metrics["composite_score"] = metrics["recall_at_1"] * 35 + metrics["recall_at_5"] * 25 + metrics["recall_at_20"] * 20 + metrics["mrr"] * 20
    metrics["critical_query_regression"] = False
    return metrics


def write_outputs(rows: list[dict[str, Any]], queries: pd.DataFrame, out_dir: Path, target_mode: bool) -> dict[str, Any]:
    del target_mode
    out_dir.mkdir(parents=True, exist_ok=True)
    ranked = pd.DataFrame(rows)
    summary = summarize_query_results(rows, queries)
    ranked.to_csv(out_dir / "ranked_results.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "query_results.csv", index=False, encoding="utf-8-sig")
    error_cols = ["query", "target_doc", "expected_page", "expected_text_contains", "best_correct_rank", "top_1_doc",
                  "top_1_page", "top_1_chunk_text", "top_1_score", "failure_reason", "comment"]
    errors = summary[(summary["best_correct_rank"].isna()) | (~summary["ground_truth"])].copy()
    errors.rename(columns={"failure_reason": "failure_category"})[[c if c != "failure_reason" else "failure_category" for c in error_cols]].to_csv(
        out_dir / "error_analysis.csv", index=False, encoding="utf-8-sig"
    )
    errors[errors["ground_truth"] == True].to_csv(out_dir / "failed_queries.csv", index=False, encoding="utf-8-sig")  # noqa: E712
    summary[summary["ground_truth"] == False].to_csv(out_dir / "public_query_diagnostics.csv", index=False, encoding="utf-8-sig")  # noqa: E712
    metrics = build_metrics(summary)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics

def _resolve_queries_encoding(args: argparse.Namespace) -> str:
    queries_encoding = getattr(args, "queries_encoding", None)
    if queries_encoding:
        return queries_encoding
    return args.encoding


def _read_queries_csv(args: argparse.Namespace) -> pd.DataFrame:
    requested_encoding = getattr(args, "queries_encoding", None)
    default_encoding = getattr(args, "encoding", "utf-8-sig")

    candidates: list[str] = []

    if requested_encoding:
        candidates.append(requested_encoding)

    if default_encoding not in candidates:
        candidates.append(default_encoding)

    for fallback in ("utf-8-sig", "utf-8", "cp932"):
        if fallback not in candidates:
            candidates.append(fallback)

    errors: list[str] = []
    for encoding in candidates:
        try:
            df = pd.read_csv(args.queries, encoding=encoding)
            validate_query_schema(df)
            return df
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")

    tried = ", ".join(candidates)
    details = " | ".join(errors)
    raise ValueError(
        f"Failed to read queries CSV: {args.queries}. "
        f"Tried encodings: {tried}. Details: {details}"
    )


def file_fingerprint(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def profile_fingerprint(profile: dict[str, Any] | None) -> str | None:
    if profile is None:
        return None
    selected = {key: profile.get(key) for key in PROFILE_FINGERPRINT_KEYS}
    payload = json.dumps(selected, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_evaluation_profiles(model_dir: Path, args: argparse.Namespace) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str, str, bool]:
    document_path = model_dir / "preprocessing_profile.json"
    document_profile = json.loads(document_path.read_text(encoding="utf-8")) if document_path.exists() else None
    requested = getattr(args, "query_preprocessing_profile", "auto")
    if requested == "off":
        query_profile, query_source = None, "off"
    elif requested in {"auto", "required"}:
        query_profile, query_source = document_profile, str(document_path)
    else:
        query_path = Path(requested)
        query_profile = json.loads(query_path.read_text(encoding="utf-8")) if query_path.exists() else None
        query_source = str(query_path)
    doc_fp, query_fp = profile_fingerprint(document_profile), profile_fingerprint(query_profile)
    return document_profile, query_profile, str(document_path), query_source, bool(doc_fp and doc_fp == query_fp)


def write_evaluation_lineage(out_dir: Path, args: argparse.Namespace, queries: pd.DataFrame, metrics: dict[str, Any],
                             document_profile: dict[str, Any] | None, query_profile: dict[str, Any] | None,
                             document_source: str, query_source: str, profiles_match: bool) -> dict[str, Any]:
    candidate = (document_profile or {}).get("candidate") or {}
    lineage = {
        "query_csv": str(args.queries), "query_csv_fingerprint": file_fingerprint(args.queries),
        "query_count": int(len(queries)), "evaluated_query_count": int(metrics.get("evaluated_query_count") or 0),
        "ground_truth_columns": [col for col in ["target_doc", "page", "docid", "expected_text_contains"] if col in queries.columns],
        "document_preprocessing_profile": document_source,
        "query_preprocessing_profile": query_source,
        "document_preprocessing_fingerprint": profile_fingerprint(document_profile),
        "query_preprocessing_fingerprint": profile_fingerprint(query_profile), "profiles_match": profiles_match,
        "vector_input_source": "lsa_tokens_str", "vector_text_col": args.vector_text_col,
        "make_ce_clean_effective": False, "make_ce_chunk_effective": False,
        "tokenizer": candidate.get("tokenizer_name"), "dictionary": candidate.get("dictionary"),
        "dictionary_path": candidate.get("dictionary_path"), "dictionary_charset": candidate.get("dictionary_charset"),
        "dictionary_version": candidate.get("dictionary_version"),
        "morphology_profile": (document_profile or {}).get("pos_policy"),
        "chunk_profile": getattr(args, "chunk_profile", "current"), "svd_dim": int(getattr(args, "svd_dim", 100)),
        "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
        "validation_mode": metrics.get("validation_mode"),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "evaluation_lineage.json").write_text(json.dumps(lineage, ensure_ascii=False, indent=2), encoding="utf-8")
    return lineage


def write_profile_mismatch_outputs(out_dir: Path, queries: pd.DataFrame, metrics: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [{
        "query": row.get("query", ""), "target_doc": row.get("target_doc", ""), "expected_page": row.get("page", ""),
        "expected_text_contains": row.get("expected_text_contains", ""), "best_correct_rank": None,
        "top_1_doc": "", "top_1_page": "", "top_1_chunk_text": "", "top_1_score": None,
        "failure_category": "tokenizer_mismatch", "comment": row.get("コメント", ""),
    } for _, row in queries.iterrows()]
    pd.DataFrame(rows).to_csv(out_dir / "error_analysis.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame().to_csv(out_dir / "query_results.csv", index=False, encoding="utf-8-sig")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


def load_query_preprocessor(model_dir: Path, args: argparse.Namespace) -> tuple[dict[str, Any] | None, str]:
    requested = getattr(args, "query_preprocessing_profile", "auto")
    if requested == "off":
        return None, "pretokenized_or_vectorizer_native"
    path = Path(requested) if requested not in {"auto", "required"} else model_dir / "preprocessing_profile.json"
    if not path.exists():
        if requested == "required":
            raise FileNotFoundError(f"query preprocessing profile not found: {path}")
        return None, "profile_missing_assume_pretokenized"
    return json.loads(path.read_text(encoding="utf-8")), str(path)


def preprocess_query(query: str, profile: dict[str, Any] | None) -> str:
    if profile is None:
        return query
    replaced, _ = apply_config_replacements(query, profile.get("replacements") or [])
    normalized, _ = normalize_text(replaced, compile_patterns(profile.get("remove_line_patterns") or []))
    candidate = profile.get("candidate") or {}
    tokens = tokenize(
        normalized, str(candidate.get("tokenizer_name") or ""),
        set(profile.get("stopwords") or []), bool(profile.get("use_noun_compounds")), profile.get("synonyms") or {}, set(profile.get("keep_words") or []),
        morphology_profile=str(profile.get("pos_policy") or "content_lemma"),
        dictionary_path=candidate.get("dictionary_path"), mecab_options=str(candidate.get("mecab_options") or ""),
    )
    return " ".join(tokens)

def evaluate_legacy_lsa(args: argparse.Namespace) -> dict[str, Any]:
    model_dir = Path(args.model_dir)
    out_dir = Path(args.output_dir)
    print(
        "[evaluate_lsa] backend=legacy_lsa "
        f"model_dir={model_dir} queries={args.queries} output_dir={out_dir} vector_text_col=lsa_tokens_str"
    )

    vectorizer = joblib.load(model_dir / "tfidf_vectorizer.joblib")
    svd = joblib.load(model_dir / "truncated_svd.joblib")
    vectors = np.load(model_dir / "lsa_vectors.npy")
    metadata = pd.read_csv(model_dir / "metadata.csv", encoding="utf-8-sig")
    queries = _read_queries_csv(args)
    document_profile, query_profile, document_profile_source, query_profile_source, profiles_match = load_evaluation_profiles(model_dir, args)
    if not profiles_match:
        blocked = {"query_count": len(queries), "evaluated_query_count": 0,
                   "diagnostic_only_query_count": int(len(queries)), "validation_mode": "preprocessing_profile_mismatch",
                   "recall_at_1": None, "recall_at_5": None, "recall_at_20": None, "mrr": None}
        write_profile_mismatch_outputs(out_dir, queries, blocked)
        write_evaluation_lineage(out_dir, args, queries, blocked, document_profile, query_profile,
                                 document_profile_source, query_profile_source, False)
        raise RuntimeError("preprocessing_profile_mismatch")

    target_mode = has_targets(queries)
    rows: list[dict[str, Any]] = []

    started = time.perf_counter()
    for query_index, qrow in queries.reset_index(drop=True).iterrows():
        query = str(qrow["query"])
        query_vector_text = preprocess_query(query, query_profile)
        tfidf = vectorizer.transform([query_vector_text])
        qvec = svd.transform(tfidf) if svd is not None else tfidf.toarray()
        qvec = normalize(qvec, norm="l2")[0]

        scores = vectors @ qvec
        order = np.argsort(-scores)[: max(args.top_k, 20)]

        for rank, idx in enumerate(order, start=1):
            hit = metadata.iloc[int(idx)]
            jaccard, missing, extra = token_jaccard(query, hit.get("lsa_tokens_str", ""))

            matched = match_hit(qrow, hit, file_col="file_name_out", page_col="page_out", display_col="chunk_text",
                                docid_col=args.docid_col, id_col="chunk_id_out")

            rows.append(
                {
                    "query_index": query_index, "query": query,
                    "query_vector_text": query_vector_text,
                    "hit_text": hit.get("chunk_text", ""),
                    "rank": rank,
                    "score": float(scores[int(idx)]),
                    "n_query_tokens": len(query_vector_text.split()),
                    "n_hit_tokens": len(str(hit.get("lsa_tokens_str", "")).split()),
                    "jaccard": jaccard,
                    "missing_terms": missing,
                    "extra_terms": extra,
                    "target_doc": qrow.get("target_doc", ""),
                    "page": qrow.get("page", ""),
                    "hit_doc": hit.get("file_name_out", ""),
                    "hit_page": hit.get("page_out", ""),
                    "hit_id": hit.get("chunk_id_out", ""),
                    "hit_docid": hit.get(args.docid_col, ""),
                    "is_hit": bool(matched["is_hit"]), "matched_by": matched["matched_by"],
                    "doc_hit": bool(matched["doc_hit"]), "page_hit": bool(matched["page_hit"]),
                    "text_hit": bool(matched["text_hit"]), "docid_hit": bool(matched["docid_hit"]),
                }
            )

    metrics = write_outputs(rows, queries, out_dir, target_mode)
    metrics["evaluation_time_seconds"] = time.perf_counter() - started
    metrics["query_preprocessing_source"] = query_profile_source
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    write_evaluation_lineage(out_dir, args, queries, metrics, document_profile, query_profile,
                             document_profile_source, query_profile_source, profiles_match)
    return metrics


def evaluate_make_ce_direct(args: argparse.Namespace) -> dict[str, Any]:
    if not args.model_dir:
        raise ValueError("--model-dir is required for backend=make_ce_direct")

    model_dir = Path(args.model_dir)
    out_dir = Path(args.output_dir)

    queries = _read_queries_csv(args)
    document_profile, query_profile, document_profile_source, query_profile_source, profiles_match = load_evaluation_profiles(model_dir, args)
    if not profiles_match:
        blocked = {"query_count": len(queries), "evaluated_query_count": 0,
                   "diagnostic_only_query_count": int(len(queries)), "validation_mode": "preprocessing_profile_mismatch",
                   "recall_at_1": None, "recall_at_5": None, "recall_at_20": None, "mrr": None}
        write_profile_mismatch_outputs(out_dir, queries, blocked)
        write_evaluation_lineage(out_dir, args, queries, blocked, document_profile, query_profile,
                                 document_profile_source, query_profile_source, False)
        raise RuntimeError("preprocessing_profile_mismatch")

    paths = autodetect_make_ce_paths(model_dir, args)

    if not paths["vectorizer"].exists():
        raise FileNotFoundError(f"vectorizer.pkl not found: {paths['vectorizer']}")
    if not paths["metadata"].exists():
        raise FileNotFoundError(f"df.pkl/df.csv not found: {paths['metadata']}")

    vectorizer = load_pickle_any(paths["vectorizer"])
    metadata = read_metadata(paths["metadata"], args.metadata_format, args.encoding)

    vector_text_col = choose_vector_text_col(metadata, args.vector_text_col)
    print(
        "[evaluate_lsa] backend=make_ce_direct "
        f"model_dir={model_dir} metadata={paths['metadata']} vectorizer={paths['vectorizer']} "
        f"queries={args.queries} output_dir={out_dir} vector_text_col={vector_text_col}"
    )

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

    started = time.perf_counter()
    for query_index, qrow in queries.reset_index(drop=True).iterrows():
        query = str(qrow["query"])
        query_vector_text = preprocess_query(query, query_profile)

        qvec = vectorizer.transform([query_vector_text])
        qvec = maybe_subtract_mean(qvec, mean_tfidf, args.allow_dense_mean)
        qvec = safe_normalize_vector(qvec)

        if sparse is not None and sparse.issparse(doc_matrix):
            scores = np.asarray(doc_matrix @ qvec.reshape(-1, 1)).ravel()
        else:
            scores = np.asarray(doc_matrix @ qvec).ravel()

        if not np.all(np.isfinite(scores)):
            scores = np.nan_to_num(scores, nan=-math.inf, posinf=math.inf, neginf=-math.inf)

        order = np.argsort(-scores)[: max(args.top_k, 20)]

        for rank, idx in enumerate(order, start=1):
            hit = metadata.iloc[int(idx)]
            hit_text = hit.get(args.display_col, "")
            vector_text = hit.get(vector_text_col, "")

            jaccard, missing, extra = token_jaccard(query, vector_text)

            matched = match_hit(qrow, hit, file_col=args.file_col, page_col=args.page_col, display_col=args.display_col,
                                docid_col=args.docid_col, id_col=args.id_col)

            rows.append(
                {
                    "query_index": query_index, "query": query, "query_vector_text": query_vector_text,
                    "hit_text": hit_text,
                    "rank": rank,
                    "score": float(scores[int(idx)]),
                    "n_query_tokens": len(query_vector_text.split()),
                    "n_hit_tokens": len(str(vector_text).split()),
                    "jaccard": jaccard,
                    "missing_terms": missing,
                    "extra_terms": extra,
                    "target_doc": qrow.get("target_doc", ""),
                    "page": qrow.get("page", ""),
                    "hit_doc": hit.get(args.file_col, ""),
                    "hit_page": hit.get(args.page_col, ""),
                    "hit_id": hit.get(args.id_col, ""),
                    "hit_docid": hit.get(args.docid_col, ""),
                    "is_hit": bool(matched["is_hit"]), "matched_by": matched["matched_by"],
                    "doc_hit": bool(matched["doc_hit"]), "page_hit": bool(matched["page_hit"]),
                    "text_hit": bool(matched["text_hit"]), "docid_hit": bool(matched["docid_hit"]),
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
        "profiles_match": profiles_match,
    }
    (out_dir / "evaluate_backend_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    metrics = write_outputs(rows, queries, out_dir, target_mode)
    metrics["evaluation_time_seconds"] = time.perf_counter() - started
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    write_evaluation_lineage(out_dir, args, queries, metrics, document_profile, query_profile,
                             document_profile_source, query_profile_source, profiles_match)
    return metrics


def evaluate_results_csv(args: argparse.Namespace) -> dict[str, Any]:
    if not args.results_csv:
        raise ValueError("--results-csv is required for backend=results_csv")

    out_dir = Path(args.output_dir)
    print(
        "[evaluate_lsa] backend=results_csv "
        f"results_csv={args.results_csv} queries={args.queries} output_dir={out_dir} text_col={args.display_col}"
    )
    queries = _read_queries_csv(args)
    results_raw = pd.read_csv(args.results_csv, encoding=args.encoding)

    if "query" not in results_raw.columns:
        raise ValueError("results CSV must contain query")
    if "rank" not in results_raw.columns:
        raise ValueError("results CSV must contain rank")

    target_mode = has_targets(queries)
    rows: list[dict[str, Any]] = []

    for query_index, qrow in queries.reset_index(drop=True).iterrows():
        query = str(qrow["query"])
        qhits = results_raw[results_raw["query"].astype(str) == query].copy()
        qhits["rank"] = pd.to_numeric(qhits["rank"], errors="coerce")
        qhits = qhits.sort_values("rank").head(max(args.top_k, 20))

        for _, hit in qhits.iterrows():
            hit_text = hit.get(args.display_col, "")
            jaccard, missing, extra = token_jaccard(query, hit_text)

            matched = match_hit(qrow, hit, file_col=args.file_col, page_col=args.page_col, display_col=args.display_col,
                                docid_col=args.docid_col, id_col=args.id_col)

            rows.append(
                {
                    "query_index": query_index, "query": query, "query_vector_text": query,
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
                    "hit_docid": hit.get(args.docid_col, ""),
                    "is_hit": bool(matched["is_hit"]), "matched_by": matched["matched_by"],
                    "doc_hit": bool(matched["doc_hit"]), "page_hit": bool(matched["page_hit"]),
                    "text_hit": bool(matched["text_hit"]), "docid_hit": bool(matched["docid_hit"]),
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
    p.add_argument("--queries-encoding")
    p.add_argument("--query-preprocessing-profile", default="auto", help="auto|required|off|path to preprocessing_profile.json")

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
    p.add_argument("--display-col", default="chunk_text")
    p.add_argument("--id-col", default="chunk_id")
    p.add_argument("--docid-col", default="docid")
    p.add_argument("--chunk-profile", default="current")
    p.add_argument("--svd-dim", type=int, default=100)

    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    print(json.dumps(evaluate(parse_args(argv)), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

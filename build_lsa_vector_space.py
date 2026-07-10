#!/usr/bin/env python3
"""Build a TF-IDF + TruncatedSVD(LSA) vector space from an LSA-ready CSV."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


def mostly_empty(series: pd.Series) -> bool:
    empty = series.isna() | (series.astype(str).str.strip() == "")
    return bool(empty.mean() > 0.05)


def analyzer(text: str) -> list[str]:
    return str(text).split()


def build_vector_space(args: argparse.Namespace) -> dict[str, object]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        "[build_lsa_vector_space] input="
        f"{args.input} output_dir={output_dir} text_col={args.text_col} "
        f"id_col={args.id_col} display_col={args.display_col}"
    )
    df = pd.read_csv(args.input, encoding=args.encoding)
    if args.text_col not in df.columns:
        raise ValueError(f"text_col missing: {args.text_col}")
    if mostly_empty(df[args.text_col]):
        raise ValueError(f"text_col is mostly empty: {args.text_col}")

    texts = df[args.text_col].fillna("").astype(str).tolist()
    min_df = min(max(1, int(args.min_df)), max(1, len(texts)))
    started = time.perf_counter()
    vectorizer = TfidfVectorizer(
        analyzer="word", tokenizer=str.split, preprocessor=None, token_pattern=None, lowercase=False,
        min_df=min_df, max_df=args.max_df, sublinear_tf=args.sublinear_tf, norm=args.norm,
        use_idf=args.use_idf, smooth_idf=args.smooth_idf, binary=args.binary,
        ngram_range=(args.ngram_min, args.ngram_max),
    )
    tfidf = vectorizer.fit_transform(texts)
    n_features = tfidf.shape[1]
    max_components = max(0, min(int(args.svd_dim), n_features - 1, len(texts) - 1))
    svd: TruncatedSVD | None = None
    if max_components >= 1:
        svd = TruncatedSVD(n_components=max_components, random_state=args.random_state, n_iter=args.n_iter, algorithm=args.algorithm)
        vectors = svd.fit_transform(tfidf)
    else:
        vectors = tfidf.toarray()
    vectors = normalize(vectors, norm="l2")

    metadata = pd.DataFrame(
        {
            "chunk_id_out": df[args.id_col] if args.id_col in df.columns else range(len(df)),
            "chunk_text": df[args.display_col] if args.display_col in df.columns else "",
            "file_name_out": df[args.file_col] if args.file_col in df.columns else "",
            "page_out": df[args.page_col] if args.page_col in df.columns else "",
            "lsa_tokens_str": texts,
            "docid": df["docid"] if "docid" in df.columns else "",
        }
    )
    metadata.to_csv(output_dir / "metadata.csv", index=False, encoding="utf-8-sig")
    np.save(output_dir / "lsa_vectors.npy", vectors)
    joblib.dump(vectorizer, output_dir / "tfidf_vectorizer.joblib")
    joblib.dump(svd, output_dir / "truncated_svd.joblib")

    token_docs: Counter[str] = Counter()
    token_freq: Counter[str] = Counter()
    for text in texts:
        toks = text.split()
        token_freq.update(toks)
        token_docs.update(set(toks))
    term_stats = pd.DataFrame(
        [{"term": t, "frequency": int(token_freq[t]), "doc_frequency": int(token_docs[t]), "doc_freq_ratio": token_docs[t] / max(len(texts), 1)} for t in sorted(token_freq)]
    )
    term_stats.to_csv(output_dir / "term_stats.csv", index=False, encoding="utf-8-sig")
    if args.export_term_vectors:
        vocab_by_index = {idx: term for term, idx in vectorizer.vocabulary_.items()}
        if svd is not None:
            term_vectors = svd.components_.T
        else:
            term_vectors = np.eye(n_features)
        term_vectors = normalize(term_vectors, norm="l2")
        terms = [vocab_by_index[i] for i in range(n_features)]
        mean_tfidf = np.asarray(tfidf.mean(axis=0)).ravel()
        term_meta = pd.DataFrame(
            {
                "term": terms,
                "term_index": list(range(n_features)),
                "doc_frequency": [int(token_docs[t]) for t in terms],
                "mean_tfidf": [float(mean_tfidf[i]) for i in range(n_features)],
            }
        )
        np.save(output_dir / "term_lsa_vectors.npy", term_vectors)
        term_meta.to_csv(output_dir / "term_metadata.csv", index=False, encoding="utf-8-sig")
    report = {
        "rows": len(df),
        "n_features": int(n_features),
        "svd_dim_requested": int(args.svd_dim),
        "svd_dim_used": int(max_components),
        "min_df_used": int(min_df),
        "max_df": args.max_df,
        "sublinear_tf": bool(args.sublinear_tf),
        "norm": args.norm,
        "use_idf": bool(args.use_idf),
        "smooth_idf": bool(args.smooth_idf),
        "binary": bool(args.binary),
        "ngram_range": [args.ngram_min, args.ngram_max],
        "tokenizer": "whitespace_analyzer", "preprocessor": None, "token_pattern": None, "lowercase": False,
        "random_state": args.random_state, "n_iter": args.n_iter, "algorithm": args.algorithm,
        "explained_variance_ratio_sum": float(svd.explained_variance_ratio_.sum()) if svd is not None else None,
        "vector_build_time_seconds": time.perf_counter() - started,
        "export_term_vectors": bool(args.export_term_vectors),
    }
    (output_dir / "build_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    profile_source = Path(args.preprocessing_profile) if args.preprocessing_profile else Path(args.input).parent / "preprocessing_profile.json"
    if profile_source.exists():
        shutil.copyfile(profile_source, output_dir / "preprocessing_profile.json")
    lineage_cols = [c for c in ["tokenizer_name", "morphology_profile", "dictionary_name", "dictionary_version", "dictionary_charset", "dictionary_path", "split_mode", "chunk_profile"] if c in df.columns]
    lineage = {c: sorted(str(x) for x in df[c].dropna().unique()) for c in lineage_cols}
    lineage.update({"input": str(args.input), "text_col": args.text_col, "tfidf_svd": report})
    (output_dir / "model_lineage.json").write_text(json.dumps(lineage, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--text-col", default="lsa_tokens_str")
    p.add_argument("--id-col", default="chunk_id_out")
    p.add_argument("--display-col", default="chunk_text")
    p.add_argument("--file-col", default="file_name_out")
    p.add_argument("--page-col", default="page_out")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--min-df", type=int, default=3)
    p.add_argument("--max-df", type=float, default=0.95)
    p.add_argument("--svd-dim", type=int, default=100)
    p.add_argument("--sublinear-tf", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--norm", choices=["l1", "l2"], default="l2")
    p.add_argument("--use-idf", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--smooth-idf", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--binary", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--ngram-min", type=int, default=1)
    p.add_argument("--ngram-max", type=int, default=1)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--n-iter", type=int, default=5)
    p.add_argument("--algorithm", choices=["randomized", "arpack"], default="randomized")
    p.add_argument("--preprocessing-profile")
    p.add_argument("--encoding", default="utf-8-sig")
    p.add_argument("--export-term-vectors", action="store_true")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    report = build_vector_space(parse_args(argv))
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build a TF-IDF + TruncatedSVD(LSA) vector space from an LSA-ready CSV."""

from __future__ import annotations

import argparse
import json
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
    df = pd.read_csv(args.input, encoding=args.encoding)
    if args.text_col not in df.columns:
        raise ValueError(f"text_col missing: {args.text_col}")
    if mostly_empty(df[args.text_col]):
        raise ValueError(f"text_col is mostly empty: {args.text_col}")

    texts = df[args.text_col].fillna("").astype(str).tolist()
    min_df = min(max(1, int(args.min_df)), max(1, len(texts)))
    vectorizer = TfidfVectorizer(analyzer=analyzer, tokenizer=None, preprocessor=None, token_pattern=None, min_df=min_df, max_df=args.max_df)
    tfidf = vectorizer.fit_transform(texts)
    n_features = tfidf.shape[1]
    max_components = max(0, min(int(args.svd_dim), n_features - 1, len(texts) - 1))
    svd: TruncatedSVD | None = None
    if max_components >= 1:
        svd = TruncatedSVD(n_components=max_components, random_state=42)
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
        "export_term_vectors": bool(args.export_term_vectors),
    }
    (output_dir / "build_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
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
    p.add_argument("--svd-dim", type=int, default=150)
    p.add_argument("--encoding", default="utf-8-sig")
    p.add_argument("--export-term-vectors", action="store_true")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    report = build_vector_space(parse_args(argv))
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

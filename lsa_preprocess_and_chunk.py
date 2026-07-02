#!/usr/bin/env python3
"""Preprocess CSV text for Japanese TF-IDF + TruncatedSVD(LSA) retrieval."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
import tempfile
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd


MODEL_TARGET = "TFIDF_TRUNCATED_SVD_LSA"

CHUNK_PROFILES: dict[str, dict[str, int | None]] = {
    "none": {"target_chars": None, "overlap": 0, "hard_max": None, "min_chars": 0},
    "small": {"target_chars": 450, "overlap": 60, "hard_max": 700, "min_chars": 120},
    "medium": {"target_chars": 750, "overlap": 100, "hard_max": 1100, "min_chars": 180},
    "large": {"target_chars": 1050, "overlap": 150, "hard_max": 1500, "min_chars": 250},
}

RISKY_KEEP_WORDS = {
    "ない",
    "ある",
    "する",
    "なる",
    "いる",
    "れる",
    "られる",
    "必要",
    "不要",
    "可能",
    "不可",
    "対象",
    "対象外",
    "例外",
    "場合",
    "以上",
    "以下",
    "以内",
    "未満",
}

DEFAULT_SAFE_STOPWORDS = {
    "の",
    "に",
    "は",
    "を",
    "が",
    "で",
    "と",
    "も",
    "や",
    "か",
    "こと",
    "もの",
    "ため",
    "よう",
    "など",
    "また",
    "その",
    "この",
    "あの",
    "それ",
    "これ",
    "あれ",
}

DEFAULT_REMOVE_PATTERNS = [
    r"^\s*$",
    r"^\s*\d+\s*$",
    r"^\s*-\s*\d+\s*-\s*$",
    r"^\s*Page\s+\d+(\s*/\s*\d+)?\s*$",
    r"^\s*Copyright.*$",
    r"^\s*©.*$",
]

ZERO_WIDTH_RE = re.compile("[\u200b\u200c\u200d\ufeff]")
UNICODE_SPACES_RE = re.compile(r"[\u00a0\u1680\u180e\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]")
JA_CHAR = r"\u3040-\u309f\u30a0-\u30ff\u3400-\u9fff々〆ヵヶ"
KANJI_RE = re.compile(r"[\u3400-\u9fff々〆]")
KANA_RE = re.compile(r"[\u3040-\u30ffー]")
ASCII_RE = re.compile(r"[A-Za-z]")
NUM_RE = re.compile(r"\d")
TOKEN_RE = re.compile(
    rf"[A-Za-z]+(?:[A-Za-z0-9_+\-./#]*[A-Za-z0-9])?|[A-Za-z]*\d+[A-Za-z0-9_+\-./#]*|"
    rf"[\u30a0-\u30ffー]{{2,}}|[\u3400-\u9fff々〆]{{2,}}|[{JA_CHAR}]{{2,}}"
)
CAMEL_RE = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])")


def is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def read_list_file(path: str | Path | None, default: Sequence[str] | None = None) -> list[str]:
    if path is None:
        return list(default or [])
    p = Path(path)
    if not p.exists():
        return list(default or [])
    values: list[str] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        item = line.strip()
        if item and not item.startswith("#"):
            values.append(item)
    return values


def default_if_exists(path: str) -> str | None:
    return path if Path(path).exists() else None


def load_synonyms(path: str | None, max_terms: int) -> dict[str, list[str]]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError("--use-synonyms-json must contain an object")
    out: dict[str, list[str]] = {}
    for key, values in raw.items():
        if isinstance(values, str):
            vals = [values]
        elif isinstance(values, list):
            vals = [str(v) for v in values if str(v).strip()]
        else:
            continue
        out[str(key)] = vals[:max_terms]
    return out


def compile_patterns(patterns: Iterable[str]) -> list[re.Pattern[str]]:
    return [re.compile(p, re.IGNORECASE) for p in patterns if p.strip()]


def normalize_text(value: Any, remove_patterns: Sequence[re.Pattern[str]] | None = None) -> tuple[str, list[str]]:
    flags: list[str] = []
    if is_missing(value):
        return "", ["empty_input"]
    text = str(value)
    if text != unicodedata.normalize("NFKC", text):
        flags.append("nfkc")
    text = unicodedata.normalize("NFKC", text)
    text = UNICODE_SPACES_RE.sub(" ", text)
    text = ZERO_WIDTH_RE.sub("", text)
    text = text.replace("[NL]", "\n").replace("\\n", "\n")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"([A-Za-z])-\n([A-Za-z])", r"\1\2", text)
    previous = None
    while previous != text:
        previous = text
        text = re.sub(rf"([{JA_CHAR}])[ \t]+([{JA_CHAR}])", r"\1\2", text)
    text = re.sub(rf"([{JA_CHAR}])\n+([A-Za-z0-9])", r"\1 \2", text)
    text = re.sub(rf"([A-Za-z0-9])\n+([{JA_CHAR}])", r"\1 \2", text)
    text = re.sub(rf"([{JA_CHAR}])([A-Za-z0-9])", r"\1 \2", text)
    text = re.sub(rf"([A-Za-z0-9])([{JA_CHAR}])", r"\1 \2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    compiled = list(remove_patterns or [])
    kept: list[str] = []
    removed = 0
    for line in text.split("\n"):
        stripped = line.strip()
        if any(p.match(stripped) for p in compiled):
            removed += 1
            continue
        kept.append(stripped)
    if removed:
        flags.append("removed_lines")
    text = "\n".join(kept)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text, flags


def split_units(text: str) -> list[tuple[str, str]]:
    units: list[tuple[str, str]] = []
    paragraphs = re.split(r"\n\s*\n|\n", text)
    for pi, paragraph in enumerate(paragraphs):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        pos = 0
        for m in re.finditer(r"[^。！？!?、，,;：:]+[。！？!?、，,;：:]?", paragraph):
            part = m.group(0).strip()
            if not part:
                continue
            last = part[-1]
            if last in "。！？!?":
                boundary = "sentence_boundary"
            elif last in "、，,;：:":
                boundary = "sub_sentence_boundary"
            elif pi < len(paragraphs) - 1:
                boundary = "paragraph_boundary"
            else:
                boundary = "sentence_boundary"
            units.append((part, boundary))
            pos = m.end()
        if pos < len(paragraph):
            units.append((paragraph[pos:].strip(), "paragraph_boundary"))
    return units or [(text.strip(), "short_as_is")]


def hard_slice_text(text: str, hard_max: int) -> list[str]:
    return [text[i : i + hard_max].strip() for i in range(0, len(text), hard_max) if text[i : i + hard_max].strip()]


def add_overlap(previous: str, current: str, overlap: int) -> str:
    if overlap <= 0 or not previous:
        return current
    prefix = previous[-overlap:].strip()
    if not prefix:
        return current
    return f"{prefix}\n{current}".strip()


def chunk_text(text: str, profile: str, overrides: dict[str, int | None]) -> list[dict[str, Any]]:
    settings = dict(CHUNK_PROFILES[profile])
    for key, value in overrides.items():
        if value is not None:
            settings[key] = value
    if profile == "none":
        stripped = text.strip()
        return [
            {
                "chunk_text": stripped,
                "cut_reason": "no_chunk",
                "forced_slice": False,
                "char_len": len(stripped),
            }
        ]
    target = int(settings["target_chars"] or 750)
    hard_max = int(settings["hard_max"] or max(target * 2, 1))
    overlap = int(settings["overlap"] or 0)
    min_chars = int(settings["min_chars"] or 0)
    if len(text) <= target:
        reason = "short_as_is" if len(text) < min_chars else "sentence_boundary"
        return [{"chunk_text": text, "cut_reason": reason, "forced_slice": False, "char_len": len(text)}]

    chunks: list[dict[str, Any]] = []
    current = ""
    reason = "sentence_boundary"
    previous_without_overlap = ""
    for unit, boundary in split_units(text):
        candidate = f"{current}{unit}" if not current else f"{current}{unit}"
        if current and len(candidate) > target:
            base = current.strip()
            for part in hard_slice_text(base, hard_max):
                forced = len(base) > hard_max
                chunk_body = add_overlap(previous_without_overlap, part, overlap)
                chunks.append(
                    {
                        "chunk_text": chunk_body,
                        "cut_reason": "hard_slice" if forced else reason,
                        "forced_slice": forced,
                        "char_len": len(chunk_body),
                    }
                )
                previous_without_overlap = part
            current = unit
            reason = boundary
        else:
            current = candidate
            reason = boundary
    if current.strip():
        base = current.strip()
        for part in hard_slice_text(base, hard_max):
            forced = len(base) > hard_max
            chunk_body = add_overlap(previous_without_overlap, part, overlap)
            chunks.append(
                {
                    "chunk_text": chunk_body,
                    "cut_reason": "hard_slice" if forced else reason,
                    "forced_slice": forced,
                    "char_len": len(chunk_body),
                }
            )
            previous_without_overlap = part
    return chunks


def script_kind(token: str) -> str:
    if KANJI_RE.search(token):
        return "kanji"
    if KANA_RE.search(token):
        return "kana"
    if ASCII_RE.search(token):
        return "ascii"
    if NUM_RE.search(token):
        return "num"
    return "other"


def split_english_parts(token: str) -> list[str]:
    rough = re.split(r"[_\-/.\s]+", token)
    parts: list[str] = []
    for item in rough:
        parts.extend(p for p in CAMEL_RE.split(item) if p)
    return parts


def regex_tokenize(text: str) -> list[tuple[str, str | None]]:
    out: list[tuple[str, str | None]] = []
    for match in TOKEN_RE.finditer(text):
        tok = match.group(0).strip("._-/")
        if not tok:
            continue
        if ASCII_RE.search(tok):
            out.append((tok.lower(), None))
            for part in split_english_parts(tok):
                part_l = part.lower()
                if len(part_l) >= 2 and part_l != tok.lower():
                    out.append((part_l, None))
        else:
            out.append((tok, None))
    return out


def fugashi_tokenize(text: str) -> list[tuple[str, str | None]]:
    try:
        import fugashi  # type: ignore
    except ImportError as exc:
        raise RuntimeError("tokenizer=fugashi requires the fugashi package; install it or use --tokenizer regex") from exc
    tagger = fugashi.Tagger()
    tokens: list[tuple[str, str | None]] = []
    for word in tagger(text):
        surface = str(word)
        features = str(word.feature).split(",")
        pos = features[0] if features else None
        if pos in {"名詞", "動詞", "形容詞"}:
            tokens.append((surface, pos))
    return tokens


def sudachi_tokenize(text: str, mode_name: str) -> list[tuple[str, str | None]]:
    try:
        from sudachipy import dictionary, tokenizer  # type: ignore
    except ImportError as exc:
        raise RuntimeError(f"tokenizer={mode_name} requires sudachipy and a Sudachi dictionary; install them or use --tokenizer regex") from exc
    mode_map = {
        "sudachi_a": tokenizer.Tokenizer.SplitMode.A,
        "sudachi_b": tokenizer.Tokenizer.SplitMode.B,
        "sudachi_c": tokenizer.Tokenizer.SplitMode.C,
    }
    tok = dictionary.Dictionary().create()
    tokens: list[tuple[str, str | None]] = []
    for m in tok.tokenize(text, mode_map[mode_name]):
        pos = m.part_of_speech()[0]
        if pos in {"名詞", "動詞", "形容詞"}:
            tokens.append((m.surface(), pos))
    return tokens


def tokenize(
    text: str,
    tokenizer_name: str,
    stopwords: set[str],
    use_noun_compounds: bool,
    synonyms: dict[str, list[str]],
) -> list[str]:
    if tokenizer_name == "regex":
        raw = regex_tokenize(text)
    elif tokenizer_name == "fugashi":
        raw = fugashi_tokenize(text)
    elif tokenizer_name.startswith("sudachi_"):
        raw = sudachi_tokenize(text, tokenizer_name)
    else:
        raise ValueError(f"unknown tokenizer: {tokenizer_name}")

    tokens: list[str] = []
    for token, _pos in raw:
        t = unicodedata.normalize("NFKC", token).strip()
        if not t:
            continue
        if ASCII_RE.fullmatch(t):
            t = t.lower()
        if len(t) < 2 and t not in RISKY_KEEP_WORDS:
            continue
        if t in stopwords and t not in RISKY_KEEP_WORDS:
            continue
        tokens.append(t)

    if use_noun_compounds:
        additions: list[str] = []
        for left, right in zip(tokens, tokens[1:]):
            if script_kind(left) == script_kind(right) and script_kind(left) in {"kanji", "kana", "ascii"}:
                combined = left + right
                if 3 <= len(combined) <= 40:
                    additions.append(combined)
        tokens.extend(additions)

    if synonyms:
        present_text = " ".join(tokens)
        for key, values in synonyms.items():
            if key in text or key in present_text:
                tokens.extend(values)

    seen: set[str] = set()
    deduped: list[str] = []
    for token in tokens:
        if token and token not in seen:
            deduped.append(token)
            seen.add(token)
    return deduped


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    idx = min(len(sorted_values) - 1, max(0, math.ceil((pct / 100.0) * len(sorted_values)) - 1))
    return float(sorted_values[idx])


def write_reports(out_df: pd.DataFrame, report_base: Path) -> dict[str, Any]:
    char_lens = [int(x) for x in out_df["char_len"].fillna(0).tolist()]
    token_counts = [int(x) for x in out_df["token_count"].fillna(0).tolist()]
    rows_out = len(out_df)
    forced_count = int(out_df["forced_slice"].astype(bool).sum()) if rows_out else 0
    report = {
        "rows_in": int(out_df["source_row"].nunique()) if "source_row" in out_df else 0,
        "rows_out": rows_out,
        "avg_chunk_chars": float(statistics.mean(char_lens)) if char_lens else 0.0,
        "median_chunk_chars": float(statistics.median(char_lens)) if char_lens else 0.0,
        "p90_chunk_chars": percentile(char_lens, 90),
        "forced_slice_count": forced_count,
        "cut_rate": forced_count / max(rows_out, 1),
        "empty_token_rows": int((out_df["token_count"].fillna(0).astype(int) == 0).sum()) if rows_out else 0,
        "avg_token_count": float(statistics.mean(token_counts)) if token_counts else 0.0,
    }
    report_json = report_base.with_suffix(".lsa_preprocess_report.json")
    top_tokens_csv = report_base.with_suffix(".top_tokens.csv")
    suspicious_csv = report_base.with_suffix(".suspicious_rows.csv")
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    counts: Counter[str] = Counter()
    for value in out_df["lsa_tokens_str"].fillna(""):
        counts.update(str(value).split())
    with open(top_tokens_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["token", "count"])
        writer.writerows(counts.most_common(200))

    suspicious = out_df[(out_df["token_count"].fillna(0).astype(int) == 0) | (out_df["char_len"].fillna(0).astype(int) == 0)]
    suspicious.to_csv(suspicious_csv, index=False, encoding="utf-8-sig")
    report["report_json"] = str(report_json)
    report["top_tokens_csv"] = str(top_tokens_csv)
    report["suspicious_rows_csv"] = str(suspicious_csv)
    return report


def output_report_base(output: Path, report_dir: str | None) -> Path:
    if report_dir:
        p = Path(report_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p / output.stem
    output.parent.mkdir(parents=True, exist_ok=True)
    return output.with_suffix("")


def build_output(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    input_path = Path(args.input)
    df = pd.read_csv(input_path, encoding=args.encoding)
    if args.text_col not in df.columns:
        raise ValueError(f"--text-col not found: {args.text_col}. columns={list(df.columns)}")
    remove_patterns = compile_patterns(read_list_file(args.remove_line_patterns, DEFAULT_REMOVE_PATTERNS))
    stopwords = set(read_list_file(args.ja_stopwords, sorted(DEFAULT_SAFE_STOPWORDS))) - RISKY_KEEP_WORDS
    synonyms = load_synonyms(args.use_synonyms_json, args.max_synonym_terms)
    protected_terms = read_list_file(args.protected_terms, [])

    overrides = {
        "target_chars": args.target_chars,
        "overlap": args.overlap,
        "hard_max": args.hard_max,
        "min_chars": args.min_chars,
    }
    rows: list[dict[str, Any]] = []
    for source_idx, row in df.iterrows():
        original_value = "" if is_missing(row[args.text_col]) else str(row[args.text_col])
        normalized, flags = normalize_text(original_value, remove_patterns)
        if protected_terms:
            flags.append("protected_terms_loaded")
        chunks = chunk_text(normalized, args.chunk_profile, overrides)
        for chunk_idx, chunk in enumerate(chunks):
            chunk_body = chunk["chunk_text"]
            tokens = tokenize(chunk_body, args.tokenizer, stopwords, args.use_noun_compounds, synonyms)
            out = row.to_dict()
            file_value = row[args.file_col] if args.file_col and args.file_col in df.columns else ""
            page_value = row[args.page_col] if args.page_col and args.page_col in df.columns else ""
            if args.chunk_id_col and args.chunk_id_col in df.columns:
                base_id = str(row[args.chunk_id_col])
            else:
                base_id = f"row{source_idx}"
            out.update(
                {
                    "chunk_id_out": f"{base_id}_chunk{chunk_idx}",
                    "source_row": int(source_idx),
                    "file_name_out": "" if is_missing(file_value) else str(file_value),
                    "page_out": "" if is_missing(page_value) else str(page_value),
                    "chunk_index": int(chunk_idx),
                    "text_original": original_value,
                    "text_normalized": normalized,
                    "chunk_text": chunk_body,
                    "lsa_tokens_str": " ".join(tokens),
                    "token_count": len(tokens),
                    "char_len": int(chunk["char_len"]),
                    "cut_reason": chunk["cut_reason"],
                    "forced_slice": bool(chunk["forced_slice"]),
                    "preprocess_flags": ",".join(sorted(set(flags))),
                    "tokenizer_name": args.tokenizer,
                    "chunk_profile": args.chunk_profile,
                    "model_target": MODEL_TARGET,
                }
            )
            rows.append(out)
    out_df = pd.DataFrame(rows)
    required_first = [
        "chunk_id_out",
        "source_row",
        "file_name_out",
        "page_out",
        "chunk_index",
        "text_original",
        "text_normalized",
        "chunk_text",
        "lsa_tokens_str",
        "token_count",
        "char_len",
        "cut_reason",
        "forced_slice",
        "preprocess_flags",
        "tokenizer_name",
        "chunk_profile",
        "model_target",
    ]
    cols = required_first + [c for c in out_df.columns if c not in required_first]
    out_df = out_df[cols]
    report = write_reports(out_df, output_report_base(Path(args.output), args.report_dir))
    return out_df, report


def safe_write_csv(df: pd.DataFrame, output: Path, encoding: str, overwrite: bool, input_path: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    same_path = output.resolve() == input_path.resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(f"output exists; pass --overwrite to replace it: {output}")
    if same_path:
        fd, tmp_name = tempfile.mkstemp(prefix=output.name + ".", suffix=".tmp", dir=str(output.parent))
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            df.to_csv(tmp_path, index=False, encoding=encoding)
            os.replace(tmp_path, output)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
    else:
        df.to_csv(output, index=False, encoding=encoding)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--text-col", required=True)
    parser.add_argument("--file-col")
    parser.add_argument("--page-col")
    parser.add_argument("--chunk-id-col")
    parser.add_argument("--encoding", default="utf-8-sig")
    parser.add_argument("--output-encoding", default="utf-8-sig")
    parser.add_argument("--tokenizer", choices=["regex", "fugashi", "sudachi_a", "sudachi_b", "sudachi_c"], default="regex")
    parser.add_argument("--chunk-profile", choices=["none", "small", "medium", "large"], default="medium")
    parser.add_argument("--target-chars", type=int)
    parser.add_argument("--overlap", type=int)
    parser.add_argument("--hard-max", type=int)
    parser.add_argument("--min-chars", type=int)
    parser.add_argument("--protected-terms", default=default_if_exists("config/protected_terms.txt"))
    parser.add_argument("--remove-line-patterns", default=default_if_exists("config/remove_line_patterns.txt"))
    parser.add_argument("--ja-stopwords", default=default_if_exists("config/safe_ja_stopwords.txt"))
    parser.add_argument("--use-noun-compounds", action="store_true")
    parser.add_argument("--use-synonyms-json")
    parser.add_argument("--max-synonym-terms", type=int, default=30)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--report-dir")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = Path(args.output)
    input_path = Path(args.input)
    out_df, report = build_output(args)
    safe_write_csv(out_df, output, args.output_encoding, args.overwrite, input_path)
    print(json.dumps({"output": str(output), "report": report}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

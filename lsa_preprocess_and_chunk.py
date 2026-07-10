#!/usr/bin/env python3
"""Preprocess CSV text for Japanese TF-IDF + TruncatedSVD(LSA) retrieval."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import importlib.util
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

try:
    from analyze_data import LOGGER  # type: ignore
except ImportError:
    import logging as _logging

    LOGGER = _logging.getLogger("script1_preprocess")
    if not LOGGER.handlers:
        _logging.basicConfig(level=_logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

MODEL_TARGET = "TFIDF_TRUNCATED_SVD_LSA"

CHUNK_PROFILES: dict[str, dict[str, Any]] = {
    "none": {"target_chars": None, "overlap": 0, "hard_max": None, "min_chars": 0},
    "current": {"target_chars": 750, "overlap": 100, "hard_max": 1100, "min_chars": 180, "strategy": "current"},
    "small": {"target_chars": 450, "overlap": 60, "hard_max": 700, "min_chars": 120},
    "medium": {"target_chars": 750, "overlap": 100, "hard_max": 1100, "min_chars": 180},
    "large": {"target_chars": 1050, "overlap": 150, "hard_max": 1500, "min_chars": 250},
    "auto": {"target_chars": None, "overlap": None, "hard_max": None, "min_chars": 0},
    "sentence_boundary_medium": {"target_chars": 750, "overlap": 100, "hard_max": 900, "min_chars": 180, "strategy": "sentence"},
    "paragraph_boundary_medium": {"target_chars": 800, "overlap": 100, "hard_max": 1000, "min_chars": 180, "strategy": "paragraph"},
    "heading_aware": {"target_chars": 750, "overlap": 100, "hard_max": 900, "min_chars": 180, "strategy": "heading"},
    "compact": {"target_chars": 500, "overlap": 65, "hard_max": 650, "min_chars": 120, "strategy": "sentence"},
    "broad": {"target_chars": 1000, "overlap": 150, "hard_max": 1300, "min_chars": 250, "strategy": "sentence"},
}

MORPHOLOGY_PROFILES = {"noun_only", "content_surface", "content_lemma", "surface_plus_lemma"}
PRODUCTION_TOKENIZERS = {
    "mecab_ipadic_utf8_v102",
    "sudachi_small_b",
    "sudachi_small_c",
    "sudachi_core_b",
    "sudachi_core_c",
}

PROTECTED_PATTERN_RE = re.compile(
    r"https?://[^\s<>]+|[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}|"
    r"(?:第\s*\d+\s*条(?:の\s*\d+)?)|(?:\d{4}\s*年度)|(?:\d+(?:\.\d+)?\s*(?:万円|円|ページ))|"
    r"(?:[A-Za-z]+[A-Za-z0-9]*[-/][A-Za-z0-9][A-Za-z0-9./_-]*)"
)

AUTO_TARGET_CHARS_FLOOR = 320
AUTO_TARGET_CHARS_CEILING = 900
AUTO_OVERLAP_FLOOR = 40
AUTO_OVERLAP_CEILING = 120
AUTO_HARD_MAX_FLOOR = 800


def derive_auto_chunk_params(char_lengths: Sequence[int]) -> tuple[int, int, int]:
    """コーパス全体の正規化後テキスト文字数分布から target_chars/overlap/hard_max を導出する。

    my_semantic_chunker.py の derive_chunk_params と同じ思想(統計分布ベース)を、
    文字数ベースで動作する本スクリプトの分割エンジンに適用したもの。
    固定プロファイル(small/medium/large)がコーパス規模と無関係に一律の値を使うのに対し、
    実データのQ50/Q75/Q90に応じて分割粒度を自動調整することで、
    「1文書=1chunkのまま素通りする」「粒度が細かすぎる/粗すぎる」を防ぐ。
    """
    if not char_lengths:
        return AUTO_TARGET_CHARS_FLOOR, AUTO_OVERLAP_FLOOR, AUTO_HARD_MAX_FLOOR
    q75 = percentile(list(char_lengths), 75)
    q90 = percentile(list(char_lengths), 90)
    target_chars = int(round(q75 * 0.9))
    target_chars = max(AUTO_TARGET_CHARS_FLOOR, min(AUTO_TARGET_CHARS_CEILING, target_chars))
    overlap = int(round(target_chars * 0.15))
    overlap = max(AUTO_OVERLAP_FLOOR, min(AUTO_OVERLAP_CEILING, overlap))
    hard_max = int(round(max(q90 * 1.1, target_chars * 1.6)))
    hard_max = max(AUTO_HARD_MAX_FLOOR, hard_max)
    return target_chars, overlap, hard_max

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
HIRAGANA_ONLY_RE = re.compile(r"^[\u3040-\u309fー]+$")
KATAKANA_RE = re.compile(r"[\u30a0-\u30ffー]")
KATAKANA_ONLY_RE = re.compile(r"^[\u30a0-\u30ffー]+$")
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

def load_word_list_excel(path: str | Path | None, default: Sequence[str] | None = None) -> list[str]:
    """Excel(A列, header無し)から用語リストを読み込む。
    """
    if not path:
        return list(default or [])
    p = Path(path)
    if not p.exists():
        return list(default or [])
    try:
        df_words = pd.read_excel(p, usecols=[0], header=None)
        values = (
            df_words[0]
            .dropna()
            .astype(str)
            .map(lambda x: unicodedata.normalize("NFKC", x.strip()))
            .unique()
            .tolist()
        )
        return [v for v in values if v]
    except Exception as exc:
        LOGGER.warning(f"⚠️ 用語リストExcelの読み込みに失敗したためデフォルト値を使用します: {p} ({exc})")
        return list(default or [])

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


def truthy(value: Any) -> bool:
    if is_missing(value):
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"", "0", "false", "n", "no", "off", "nan", "none"}


def load_enabled_excel_column(path: Path, column: str) -> list[str]:
    if not path.exists():
        return []
    try:
        df = pd.read_excel(path)
    except Exception as exc:
        LOGGER.warning(f"⚠️ config Excelを読み込めませんでした: {path} ({exc})")
        return []
    if column not in df.columns:
        return []
    if "enabled" in df.columns:
        df = df[df["enabled"].map(truthy).astype(bool)]
    return [unicodedata.normalize("NFKC", str(v).strip()) for v in df[column].dropna() if str(v).strip()]


def load_config_synonyms(config_dir: str | Path | None, max_terms: int) -> dict[str, list[str]]:
    if not config_dir:
        return {}
    path = Path(config_dir) / "list_synonym.xlsx"
    if not path.exists():
        return {}
    try:
        df = pd.read_excel(path)
    except Exception as exc:
        LOGGER.warning(f"⚠️ synonym configを読み込めませんでした: {path} ({exc})")
        return {}
    if "enabled" in df.columns:
        df = df[df["enabled"].map(truthy).astype(bool)]
    out: dict[str, list[str]] = {}
    for _, row in df.iterrows():
        canonical = unicodedata.normalize("NFKC", str(row.get("canonical", "")).strip())
        variant = unicodedata.normalize("NFKC", str(row.get("variant", "")).strip())
        direction = str(row.get("direction", "query_to_doc")).strip()
        if not canonical or not variant or canonical.lower() == "nan" or variant.lower() == "nan":
            continue
        if direction in {"doc_to_query", "bidirectional"}:
            out.setdefault(canonical, []).append(variant)
        if direction in {"query_to_doc", "bidirectional", ""}:
            out.setdefault(variant, []).append(canonical)
    return {k: v[:max_terms] for k, v in out.items()}


def load_config_replacements(config_dir: str | Path | None) -> list[dict[str, str]]:
    if not config_dir:
        return []
    path = Path(config_dir) / "df_replace.xlsx"
    if not path.exists():
        return []
    try:
        df = pd.read_excel(path)
    except Exception as exc:
        LOGGER.warning(f"⚠️ replacement configを読み込めませんでした: {path} ({exc})")
        return []
    if "enabled" in df.columns:
        df = df[df["enabled"].map(truthy).astype(bool)]
    rows: list[dict[str, str]] = []
    for _, row in df.iterrows():
        pattern = str(row.get("pattern", "")).strip()
        replacement = str(row.get("replacement", "")).strip()
        stage = str(row.get("stage", "preprocess")).strip()
        if not pattern or pattern.lower() == "nan" or stage not in {"", "preprocess"}:
            continue
        rows.append(
            {
                "pattern": unicodedata.normalize("NFKC", pattern),
                "replacement": unicodedata.normalize("NFKC", replacement),
                "match_type": str(row.get("match_type", "literal")).strip() or "literal",
            }
        )
    return rows


def apply_config_replacements(text: str, replacements: Sequence[dict[str, str]]) -> tuple[str, int]:
    changed = 0
    out = text
    for row in replacements:
        pattern = row["pattern"]
        replacement = row["replacement"]
        before = out
        if row.get("match_type") == "regex":
            out = re.sub(pattern, replacement, out)
        else:
            out = out.replace(pattern, replacement)
        if out != before:
            changed += 1
    return out, changed


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


def is_heading(line: str) -> bool:
    value = line.strip()
    if not value or len(value) > 80 or value[-1:] in "。！？!?":
        return False
    return bool(re.match(r"^(?:第\d+[章節款項][ 　]*|\d+(?:[.．-]\d+)*[ 　]|[■◆●○□▼▽]|【.+】|.+[:：]$)", value))


def structural_units(text: str, strategy: str = "sentence") -> list[dict[str, Any]]:
    """Return indivisible sentence/paragraph units with source offsets."""
    units: list[dict[str, Any]] = []
    sentence_no = 0
    paragraph_no = 0
    current_heading = ""
    for line_match in re.finditer(r"[^\n]+", text):
        raw_line = line_match.group(0)
        line = raw_line.strip()
        if not line:
            continue
        heading = is_heading(line)
        if heading:
            current_heading = line
            if strategy == "heading":
                paragraph_no += 1
                continue
        if strategy == "paragraph" and len(line) <= 1000:
            matches = [(line, 0, len(line))]
        else:
            matches = [
                (m.group(0).strip(), m.start(), m.end())
                for m in re.finditer(r"[^。！？!?]+[。！？!?]?", line)
                if m.group(0).strip()
            ]
        for part, local_start, local_end in matches:
            units.append(
                {
                    "text": part,
                    "start": line_match.start() + raw_line.find(line) + local_start,
                    "end": line_match.start() + raw_line.find(line) + local_end,
                    "sentence_no": sentence_no,
                    "paragraph_no": paragraph_no,
                    "heading": current_heading if strategy == "heading" else "",
                    "boundary": "paragraph_boundary" if strategy == "paragraph" else "sentence_boundary",
                }
            )
            sentence_no += 1
        paragraph_no += 1
    return units


def overlap_units(previous: Sequence[dict[str, Any]], overlap_target: int, strategy: str) -> list[dict[str, Any]]:
    if overlap_target <= 0:
        return []
    selected: list[dict[str, Any]] = []
    total = 0
    max_units = 1 if strategy == "paragraph" else 2
    for unit in reversed(previous):
        selected.insert(0, unit)
        total += len(str(unit["text"]))
        if total >= overlap_target or len(selected) >= max_units:
            break
    return selected


def chunk_text(text: str, profile: str, overrides: dict[str, int | None]) -> list[dict[str, Any]]:
    settings = dict(CHUNK_PROFILES[profile])
    for key, value in overrides.items():
        if value is not None:
            settings[key] = value
    if profile == "none":
        stripped = text.strip()
        if not stripped:
            return []
        return [
            {
                "chunk_text": stripped,
                "cut_reason": "no_chunk",
                "forced_slice": False,
                "char_len": len(stripped),
                "source_start": 0,
                "source_end": len(stripped),
                "sentence_start": 0,
                "sentence_end": max(0, len(structural_units(stripped)) - 1),
                "paragraph_start": 0,
                "paragraph_end": max(0, stripped.count("\n")),
                "heading": "",
                "overlap_chars": 0,
            }
        ]
    target = int(settings["target_chars"]) if settings["target_chars"] is not None else 750
    hard_max = int(settings["hard_max"]) if settings["hard_max"] is not None else max(target * 2, 1)
    overlap = int(settings["overlap"]) if settings["overlap"] is not None else 0
    min_chars = int(settings["min_chars"]) if settings["min_chars"] is not None else 0
    text = text.strip()
    if not text:
        return []
    strategy = str(settings.get("strategy") or ("current" if profile in {"small", "medium", "large", "auto", "current"} else "sentence"))
    structural_strategy = "paragraph" if strategy == "paragraph" else "heading" if strategy == "heading" else "sentence"
    if len(text) <= target:
        reason = "short_as_is" if len(text) < min_chars else "sentence_boundary"
        units = structural_units(text, structural_strategy)
        first, last = (units[0], units[-1]) if units else ({"start": 0, "sentence_no": 0, "paragraph_no": 0}, {"end": len(text), "sentence_no": 0, "paragraph_no": 0})
        return [{"chunk_text": text, "cut_reason": reason, "forced_slice": False, "char_len": len(text),
                 "source_start": first["start"], "source_end": last["end"], "sentence_start": first["sentence_no"],
                 "sentence_end": last["sentence_no"], "paragraph_start": first["paragraph_no"],
                 "paragraph_end": last["paragraph_no"], "heading": first.get("heading", ""), "overlap_chars": 0}]

    units = structural_units(text, structural_strategy)
    chunks: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    previous: list[dict[str, Any]] = []

    def emit(body_units: Sequence[dict[str, Any]]) -> None:
        nonlocal previous
        if not body_units:
            return
        overlap_part = overlap_units(previous, overlap, structural_strategy)
        heading = str(body_units[0].get("heading") or "")
        parts = [str(x["text"]) for x in overlap_part]
        if heading and (not parts or parts[0] != heading):
            parts.insert(0, heading)
        parts.extend(str(x["text"]) for x in body_units)
        body = "\n".join(p for p in parts if p).strip()
        overlap_chars = sum(len(str(x["text"])) for x in overlap_part)
        chunks.append({
            "chunk_text": body,
            "cut_reason": str(body_units[-1]["boundary"]),
            "forced_slice": False,
            "char_len": len(body),
            "source_start": int(body_units[0]["start"]),
            "source_end": int(body_units[-1]["end"]),
            "sentence_start": int(body_units[0]["sentence_no"]),
            "sentence_end": int(body_units[-1]["sentence_no"]),
            "paragraph_start": int(body_units[0]["paragraph_no"]),
            "paragraph_end": int(body_units[-1]["paragraph_no"]),
            "heading": heading,
            "overlap_chars": overlap_chars,
            "oversized": len(body) > hard_max,
        })
        previous = list(body_units)

    for unit in units:
        candidate_len = sum(len(str(x["text"])) + 1 for x in current + [unit]) - 1
        if current and candidate_len > target:
            emit(current)
            current = []
        current.append(unit)
        # A single indivisible sentence may exceed hard_max; retain it intact and report oversized.
        if len(str(unit["text"])) > hard_max:
            emit(current)
            current = []
    emit(current)
    return chunks


def script_kind(token: str) -> str:
    if KANJI_RE.search(token):
        return "kanji"
    if HIRAGANA_ONLY_RE.match(token):
        return "hiragana_only"
    if KATAKANA_ONLY_RE.match(token):
        return "katakana"
    if KANA_RE.search(token):
        return "mixed_kana_blob"
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


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def fugashi_tokenize(text: str, dictionary_path: str | None, mecab_options: str = "") -> list[dict[str, Any]]:
    try:
        import fugashi  # type: ignore
    except ImportError as exc:
        raise RuntimeError("MeCab baseline unavailable: fugashi is not installed (regex fallback is disabled)") from exc
    if not dictionary_path:
        raise RuntimeError("MeCab baseline requires explicit --dictionary-path; implicit system dictionary selection is disabled")
    dic = Path(dictionary_path)
    if not dic.exists():
        raise RuntimeError(f"MeCab dictionary path not found: {dic}")
    option = f'-d "{dic}" {mecab_options}'.strip()
    tagger = fugashi.Tagger(option)
    tokens: list[dict[str, Any]] = []
    for word in tagger(text):
        surface = str(word)
        feature = word.feature
        values = str(feature).split(",")
        pos = getattr(feature, "pos1", None) or (values[0] if values else None)
        pos2 = getattr(feature, "pos2", None) or (values[1] if len(values) > 1 else None)
        lemma = getattr(feature, "lemma", None) or getattr(feature, "base", None)
        if not lemma or lemma == "*":
            lemma = values[6] if len(values) > 6 and values[6] != "*" else surface
        tokens.append({"surface": surface, "lemma": lemma, "normalized": lemma, "pos": pos, "pos2": pos2,
                       "oov": bool(getattr(word, "is_unk", False))})
    return tokens


def sudachi_tokenize(text: str, mode_name: str, dictionary_path: str | None = None) -> list[dict[str, Any]]:
    try:
        from sudachipy import dictionary, tokenizer  # type: ignore
    except ImportError as exc:
        raise RuntimeError(f"{mode_name} unavailable: sudachipy is not installed (regex fallback is disabled)") from exc
    split = mode_name.rsplit("_", 1)[-1]
    dict_kind = "small" if "_small_" in mode_name else "core"
    package = f"sudachidict_{dict_kind}"
    if not importlib.util.find_spec(package):
        raise RuntimeError(f"{mode_name} unavailable: {package} is not installed")
    mode_map = {"a": tokenizer.Tokenizer.SplitMode.A, "b": tokenizer.Tokenizer.SplitMode.B, "c": tokenizer.Tokenizer.SplitMode.C}
    kwargs: dict[str, Any] = {"dict": dict_kind}
    if dictionary_path:
        kwargs = {"path": dictionary_path}
    try:
        tok = dictionary.Dictionary(**kwargs).create()
    except TypeError:
        # Compatibility with older SudachiPy; package availability was already checked.
        tok = dictionary.Dictionary().create()
    tokens: list[dict[str, Any]] = []
    for m in tok.tokenize(text, mode_map[split]):
        pos_values = m.part_of_speech()
        tokens.append({"surface": m.surface(), "lemma": m.dictionary_form(), "normalized": m.normalized_form(),
                       "pos": pos_values[0], "pos2": pos_values[1], "oov": bool(m.is_oov())})
    return tokens


def morphology_candidate_metadata(tokenizer_name: str, dictionary_path: str | None = None,
                                  dictionary_version: str | None = None, dictionary_charset: str | None = None,
                                  mecab_options: str = "", morphology_profile: str = "content_lemma") -> dict[str, Any]:
    if tokenizer_name == "mecab_ipadic_utf8_v102":
        blockers = []
        if not importlib.util.find_spec("fugashi"):
            blockers.append("fugashi_not_installed")
        if not dictionary_path:
            blockers.append("mecab_dictionary_path_not_configured")
        elif not Path(dictionary_path).exists():
            blockers.append("mecab_dictionary_path_not_found")
        if dictionary_charset and dictionary_charset.upper().replace("-", "") != "UTF8":
            blockers.append("dictionary_charset_must_be_utf8")
        if dictionary_version and str(dictionary_version) != "102":
            blockers.append("dictionary_version_must_be_102")
        return {"candidate_id": f"{tokenizer_name}_{morphology_profile}", "tokenizer_name": tokenizer_name, "engine": "MeCab", "wrapper": "fugashi",
                "dictionary": "IPA dictionary", "dictionary_path": dictionary_path, "dictionary_version": dictionary_version or "102",
                "dictionary_charset": dictionary_charset or "UTF-8", "split_mode": None, "morphology_profile": morphology_profile,
                "mecab_options": mecab_options, "availability": not blockers, "environment_blockers": blockers,
                "normalization_method": "NFKC + morphology profile", "oov_handling": "surface"}
    if tokenizer_name.startswith("sudachi_"):
        kind = "small" if "_small_" in tokenizer_name else "core"
        package = f"sudachidict_{kind}"
        blockers = []
        if not importlib.util.find_spec("sudachipy"):
            blockers.append("sudachipy_not_installed")
        if not importlib.util.find_spec(package):
            blockers.append(f"{package}_not_installed")
        return {"candidate_id": f"{tokenizer_name}_{morphology_profile}", "tokenizer_name": tokenizer_name, "engine": "SudachiPy", "wrapper": "sudachipy",
                "dictionary": package, "dictionary_path": dictionary_path, "dictionary_version": package_version(package),
                "dictionary_charset": "UTF-8", "split_mode": tokenizer_name[-1].upper(), "morphology_profile": morphology_profile,
                "mecab_options": None, "availability": not blockers, "environment_blockers": blockers,
                "normalization_method": "Sudachi normalized_form", "oov_handling": "surface"}
    return {"candidate_id": tokenizer_name, "tokenizer_name": tokenizer_name, "engine": "compatibility_only", "availability": tokenizer_name == "regex",
            "environment_blockers": [], "morphology_profile": morphology_profile}


def write_candidate_catalog(output_dir: str | Path, mecab_dictionary_path: str | None = None) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    candidates = [
        morphology_candidate_metadata(name, mecab_dictionary_path, "102", "UTF-8", "", "content_lemma")
        for name in sorted(PRODUCTION_TOKENIZERS)
    ]
    (out / "morphology_candidates.json").write_text(json.dumps(candidates, ensure_ascii=False, indent=2), encoding="utf-8")
    profiles = []
    for profile_id in ["current", "sentence_boundary_medium", "paragraph_boundary_medium", "heading_aware", "compact", "broad"]:
        value = CHUNK_PROFILES[profile_id]
        strategy = str(value.get("strategy", "current"))
        profiles.append({
            "profile_id": profile_id, "boundary_strategy": strategy, "target_size": value["target_chars"],
            "max_size": value["hard_max"], "overlap_strategy": "complete paragraph" if strategy == "paragraph" else "complete 1-2 sentences",
            "overlap_target": value["overlap"], "heading_behavior": "replicate into child chunks" if strategy == "heading" else "preserve in source order",
            "paragraph_behavior": "atomic unless oversized" if strategy == "paragraph" else "sentence-aware",
        })
    (out / "chunk_profiles.json").write_text(json.dumps(profiles, ensure_ascii=False, indent=2), encoding="utf-8")


def select_morph_token(item: dict[str, Any], profile: str, keep: bool) -> list[str]:
    surface = str(item.get("surface") or "")
    lemma = str(item.get("normalized") or item.get("lemma") or surface)
    pos, pos2 = str(item.get("pos") or ""), str(item.get("pos2") or "")
    if keep or item.get("oov"):
        return [surface]
    is_noun = pos == "名詞"
    if profile == "noun_only":
        return [surface] if is_noun else []
    if pos not in {"名詞", "動詞", "形容詞"}:
        return []
    if profile == "content_surface":
        return [surface]
    if profile == "surface_plus_lemma":
        return [surface] if lemma == surface else [surface, lemma]
    return [surface if is_noun else lemma]


def tokenize(
    text: str,
    tokenizer_name: str,
    stopwords: set[str],
    use_noun_compounds: bool,
    synonyms: dict[str, list[str]],
    keep_words: set[str] | None = None,
    morphology_profile: str = "content_lemma",
    dictionary_path: str | None = None,
    mecab_options: str = "",
) -> list[str]:
    keep_words = keep_words if keep_words is not None else RISKY_KEEP_WORDS
    if tokenizer_name == "regex":
        raw_compat = regex_tokenize(text)
        raw = [{"surface": s, "lemma": s, "normalized": s, "pos": p, "pos2": None, "oov": False} for s, p in raw_compat]
    elif tokenizer_name in {"fugashi", "mecab_ipadic_utf8_v102"}:
        raw = fugashi_tokenize(text, dictionary_path, mecab_options)
    elif tokenizer_name.startswith("sudachi_"):
        raw = sudachi_tokenize(text, tokenizer_name, dictionary_path)
    else:
        raise ValueError(f"unknown tokenizer: {tokenizer_name}")

    tokens: list[str] = []
    normalized_keep = {unicodedata.normalize("NFKC", x) for x in keep_words}
    protected_present = [x for x in normalized_keep if x and x in text]
    protected_present.extend(m.group(0) for m in PROTECTED_PATTERN_RE.finditer(text))
    for item in raw:
        surface = unicodedata.normalize("NFKC", str(item.get("surface") or "")).strip()
        keep = surface in normalized_keep or any(surface and surface in phrase for phrase in protected_present)
        for token in select_morph_token(item, morphology_profile, keep):
            t = unicodedata.normalize("NFKC", token).strip()
            if not t:
                continue
            t = re.sub(r"[A-Za-z]+", lambda m: m.group(0).lower(), t)
            if len(t) < 2 and not keep:
                continue
            if t in stopwords and not keep:
                continue
            if len(t) >= 4 and HIRAGANA_ONLY_RE.match(t) and not keep:
                continue
            tokens.append(t)

    # Compound keep words and structured identifiers remain searchable in addition to their morphemes.
    for phrase in protected_present:
        protected = re.sub(r"[A-Za-z]+", lambda m: m.group(0).lower(), unicodedata.normalize("NFKC", phrase).strip())
        if protected:
            if re.fullmatch(r"(?:第\s*\d+\s*条(?:の\s*\d+)?|\d{4}\s*年度|\d+(?:\.\d+)?\s*(?:万円|円|ページ))", protected):
                protected = re.sub(r"\s+", "", protected)
            else:
                protected = protected.replace(" ", "_")
            tokens.append(protected)

    if use_noun_compounds:
        additions: list[str] = []
        for left, right in zip(tokens, tokens[1:]):
            if script_kind(left) == script_kind(right) and script_kind(left) in {"kanji", "katakana", "ascii"}:
                combined = left + right
                if 3 <= len(combined) <= 40:
                    additions.append(combined)
        tokens.extend(additions)

    if synonyms:
        present_text = " ".join(tokens)
        for key, values in synonyms.items():
            if key in text or key in present_text:
                tokens.extend(values)

    return [token for token in tokens if token]


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    idx = min(len(sorted_values) - 1, max(0, math.ceil((pct / 100.0) * len(sorted_values)) - 1))
    return float(sorted_values[idx])


def write_reports(out_df: pd.DataFrame, report_base: Path, extra_fields: dict[str, Any] | None = None) -> dict[str, Any]:
    required_columns = ["char_len", "token_count", "forced_slice", "lsa_tokens_str"]
    missing_columns = [c for c in required_columns if c not in out_df.columns]
    if missing_columns:
        raise ValueError(
            f"write_reports: 入力DataFrameに必要な列が存在しません: {missing_columns}. "
            f"実際の列: {list(out_df.columns)}"
        )

    char_lens = [int(x) for x in out_df["char_len"].fillna(0).tolist()]
    token_counts = [int(x) for x in out_df["token_count"].fillna(0).tolist()]
    rows_out = len(out_df)
    forced_count = int(out_df["forced_slice"].astype(bool).sum()) if rows_out else 0
    all_token_lens: list[int] = []
    for value in out_df["lsa_tokens_str"].fillna(""):
        for tok in str(value).split():
            all_token_lens.append(len(tok))
    avg_token_char_len = float(statistics.mean(all_token_lens)) if all_token_lens else 0.0
    long_token_ratio = (
        sum(1 for x in all_token_lens if x >= 8) / max(len(all_token_lens), 1) if all_token_lens else 0.0
    )
    unsegmented_warning = avg_token_char_len >= 6.0 or long_token_ratio >= 0.15
    unique_tokens: set[str] = set()
    for value in out_df["lsa_tokens_str"].fillna(""):
        unique_tokens.update(str(value).split())
    unique_token_count = len(unique_tokens)
    normalized_chunks = out_df.get("chunk_text", pd.Series(dtype=str)).fillna("").astype(str).map(lambda x: re.sub(r"\s+", "", x))
    duplicate_count = int(normalized_chunks.duplicated(keep=False).sum()) if rows_out else 0
    near_duplicate_count = 0
    if rows_out:
        previous_by_source: dict[Any, set[str]] = {}
        for _, row in out_df.iterrows():
            terms = set(str(row.get("lsa_tokens_str", "")).split())
            previous = previous_by_source.get(row.get("source_row"))
            if previous and terms and len(previous & terms) / max(len(previous | terms), 1) >= 0.9:
                near_duplicate_count += 1
            previous_by_source[row.get("source_row")] = terms
    overlap_total = int(pd.to_numeric(out_df.get("overlap_chars", 0), errors="coerce").fillna(0).sum()) if rows_out else 0

    report = {
        "rows_in": int(out_df["source_row"].nunique()) if "source_row" in out_df else 0,
        "rows_out": rows_out,
        "avg_chunk_chars": float(statistics.mean(char_lens)) if char_lens else 0.0,
        "median_chunk_chars": float(statistics.median(char_lens)) if char_lens else 0.0,
        "p90_chunk_chars": percentile(char_lens, 90),
        "p10_chunk_chars": percentile(char_lens, 10),
        "min_chunk_chars": min(char_lens) if char_lens else 0,
        "max_chunk_chars": max(char_lens) if char_lens else 0,
        "forced_slice_count": forced_count,
        "cut_rate": forced_count / max(rows_out, 1),
        "empty_token_rows": int((out_df["token_count"].fillna(0).astype(int) == 0).sum()) if rows_out else 0,
        "avg_token_count": float(statistics.mean(token_counts)) if token_counts else 0.0,
        "median_token_count": float(statistics.median(token_counts)) if token_counts else 0.0,
        "token_count_q25": percentile(token_counts, 25),
        "token_count_q50": percentile(token_counts, 50),
        "token_count_q75": percentile(token_counts, 75),
        "token_count_q90": percentile(token_counts, 90),
        "unique_token_count": unique_token_count,
        "avg_token_char_len": avg_token_char_len,
        "long_token_ratio_ge8chars": long_token_ratio,
        "unsegmented_regex_tokenizer_warning": unsegmented_warning,
        "document_count": int(out_df["file_name_out"].nunique()) if "file_name_out" in out_df else 0,
        "page_count": int(out_df[["file_name_out", "page_out"]].drop_duplicates().shape[0]) if {"file_name_out", "page_out"}.issubset(out_df.columns) else 0,
        "chunk_count": rows_out,
        "short_chunk_count": int((out_df["char_len"] < 120).sum()) if rows_out else 0,
        "oversized_chunk_count": int(out_df.get("oversized", pd.Series(False, index=out_df.index)).fillna(False).astype(bool).sum()) if rows_out else 0,
        "sentence_split_count": int((out_df.get("cut_reason", "") == "sentence_boundary").sum()) if rows_out else 0,
        "paragraph_split_count": int((out_df.get("cut_reason", "") == "paragraph_boundary").sum()) if rows_out else 0,
        "heading_only_chunk_count": int(sum(bool(str(r.get("heading", ""))) and str(r.get("chunk_text", "")).strip() == str(r.get("heading", "")).strip() for _, r in out_df.iterrows())),
        "overlap_chars_total": overlap_total,
        "overlap_ratio": overlap_total / max(sum(char_lens), 1),
        "duplicate_chunk_rate": duplicate_count / max(rows_out, 1),
        "near_duplicate_chunk_rate": near_duplicate_count / max(rows_out, 1),
        "chunks_per_page": rows_out / max(int(out_df[["file_name_out", "page_out"]].drop_duplicates().shape[0]), 1) if {"file_name_out", "page_out"}.issubset(out_df.columns) else None,
        "chunks_per_document": rows_out / max(int(out_df["file_name_out"].nunique()), 1) if "file_name_out" in out_df else None,
    }
    if extra_fields:
        report.update(extra_fields)
    if unsegmented_warning:
        LOGGER.warning(
            "WARNING: regexトークナイザの平均トークン文字長が異常に長い"
            f"(avg={avg_token_char_len:.1f}, long_token_ratio={long_token_ratio:.2f})。"
            " 入力テキストがMeCab等で事前分かち書きされていない可能性が高く、"
            "文単位の塊がそのままTF-IDF語彙に混入しています。"
            " --tokenizer fugashi または事前分かち書き済み入力の使用を検討してください。"
        )
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
    quality_dir = report_base.parent
    tokenizer_quality = pd.DataFrame([{
        "candidate_id": report.get("candidate_id"), "token_count": sum(token_counts),
        "unique_token_count": unique_token_count, "average_tokens_per_chunk": report["avg_token_count"],
        "median_tokens_per_chunk": report["median_token_count"], "empty_chunk_count": report["empty_token_rows"],
        "average_token_chars": avg_token_char_len, "processing_time_seconds": report.get("tokenizer_processing_time_seconds"),
    }])
    tokenizer_quality.to_csv(quality_dir / "tokenizer_quality.csv", index=False, encoding="utf-8-sig")
    chunk_quality_fields = [
        "document_count", "page_count", "chunk_count", "avg_chunk_chars", "median_chunk_chars", "p10_chunk_chars",
        "p90_chunk_chars", "min_chunk_chars", "max_chunk_chars", "avg_token_count", "median_token_count", "empty_token_rows",
        "short_chunk_count", "oversized_chunk_count", "sentence_split_count", "paragraph_split_count", "heading_only_chunk_count",
        "overlap_chars_total", "overlap_ratio", "duplicate_chunk_rate", "near_duplicate_chunk_rate", "chunks_per_page", "chunks_per_document",
    ]
    pd.DataFrame([{**{"profile_id": report.get("effective_chunk_profile")}, **{k: report.get(k) for k in chunk_quality_fields}}]).to_csv(
        quality_dir / "chunk_quality.csv", index=False, encoding="utf-8-sig"
    )
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
    config_dir = Path(args.config_dir) if args.config_dir else None
    if config_dir:
        protected_path = config_dir / "protected_terms.txt"
        remove_path = config_dir / "remove_line_patterns.txt"
        stopword_path = config_dir / "safe_ja_stopwords.txt"
        if protected_path.exists():
            args.protected_terms = str(protected_path)
        if remove_path.exists():
            args.remove_line_patterns = str(remove_path)
        if stopword_path.exists():
            args.ja_stopwords = str(stopword_path)
    remove_patterns = compile_patterns(read_list_file(args.remove_line_patterns, DEFAULT_REMOVE_PATTERNS))
    keep_words = set(load_word_list_excel(args.keep_words_excel, sorted(RISKY_KEEP_WORDS)))
    if config_dir:
        keep_words.update(load_enabled_excel_column(config_dir / "list_keep.xlsx", "term"))
    stopword_base = (
        load_word_list_excel(args.stopwords_excel, None)
        if args.stopwords_excel
        else read_list_file(args.ja_stopwords, sorted(DEFAULT_SAFE_STOPWORDS))
    )
    if config_dir:
        stopword_base = list(stopword_base) + load_enabled_excel_column(config_dir / "list_stopword.xlsx", "term")
    stopwords = set(stopword_base) - keep_words

    synonyms = load_synonyms(args.use_synonyms_json, args.max_synonym_terms)
    if config_dir:
        for key, values in load_config_synonyms(config_dir, args.max_synonym_terms).items():
            synonyms.setdefault(key, [])
            synonyms[key].extend(v for v in values if v not in synonyms[key])
            synonyms[key] = synonyms[key][: args.max_synonym_terms]
    replacements = load_config_replacements(config_dir)
    protected_terms = read_list_file(args.protected_terms, [])
    keep_words.update(unicodedata.normalize("NFKC", x) for x in protected_terms)
    candidate_meta = morphology_candidate_metadata(
        args.tokenizer, args.dictionary_path, args.dictionary_version, args.dictionary_charset,
        args.mecab_options, args.morphology_profile,
    )
    if args.tokenizer in PRODUCTION_TOKENIZERS and not candidate_meta["availability"]:
        raise RuntimeError("tokenizer candidate unavailable: " + ", ".join(candidate_meta["environment_blockers"]))

    overrides = {
        "target_chars": args.target_chars,
        "overlap": args.overlap,
        "hard_max": args.hard_max,
        "min_chars": args.min_chars,
    }
    normalized_cache: list[tuple[str, str, list[str]]] = []
    for source_idx, row in df.iterrows():
        original_value = "" if is_missing(row[args.text_col]) else str(row[args.text_col])
        replacement_input, replacement_count = apply_config_replacements(original_value, replacements)
        normalized, flags = normalize_text(replacement_input, remove_patterns)
        if protected_terms:
            flags.append("protected_terms_loaded")
        if replacement_count:
            flags.append("config_replaced")
        normalized_cache.append((original_value, normalized, flags))

    if args.chunk_profile == "auto":
        char_lengths = [len(normalized) for _, normalized, _ in normalized_cache if normalized]
        auto_target, auto_overlap, auto_hard_max = derive_auto_chunk_params(char_lengths)
        # CLIで明示的に上書き指定された値(--target-chars等)は自動導出値より優先する
        if overrides["target_chars"] is None:
            overrides["target_chars"] = auto_target
        if overrides["overlap"] is None:
            overrides["overlap"] = auto_overlap
        if overrides["hard_max"] is None:
            overrides["hard_max"] = auto_hard_max

    rows: list[dict[str, Any]] = []
    import time
    tokenize_seconds = 0.0
    for source_idx, (original_value, normalized, flags) in zip(df.index, normalized_cache):
        # 修正: auto プロファイルの統計事前算出のため正規化を先出ししたが、
        # 元の row (ファイル名・ページ番号等の付随カラム参照用) は df.loc で引き直す。
        row = df.loc[source_idx]
        chunks = chunk_text(normalized, args.chunk_profile, overrides)
        for chunk_idx, chunk in enumerate(chunks):
            chunk_body = chunk["chunk_text"]
            token_started = time.perf_counter()
            tokens = tokenize(chunk_body, args.tokenizer, stopwords, args.use_noun_compounds, synonyms, keep_words=keep_words,
                              morphology_profile=args.morphology_profile, dictionary_path=args.dictionary_path,
                              mecab_options=args.mecab_options)
            tokenize_seconds += time.perf_counter() - token_started
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
                    "source_char_start": int(chunk.get("source_start", 0)),
                    "source_char_end": int(chunk.get("source_end", len(chunk_body))),
                    "source_sentence_start": int(chunk.get("sentence_start", 0)),
                    "source_sentence_end": int(chunk.get("sentence_end", 0)),
                    "source_paragraph_start": int(chunk.get("paragraph_start", 0)),
                    "source_paragraph_end": int(chunk.get("paragraph_end", 0)),
                    "source_heading": str(chunk.get("heading", "")),
                    "overlap_chars": int(chunk.get("overlap_chars", 0)),
                    "oversized": bool(chunk.get("oversized", False)),
                    "preprocess_flags": ",".join(sorted(set(flags))),
                    "tokenizer_name": args.tokenizer,
                    "morphology_profile": args.morphology_profile,
                    "dictionary_name": candidate_meta.get("dictionary"),
                    "dictionary_version": candidate_meta.get("dictionary_version"),
                    "dictionary_charset": candidate_meta.get("dictionary_charset"),
                    "dictionary_path": candidate_meta.get("dictionary_path"),
                    "split_mode": candidate_meta.get("split_mode"),
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
        "source_char_start", "source_char_end", "source_sentence_start", "source_sentence_end",
        "source_paragraph_start", "source_paragraph_end", "source_heading", "overlap_chars", "oversized",
        "preprocess_flags",
        "tokenizer_name",
        "morphology_profile", "dictionary_name", "dictionary_version", "dictionary_charset", "dictionary_path", "split_mode",
        "chunk_profile",
        "model_target",
    ]
    cols = required_first + [c for c in out_df.columns if c not in required_first]
    out_df = out_df[cols]
    effective_fields = {
        "input_path": str(input_path),
        "output_path": str(Path(args.output)),
        "text_col": args.text_col,
        "config_dir": str(config_dir) if config_dir else None,
        "effective_chunk_profile": args.chunk_profile,
        "effective_target_chars": overrides["target_chars"],
        "effective_overlap": overrides["overlap"],
        "effective_hard_max": overrides["hard_max"],
        "config_keep_terms": len(keep_words),
        "config_stopwords": len(stopwords),
        "config_synonym_keys": len(synonyms),
        "config_replacements": len(replacements),
        "candidate_id": candidate_meta.get("candidate_id"),
        "tokenizer_processing_time_seconds": tokenize_seconds,
        "environment_blockers": candidate_meta.get("environment_blockers", []),
    }
    report = write_reports(
        out_df,
        output_report_base(Path(args.output), args.report_dir),
        extra_fields=effective_fields,
    )
    report_dir = output_report_base(Path(args.output), args.report_dir).parent
    profile_snapshot = {
        "unicode_normalization": "NFKC", "lowercase_policy": "ASCII letters lowercased in both documents and queries",
        "numeric_policy": "retain; protect year/article/page/money/model identifiers",
        "punctuation_policy": "exclude standalone symbols; retain protected internal hyphen/slash",
        "protected_patterns": ["URL", "email", "model_number", "article_number", "fiscal_year", "page", "money"],
        "pos_policy": args.morphology_profile, "lemma_policy": "nouns surface; verbs/adjectives lemma for content_lemma",
        "stopwords_path": args.ja_stopwords, "keep_words_path": args.keep_words_excel or args.protected_terms,
        "stopwords": sorted(stopwords), "keep_words": sorted(keep_words), "candidate": candidate_meta,
        "synonyms": synonyms, "replacements": replacements,
        "remove_line_patterns": [p.pattern for p in remove_patterns],
        "use_noun_compounds": bool(args.use_noun_compounds),
        "token_join_policy": "single ASCII space",
    }
    (report_dir / "preprocessing_profile.json").write_text(json.dumps(profile_snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    write_candidate_catalog(report_dir, args.dictionary_path)
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
    parser.add_argument("--config-dir")
    parser.add_argument("--encoding", default="utf-8-sig")
    parser.add_argument("--output-encoding", default="utf-8-sig")
    parser.add_argument(
        "--tokenizer",
        choices=["regex", "fugashi", "mecab_ipadic_utf8_v102", "sudachi_small_b", "sudachi_small_c", "sudachi_core_b", "sudachi_core_c"],
        default="mecab_ipadic_utf8_v102",
        help="regex/fugashi are compatibility-only; production tuning candidates use explicit engine+dictionary ids",
    )
    parser.add_argument("--dictionary-path", help="required explicit IPA dictionary directory for the MeCab baseline")
    parser.add_argument("--dictionary-version", default="102")
    parser.add_argument("--dictionary-charset", default="UTF-8")
    parser.add_argument("--mecab-options", default="")
    parser.add_argument("--morphology-profile", choices=sorted(MORPHOLOGY_PROFILES), default="content_lemma")
    parser.add_argument("--chunk-profile", choices=list(CHUNK_PROFILES), default="current")
    parser.add_argument("--target-chars", type=int)
    parser.add_argument("--overlap", type=int)
    parser.add_argument("--hard-max", type=int)
    parser.add_argument("--min-chars", type=int)
    parser.add_argument("--protected-terms", default=default_if_exists("config/protected_terms.txt"))
    parser.add_argument("--remove-line-patterns", default=default_if_exists("config/remove_line_patterns.txt"))
    parser.add_argument("--ja-stopwords", default=default_if_exists("config/safe_ja_stopwords.txt"))
    parser.add_argument(
        "--stopwords-excel",
        default=default_if_exists(r"C:\project\document_viewer\_data_assets\model_my_project\step1\list_stopword.xlsx"),
    )
    parser.add_argument(
        "--keep-words-excel",
        default=default_if_exists(r"C:\project\document_viewer\_data_assets\model_my_project\step1\list_keep.xlsx"),
    )
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

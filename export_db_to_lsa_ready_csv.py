#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
既存SQLite DBから、TF-IDF + LSA用のCSVを作る。

前提:
- inspect_existing_db.py で table/text_col/metadata_col を確認済み
- 既存モデル本体は変えない
- 出力CSVの search_text を既存TF-IDF+LSAに渡す

例:
  python tools/export_db_to_lsa_ready_csv.py \
    --db path/to/app.db \
    --table chunks \
    --text-col text \
    --metadata-col metadata \
    --output data/lsa_ready_from_db.csv
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


DEFAULT_SYNONYMS = {
    "ログイン": ["サインイン", "ログオン", "認証"],
    "パスワード": ["暗証番号", "認証情報", "再設定"],
    "請求書": ["インボイス", "invoice", "請求"],
    "支払い": ["決済", "入金", "振込", "精算"],
    "契約": ["契約書", "締結", "更新", "解約"],
    "退職": ["退社", "離職", "辞める"],
    "勤務時間": ["労働時間", "就業時間", "始業", "終業"],
    "休憩": ["休憩時間", "昼休み"],
    "有給": ["有給休暇", "年休", "休暇"],
    "経費": ["精算", "経費精算", "立替", "領収書"],
    "申請": ["申し込み", "届出", "手続き", "依頼"],
    "承認": ["許可", "確認", "決裁", "稟議"],
}


REMOVE_PATTERNS = [
    r"^\s*$",
    r"^\s*\d+\s*$",
    r"^\s*-\s*\d+\s*-\s*$",
    r"^\s*Page\s+\d+\s*(of\s+\d+)?\s*$",
    r"^\s*Confidential\s*$",
    r"^\s*Copyright.*$",
    r"^\s*©.*$",
    r"^\s*目次\s*$",
]


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""

    text = str(value)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u3000", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"([A-Za-z])-\n([A-Za-z])", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def safe_json_loads(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value

    if not isinstance(value, str):
        return {}

    value = value.strip()
    if not value:
        return {}

    try:
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
        return {}
    except Exception:
        return {}


def clean_file_stem(file_name: str) -> str:
    file_name = normalize_text(file_name)
    file_name = re.sub(r"\.[A-Za-z0-9]+$", "", file_name)
    file_name = re.sub(r"[_\-]+", " ", file_name)
    file_name = re.sub(r"\s+", " ", file_name)
    return file_name.strip()


def infer_file_type(file_name: str) -> str:
    lower = str(file_name).lower()
    if lower.endswith(".pdf"):
        return "pdf"
    if lower.endswith(".pptx") or lower.endswith(".ppt"):
        return "pptx"
    if lower.endswith(".docx") or lower.endswith(".doc"):
        return "docx"
    if lower.endswith(".xlsx") or lower.endswith(".csv"):
        return "table"
    return "unknown"


def remove_noise_lines(text: str) -> str:
    compiled = [re.compile(p, re.IGNORECASE) for p in REMOVE_PATTERNS]
    lines = []

    for line in normalize_text(text).splitlines():
        line = line.strip()
        if not line:
            continue

        should_remove = any(p.match(line) for p in compiled)
        if not should_remove:
            lines.append(line)

    return "\n".join(lines).strip()


def extract_heading(text: str) -> str:
    lines = [x.strip() for x in normalize_text(text).splitlines() if x.strip()]
    if not lines:
        return ""

    for line in lines[:5]:
        if len(line) <= 80:
            return line

    return ""


def expand_synonyms(text: str, synonyms: Dict[str, List[str]]) -> str:
    additions = []

    for key, values in synonyms.items():
        if key in text:
            additions.append(key)
            additions.extend(values)

    seen = set()
    out = []
    for x in additions:
        if x not in seen:
            out.append(x)
            seen.add(x)

    return " ".join(out)


def get_meta_value(meta: Dict[str, Any], keys: List[str], default: str = "") -> str:
    for k in keys:
        if k in meta and meta[k] not in [None, ""]:
            return str(meta[k])
    return default


def build_search_text(
    file_name: str,
    page: str,
    heading: str,
    chunk_text: str,
    synonym_weight: int,
    filename_weight: int,
) -> str:
    file_stem = clean_file_stem(file_name)
    file_type = infer_file_type(file_name)

    synonym_text = expand_synonyms(
        " ".join([file_stem, heading, chunk_text]),
        DEFAULT_SYNONYMS,
    )

    file_part = " ".join([file_stem] * max(1, filename_weight))
    synonym_part = " ".join([synonym_text] * max(0, synonym_weight))

    search_text = f"""
文書名 {file_part}
ファイル種別 {file_type}
ページ {page}
見出し {heading}
関連語 {synonym_part}
本文
{chunk_text}
""".strip()

    return normalize_text(search_text)


def export_sqlite_to_csv(args) -> None:
    db_path = Path(args.db)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    query = f'SELECT * FROM "{args.table}"'
    df = pd.read_sql_query(query, conn)
    conn.close()

    if args.text_col not in df.columns:
        raise ValueError(f"text_col not found: {args.text_col}. columns={list(df.columns)}")

    if args.metadata_col and args.metadata_col not in df.columns:
        raise ValueError(f"metadata_col not found: {args.metadata_col}. columns={list(df.columns)}")

    records = []

    for idx, row in df.iterrows():
        raw_text = normalize_text(row[args.text_col])
        chunk_text = remove_noise_lines(raw_text)

        if len(chunk_text) < args.min_chars:
            continue

        meta = {}
        if args.metadata_col:
            meta = safe_json_loads(row[args.metadata_col])

        file_name = (
            get_meta_value(meta, ["file_name", "filename", "source", "source_file", "document_name"])
            or str(row[args.file_col]) if args.file_col and args.file_col in df.columns else ""
        )

        page = (
            get_meta_value(meta, ["page", "page_number", "page_no", "slide", "slide_number"])
            or str(row[args.page_col]) if args.page_col and args.page_col in df.columns else ""
        )

        chunk_id = (
            get_meta_value(meta, ["chunk_id", "id"])
            or str(row[args.id_col]) if args.id_col and args.id_col in df.columns else f"row_{idx}"
        )

        heading = get_meta_value(meta, ["heading", "title", "section"]) or extract_heading(chunk_text)
        file_type = infer_file_type(file_name)

        search_text = build_search_text(
            file_name=file_name,
            page=page,
            heading=heading,
            chunk_text=chunk_text,
            synonym_weight=args.synonym_weight,
            filename_weight=args.filename_weight,
        )

        metadata_out = dict(meta)
        metadata_out.update(
            {
                "file_name": file_name,
                "page": page,
                "file_type": file_type,
                "heading": heading,
                "source_table": args.table,
                "source_db": str(db_path),
                "source_row": int(idx),
            }
        )

        records.append(
            {
                "chunk_id": chunk_id,
                "source_row": idx,
                "file_name": file_name,
                "file_type": file_type,
                "page": page,
                "heading": heading,
                "chunk_text": chunk_text,
                "search_text": search_text,
                "metadata_json": json.dumps(metadata_out, ensure_ascii=False),
                "char_len": len(chunk_text),
            }
        )

    out = pd.DataFrame(records)
    out.to_csv(output_path, index=False, encoding=args.output_encoding)

    summary = {
        "db": str(db_path),
        "table": args.table,
        "input_rows": int(len(df)),
        "output_rows": int(len(out)),
        "output": str(output_path),
        "text_col": args.text_col,
        "metadata_col": args.metadata_col,
        "file_col": args.file_col,
        "page_col": args.page_col,
        "id_col": args.id_col,
        "avg_chars": float(out["char_len"].mean()) if len(out) else 0,
        "file_type_counts": out["file_type"].value_counts().to_dict() if len(out) else {},
    }

    with open(output_path.with_suffix(".summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Export completed.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--table", required=True)
    parser.add_argument("--text-col", required=True)
    parser.add_argument("--metadata-col", default=None)
    parser.add_argument("--id-col", default=None)
    parser.add_argument("--file-col", default=None)
    parser.add_argument("--page-col", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-encoding", default="utf-8-sig")
    parser.add_argument("--min-chars", type=int, default=30)
    parser.add_argument("--filename-weight", type=int, default=2)
    parser.add_argument("--synonym-weight", type=int, default=1)

    args = parser.parse_args()
    export_sqlite_to_csv(args)


if __name__ == "__main__":
    main()
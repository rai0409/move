#!/usr/bin/env python3
"""
Convert lsa_preprocess_and_chunk.py output into make_ce_v1.py df0.csv format.

Confirmed make_ce mapping:
  file name : name_org
  page      : pageno
  text      : text
  id        : chunk_id

Default input columns from lsa_preprocess_and_chunk.py:
  chunk_id_out
  file_name_out
  page_out
  chunk_text

Default output:
  {base_dir}/model_{craw_name}/step0/df0.csv
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Sequence

import pandas as pd


MAKE_CE_COLUMNS = [
    "id",
    "set_dir",
    "name_org",
    "path_org",
    "path_cpy",
    "pagenum",
    "filetextdata",
    "filehash",
    "fileinfo",
    "metainfo",
    "cr_files_id",
    "pageno",
    "text",
    "textdata_yomi",
    "textdata_tess",
    "textdata_abbyy",
    "timestamp",
    "len",
    "original_text",
    "wakati_text",
    "形態素数",
    "一文字形態素数",
    "一文字形態素数割合",
    "chunk_id",
]


def _stable_hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8", errors="ignore")).hexdigest()


def convert_lsa_ready_to_make_ce_df0(args: argparse.Namespace) -> pd.DataFrame:
    src = Path(args.input)
    if not src.exists():
        raise FileNotFoundError(f"input not found: {src}")

    df = pd.read_csv(src, encoding=args.encoding)

    required = [args.chunk_id_col, args.file_col, args.page_col, args.text_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"missing required input columns: {missing}; available={list(df.columns)}")

    out = pd.DataFrame()

    text = df[args.text_col].fillna("").astype(str)
    file_name = df[args.file_col].fillna("").astype(str)
    page = pd.to_numeric(df[args.page_col], errors="coerce").fillna(0).astype(int)
    chunk_id = df[args.chunk_id_col].fillna("").astype(str)

    if (chunk_id == "").any():
        chunk_id = [
            cid if cid else f"{fname}:p{pg}:r{i}"
            for i, (cid, fname, pg) in enumerate(zip(chunk_id, file_name, page))
        ]
        chunk_id = pd.Series(chunk_id)

    out["id"] = range(1, len(df) + 1)
    out["set_dir"] = args.craw_name
    out["name_org"] = file_name
    out["path_org"] = file_name
    out["path_cpy"] = file_name
    out["pagenum"] = page
    out["filetextdata"] = text
    out["filehash"] = [
        _stable_hash(f"{fname}|{pg}|{cid}") for fname, pg, cid in zip(file_name, page, chunk_id)
    ]
    out["fileinfo"] = ""
    out["metainfo"] = ""
    out["cr_files_id"] = out["id"]
    out["pageno"] = page
    out["text"] = text
    out["textdata_yomi"] = ""
    out["textdata_tess"] = text
    out["textdata_abbyy"] = ""
    out["timestamp"] = pd.Timestamp.now().isoformat()
    out["len"] = text.str.len()
    out["original_text"] = text
    out["wakati_text"] = text
    out["形態素数"] = 0
    out["一文字形態素数"] = 0
    out["一文字形態素数割合"] = 0.0
    out["chunk_id"] = chunk_id

    out = out[MAKE_CE_COLUMNS]

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(args.base_dir) / f"model_{args.craw_name}" / "step0" / "df0.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False, encoding=args.output_encoding)

    report = {
        "input": str(src),
        "output": str(output_path),
        "rows": int(len(out)),
        "columns": list(out.columns),
        "mapping": {
            args.chunk_id_col: "chunk_id",
            args.file_col: "name_org",
            args.page_col: "pageno",
            args.text_col: "text",
        },
    }
    report_path = output_path.with_suffix(".convert_report.json")
    report_path.write_text(pd.Series(report).to_json(force_ascii=False, indent=2), encoding="utf-8")

    return out


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="lsa_ready.csv from lsa_preprocess_and_chunk.py")
    p.add_argument("--output", default=None, help="output df0.csv path")
    p.add_argument("--base-dir", default=r"C:\project\document_viewer\_data_assets")
    p.add_argument("--craw-name", required=True)
    p.add_argument("--chunk-id-col", default="chunk_id_out")
    p.add_argument("--file-col", default="file_name_out")
    p.add_argument("--page-col", default="page_out")
    p.add_argument("--text-col", default="chunk_text")
    p.add_argument("--encoding", default="utf-8-sig")
    p.add_argument("--output-encoding", default="utf-8-sig")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    out = convert_lsa_ready_to_make_ce_df0(args)
    print({"rows": len(out), "output_columns": list(out.columns)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
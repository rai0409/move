#!/usr/bin/env python3
"""
Backend adapter for existing make_ce_v1.py.

Flow:
  lsa_ready.csv
  -> convert_lsa_ready_to_make_ce_df0.py
  -> make_ce_v1.py --run_type space
  -> make_ce_v1.py --run_type vectors
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from convert_lsa_ready_to_make_ce_df0 import convert_lsa_ready_to_make_ce_df0, parse_args as parse_convert_args


def run_command(cmd: list[str], cwd: str | None = None) -> None:
    print("[make_ce_backend] RUN:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(f"command failed with exit code {result.returncode}: {' '.join(cmd)}")


def build_make_ce_vectors(args: argparse.Namespace) -> dict[str, Any]:
    base_dir = Path(args.base_dir)
    make_ce_script = Path(args.make_ce_script)

    if not make_ce_script.exists():
        raise FileNotFoundError(f"make_ce_v1.py not found: {make_ce_script}")

    model_dir = base_dir / f"model_{args.craw_name}"
    step0_dir = model_dir / "step0"
    df0_path = step0_dir / "df0.csv"
    step0_dir.mkdir(parents=True, exist_ok=True)
    print(
        "[make_ce_backend] convert input="
        f"{args.lsa_ready} output={df0_path} text_col={args.text_col} "
        f"id_col={args.chunk_id_col} file_col={args.file_col} page_col={args.page_col}"
    )

    convert_args = parse_convert_args(
        [
            "--input",
            args.lsa_ready,
            "--output",
            str(df0_path),
            "--base-dir",
            str(base_dir),
            "--craw-name",
            args.craw_name,
            "--chunk-id-col",
            args.chunk_id_col,
            "--file-col",
            args.file_col,
            "--page-col",
            args.page_col,
            "--text-col",
            args.text_col,
            "--display-text-col",
            args.display_text_col,
            "--docid-col",
            args.docid_col,
        ]
    )
    df0 = convert_lsa_ready_to_make_ce_df0(convert_args)
    profile_source = Path(args.preprocessing_profile) if args.preprocessing_profile else Path(args.lsa_ready).parent / "preprocessing_profile.json"
    if profile_source.exists():
        shutil.copyfile(profile_source, model_dir / "preprocessing_profile.json")

    python_exe = args.python_exe or sys.executable

    common = [
        python_exe,
        str(make_ce_script),
        "--base_dir",
        str(base_dir),
        "--craw_name",
        args.craw_name,
        "--chunk_on",
        "n",
    ]

    if not args.skip_space:
        run_command(common + ["--run_type", "space"], cwd=str(make_ce_script.parent))

    if not args.skip_vectors:
        run_command(
            common
            + [
                "--n_clusters",
                str(args.n_clusters),
                "--run_type",
                "vectors",
            ],
            cwd=str(make_ce_script.parent),
        )

    result = {
        "backend": "make_ce",
        "lsa_ready_csv": str(args.lsa_ready),
        "lsa_ready_text_col": args.text_col,
        "lsa_ready_display_text_col": args.display_text_col,
        "vector_input_source": "lsa_tokens_str",
        "lsa_ready_chunk_id_col": args.chunk_id_col,
        "lsa_ready_file_col": args.file_col,
        "lsa_ready_page_col": args.page_col,
        "base_dir": str(base_dir),
        "craw_name": args.craw_name,
        "model_dir": str(model_dir),
        "df0_csv": str(df0_path),
        "df0_rows": int(len(df0)),
        "step1_dir": str(model_dir / "step1"),
        "step2_dir": str(model_dir / "step2"),
        "make_ce_script": str(make_ce_script),
        "n_clusters": int(args.n_clusters),
        "make_ce_clean_effective": False,
        "make_ce_chunk_effective": False,
        "one_input_row_one_vector": True,
        "preprocessing_profile": str(model_dir / "preprocessing_profile.json") if (model_dir / "preprocessing_profile.json").exists() else None,
        "conversion_report": str(df0_path.with_suffix(".convert_report.json")),
    }

    out_dir = Path(args.output_dir) if args.output_dir else model_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "make_ce_backend_report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lsa-ready", required=True)
    p.add_argument("--base-dir", default=r"C:\project\document_viewer\_data_assets")
    p.add_argument("--craw-name", required=True)
    p.add_argument("--make-ce-script", default="make_ce_v1.py")
    p.add_argument("--python-exe")
    p.add_argument("--output-dir")

    p.add_argument("--chunk-id-col", default="chunk_id_out")
    p.add_argument("--file-col", default="file_name_out")
    p.add_argument("--page-col", default="page_out")
    p.add_argument("--text-col", default="lsa_tokens_str")
    p.add_argument("--display-text-col", default="chunk_text")
    p.add_argument("--docid-col", default="docid")
    p.add_argument("--preprocessing-profile")

    p.add_argument("--n-clusters", type=int, default=0)
    p.add_argument("--skip-space", action="store_true")
    p.add_argument("--skip-vectors", action="store_true")

    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    print(json.dumps(build_make_ce_vectors(parse_args(argv)), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

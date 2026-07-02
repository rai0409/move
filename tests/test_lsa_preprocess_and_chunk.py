from __future__ import annotations

import pandas as pd

from lsa_preprocess_and_chunk import build_output, normalize_text, parse_args


def make_args(tmp_path, input_path, profile="none"):
    return parse_args(
        [
            "--input",
            str(input_path),
            "--output",
            str(tmp_path / "out.csv"),
            "--text-col",
            "text",
            "--file-col",
            "file_name",
            "--page-col",
            "page",
            "--tokenizer",
            "regex",
            "--chunk-profile",
            profile,
            "--overwrite",
            "--report-dir",
            str(tmp_path),
        ]
    )


def test_nbsp_and_zero_width_normalization():
    text, flags = normalize_text("勤務\u00a0時間\u200b[NL]確認")
    assert "勤務時間" in text
    assert "\u200b" not in text
    assert "\n" in text
    assert "nfkc" in flags


def test_japanese_mystery_spaces_are_fixed():
    text, _ = normalize_text("勤 務 時 間")
    assert text == "勤務時間"


def test_japanese_english_boundary_uses_spaces_not_newlines():
    text, _ = normalize_text("業務要件QA\nLINEWORKSクラウド")
    assert "業務要件 QA" in text
    assert "LINEWORKS クラウド" in text
    assert "要件\nQA" not in text


def test_no_chunk_profile_creates_one_output_per_input_row(tmp_path):
    input_path = tmp_path / "in.csv"
    pd.DataFrame(
        {"file_name": ["a.pdf", "b.pdf"], "page": [1, 2], "text": ["申請できない場合は確認する。", "必要な手順を説明する。"]}
    ).to_csv(input_path, index=False)
    out_df, _ = build_output(make_args(tmp_path, input_path, "none"))
    assert len(out_df) == 2
    assert set(out_df["cut_reason"]) == {"no_chunk"}


def test_medium_profile_creates_multiple_chunks_for_long_text(tmp_path):
    input_path = tmp_path / "in.csv"
    long_text = "これは勤務時間の申請手順を説明する文章です。" * 80
    pd.DataFrame({"file_name": ["a.pdf"], "page": [1], "text": [long_text]}).to_csv(input_path, index=False)
    out_df, _ = build_output(make_args(tmp_path, input_path, "medium"))
    assert len(out_df) > 1


def test_output_required_columns_exist(tmp_path):
    input_path = tmp_path / "in.csv"
    pd.DataFrame({"file_name": ["a.pdf"], "page": [1], "text": ["申請できない場合は管理者に確認する。"]}).to_csv(input_path, index=False)
    out_df, _ = build_output(make_args(tmp_path, input_path, "none"))
    required = {
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
    }
    assert required.issubset(out_df.columns)


def test_lsa_tokens_str_non_empty_for_normal_japanese_text(tmp_path):
    input_path = tmp_path / "in.csv"
    pd.DataFrame({"file_name": ["a.pdf"], "page": [1], "text": ["勤務時間について申請方法を説明する。"]}).to_csv(input_path, index=False)
    out_df, _ = build_output(make_args(tmp_path, input_path, "none"))
    assert out_df.loc[0, "lsa_tokens_str"]
    assert out_df.loc[0, "token_count"] > 0


def test_risky_words_are_not_removed_by_stopwords(tmp_path):
    input_path = tmp_path / "in.csv"
    pd.DataFrame({"file_name": ["a.pdf"], "page": [1], "text": ["申請できない場合は必要な確認をする。"]}).to_csv(input_path, index=False)
    out_df, _ = build_output(make_args(tmp_path, input_path, "none"))
    tokens = out_df.loc[0, "lsa_tokens_str"].split()
    assert "ない" in tokens or any("ない" in token for token in tokens)
    assert "必要" in tokens or any("必要" in token for token in tokens)

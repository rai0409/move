from __future__ import annotations

import pandas as pd

from audit_lsa_csv_columns import audit_dataframe


def valid_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "chunk_id_out": ["a_0", "a_1"],
            "file_name_out": ["a.pdf", "a.pdf"],
            "page_out": [1, 2],
            "chunk_text": ["勤務時間の申請方法を説明する。", "必要な確認手順を説明する。"],
            "lsa_tokens_str": ["勤務時間 申請 方法 説明 する 必要", "必要 確認 手順 説明 する 場合"],
            "token_count": [6, 6],
            "char_len": [16, 14],
            "cut_reason": ["no_chunk", "no_chunk"],
            "forced_slice": [False, False],
            "tokenizer_name": ["regex", "regex"],
            "chunk_profile": ["none", "none"],
            "model_target": ["TFIDF_TRUNCATED_SVD_LSA", "TFIDF_TRUNCATED_SVD_LSA"],
        }
    )


def test_audit_pass_for_valid_generated_csv():
    report = audit_dataframe(valid_df())
    assert report["verdict"] == "PASS"
    assert report["score"] == 100


def test_audit_fail_when_lsa_tokens_str_missing():
    report = audit_dataframe(valid_df().drop(columns=["lsa_tokens_str"]))
    assert report["verdict"] == "FAIL"
    assert any("lsa_tokens_str" in item for item in report["failures"])

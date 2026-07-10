from __future__ import annotations

from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd
import pytest

from build_lsa_vector_space import build_vector_space, parse_args as parse_build_args
from mine_lsa_improvement_candidates import mine_candidates, parse_args as parse_mine_args
from run_lsa_auto_improve_cycles import parse_args as parse_cycle_args, run_cycles


def lsa_csv(path: Path) -> None:
    pd.DataFrame(
        {
            "chunk_id_out": ["c1", "c2", "c3"],
            "file_name_out": ["a.pdf", "a.pdf", "b.pdf"],
            "page_out": [1, 2, 3],
            "chunk_text": ["勤怠 勤務時間 申請", "勤怠 勤務時間 確認", "休暇 申請"],
            "lsa_tokens_str": ["勤怠 勤務時間 申請", "勤怠 勤務時間 確認", "休暇 申請"],
            "token_count": [3, 3, 2],
            "char_len": [12, 12, 7],
            "cut_reason": ["no_chunk", "no_chunk", "no_chunk"],
            "forced_slice": [False, False, False],
            "tokenizer_name": ["regex", "regex", "regex"],
            "chunk_profile": ["none", "none", "none"],
            "model_target": ["TFIDF_TRUNCATED_SVD_LSA"] * 3,
        }
    ).to_csv(path, index=False, encoding="utf-8-sig")


def build_term_model(tmp_path: Path) -> Path:
    csv_path = tmp_path / "lsa.csv"
    model_dir = tmp_path / "model"
    lsa_csv(csv_path)
    build_vector_space(
        parse_build_args(
            [
                "--input",
                str(csv_path),
                "--output-dir",
                str(model_dir),
                "--min-df",
                "1",
                "--max-df",
                "1.0",
                "--svd-dim",
                "2",
                "--export-term-vectors",
            ]
        )
    )
    return model_dir


def cycle_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    input_csv = tmp_path / "input.csv"
    queries = tmp_path / "queries.csv"
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    pd.DataFrame(
        {
            "file_name": ["manual.pdf", "ops.pdf"],
            "page": [1, 2],
            "text": ["勤務時間の申請には必要な確認がある。", "対象システムのAPI連携手順を説明する。"],
        }
    ).to_csv(input_csv, index=False)
    pd.DataFrame(
        {
            "query": ["LONGAPI連携 対象", "LONGAPI連携 手順"],
            "target_doc": ["ops.pdf", "ops.pdf"],
            "page": [2, 2],
        }
    ).to_csv(queries, index=False)
    return input_csv, queries, config_dir


def run_cycle_fixture(tmp_path: Path, cycles: int = 3) -> dict[str, object]:
    if importlib.util.find_spec("make_space_v1_r2") is None:
        pytest.skip("make_ce dependencies are not shared on this PC")
    input_csv, queries, config_dir = cycle_fixture(tmp_path)
    return run_cycles(
        parse_cycle_args(
            [
                "--input",
                str(input_csv),
                "--queries",
                str(queries),
                "--text-col",
                "text",
                "--file-col",
                "file_name",
                "--page-col",
                "page",
                "--config-dir",
                str(config_dir),
                "--output-dir",
                str(tmp_path / "cycles"),
                "--cycles",
                str(cycles),
                "--max-candidates-per-cycle",
                "20",
                "--max-approved-per-cycle",
                "5",
                "--min-score-delta",
                "0.01",
                "--rollback-if-score-drops",
                "--dry-run",
                "--tokenizer",
                "regex",
            ]
        )
    )


def test_term_vector_export_creates_outputs(tmp_path):
    model_dir = build_term_model(tmp_path)
    vectors = np.load(model_dir / "term_lsa_vectors.npy")
    metadata = pd.read_csv(model_dir / "term_metadata.csv")
    assert vectors.shape[0] == len(metadata)
    assert {"term", "term_index", "doc_frequency", "mean_tfidf"}.issubset(metadata.columns)


def test_synonym_mining_from_lsa_neighbor_signals(tmp_path):
    model_dir = build_term_model(tmp_path)
    query_results = tmp_path / "query_results.csv"
    config_dir = tmp_path / "config"
    pd.DataFrame(
        {
            "query": ["勤怠"],
            "missing_terms": ["勤怠"],
            "hit_text": ["勤務時間 申請"],
            "target_doc": ["a.pdf"],
            "page": [1],
        }
    ).to_csv(query_results, index=False)
    actions, _ = mine_candidates(
        parse_mine_args(
            [
                "--query-results",
                str(query_results),
                "--term-stats",
                str(model_dir / "term_stats.csv"),
                "--config-dir",
                str(config_dir),
                "--output-dir",
                str(tmp_path / "candidates"),
                "--term-vectors",
                str(model_dir / "term_lsa_vectors.npy"),
                "--term-metadata",
                str(model_dir / "term_metadata.csv"),
                "--min-term-similarity",
                "0.0",
            ]
        )
    )
    assert any((actions["action"] == "synonym") & (actions["reason"] == "lsa_neighbor_missing_term"))


def test_multi_cycle_dry_run_completes(tmp_path):
    result = run_cycle_fixture(tmp_path)
    assert Path(result["output_dir"], "final", "final_metrics.json").exists()


def test_dry_run_does_not_modify_original_config_files(tmp_path):
    if importlib.util.find_spec("make_space_v1_r2") is None:
        pytest.skip("make_ce dependencies are not shared on this PC")
    input_csv, queries, config_dir = cycle_fixture(tmp_path)
    before = sorted(p.name for p in config_dir.iterdir())
    run_cycles(
        parse_cycle_args(
            [
                "--input",
                str(input_csv),
                "--queries",
                str(queries),
                "--text-col",
                "text",
                "--file-col",
                "file_name",
                "--page-col",
                "page",
                "--config-dir",
                str(config_dir),
                "--output-dir",
                str(tmp_path / "cycles"),
                "--cycles",
                "1",
                    "--dry-run",
                    "--tokenizer",
                    "regex",
            ]
        )
    )
    after = sorted(p.name for p in config_dir.iterdir())
    assert before == after == []


def test_rollback_occurs_when_score_does_not_improve(tmp_path):
    result = run_cycle_fixture(tmp_path)
    assert result["rolled_back"] >= 1


def test_cycle_summary_is_written(tmp_path):
    result = run_cycle_fixture(tmp_path)
    assert Path(result["output_dir"], "cycle_summary.csv").exists()


def test_approved_history_is_written_when_approvals_exist(tmp_path):
    result = run_cycle_fixture(tmp_path)
    history = pd.read_csv(Path(result["output_dir"], "approved_history.csv"))
    assert len(history) >= 1


def test_final_metrics_is_written(tmp_path):
    result = run_cycle_fixture(tmp_path)
    assert Path(result["output_dir"], "final", "final_metrics.json").exists()


def test_max_cycles_is_respected(tmp_path):
    result = run_cycle_fixture(tmp_path, cycles=1)
    assert result["cycles_run"] <= 1


def test_no_transformer_related_outputs_are_created(tmp_path):
    result = run_cycle_fixture(tmp_path)
    names = [p.name.lower() for p in Path(result["output_dir"]).rglob("*")]
    assert not any("transformer" in name or "embedding" in name or "bert" in name for name in names)

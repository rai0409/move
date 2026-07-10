from __future__ import annotations

from pathlib import Path
import importlib.util
import json

import pandas as pd
import pytest

from apply_lsa_list_updates import apply_updates, parse_args as parse_apply_args
from auto_improve_lsa_lists import parse_args as parse_auto_args, run_auto_improve
from build_lsa_vector_space import build_vector_space, parse_args as parse_build_args
from convert_lsa_ready_to_make_ce_df0 import convert_lsa_ready_to_make_ce_df0, parse_args as parse_convert_args
from evaluate_lsa_retrieval import evaluation_normalize, evaluate, match_hit, parse_args as parse_eval_args
from mine_lsa_improvement_candidates import ensure_config_xlsx, mine_candidates, parse_args as parse_mine_args
from validate_lsa_candidate_actions import measured_decision, query_rank_deltas


def tiny_lsa_csv(path: Path) -> None:
    pd.DataFrame(
        {
            "chunk_id_out": ["c1", "c2", "c3"],
            "file_name_out": ["manual.pdf", "manual.pdf", "ops.pdf"],
            "page_out": [1, 2, 3],
            "chunk_text": ["LINE WORKS クラウド の利用方法", "勤務時間の申請には必要な確認がある", "対象 システム API連携"],
            "lsa_tokens_str": ["LINE WORKS クラウド 利用 方法", "勤務時間 申請 必要 確認 ある", "対象 システム API連携"],
            "token_count": [5, 5, 3],
            "char_len": [20, 22, 14],
            "cut_reason": ["no_chunk", "no_chunk", "no_chunk"],
            "forced_slice": [False, False, False],
            "tokenizer_name": ["regex", "regex", "regex"],
            "chunk_profile": ["none", "none", "none"],
            "model_target": ["TFIDF_TRUNCATED_SVD_LSA"] * 3,
        }
    ).to_csv(path, index=False, encoding="utf-8-sig")
    (path.parent / "preprocessing_profile.json").write_text(
        '{"unicode_normalization":"NFKC","lowercase_policy":"ASCII lower",'
        '"pos_policy":"content_lemma","lemma_policy":"test", "stopwords":[],"keep_words":[],'
        '"protected_patterns":[],"token_join_policy":"single ASCII space","use_noun_compounds":false,'
        '"synonyms":{},"replacements":[],"candidate":{"tokenizer_name":"regex"}}', encoding="utf-8"
    )


def build_model(tmp_path: Path) -> Path:
    csv_path = tmp_path / "lsa.csv"
    model_dir = tmp_path / "model"
    tiny_lsa_csv(csv_path)
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
            ]
        )
    )
    return model_dir


def test_build_vector_space_on_tiny_lsa_csv(tmp_path):
    model_dir = build_model(tmp_path)
    assert (model_dir / "tfidf_vectorizer.joblib").exists()
    assert (model_dir / "truncated_svd.joblib").exists()
    assert (model_dir / "lsa_vectors.npy").exists()
    assert (model_dir / "metadata.csv").exists()
    assert (model_dir / "term_stats.csv").exists()


def test_evaluate_query_with_target_doc_page(tmp_path):
    model_dir = build_model(tmp_path)
    queries = tmp_path / "queries.csv"
    pd.DataFrame({"query": ["勤務時間 申請 必要"], "target_doc": ["manual.pdf"], "page": [2]}).to_csv(queries, index=False)
    metrics = evaluate(parse_eval_args(["--model-dir", str(model_dir), "--queries", str(queries), "--output-dir", str(tmp_path / "eval"), "--top-k", "3"]))
    assert metrics["recall_at_3"] == 1.0


def test_query_results_includes_missing_terms_and_jaccard(tmp_path):
    model_dir = build_model(tmp_path)
    queries = tmp_path / "queries.csv"
    pd.DataFrame({"query": ["LINEWORKS クラウド"], "target_doc": ["manual.pdf"], "page": [1]}).to_csv(queries, index=False)
    evaluate(parse_eval_args(["--model-dir", str(model_dir), "--queries", str(queries), "--output-dir", str(tmp_path / "eval")]))
    results = pd.read_csv(tmp_path / "eval" / "ranked_results.csv")
    assert "missing_terms" in results.columns
    assert "jaccard" in results.columns


def test_mine_candidates_detects_spacing_replace_candidate(tmp_path):
    config_dir = tmp_path / "config"
    ensure_config_xlsx(config_dir)
    query_results = tmp_path / "query_results.csv"
    term_stats = tmp_path / "term_stats.csv"
    pd.DataFrame(
        {
            "query": ["LINEWORKS クラウド"],
            "missing_terms": ["LINEWORKS"],
            "hit_text": ["LINE WORKS クラウド"],
            "target_doc": ["manual.pdf"],
            "page": [1],
            "rank": [1],
        }
    ).to_csv(query_results, index=False)
    pd.DataFrame({"term": ["LINEWORKS", "LINEWORKS", "LINEWORKS"], "frequency": [1, 1, 1], "doc_frequency": [1, 1, 1], "doc_freq_ratio": [0.3, 0.3, 0.3]}).to_csv(term_stats, index=False)
    actions, _ = mine_candidates(parse_mine_args(["--query-results", str(query_results), "--term-stats", str(term_stats), "--config-dir", str(config_dir), "--output-dir", str(tmp_path / "candidates")]))
    assert any(actions["action"] == "replace")


def test_mine_candidates_detects_keep_candidate_for_technical_term(tmp_path):
    config_dir = tmp_path / "config"
    query_results = tmp_path / "query_results.csv"
    term_stats = tmp_path / "term_stats.csv"
    pd.DataFrame({"query": ["API連携"], "missing_terms": ["API連携"], "hit_text": ["システム 連携"], "target_doc": ["ops.pdf"], "page": [3]}).to_csv(query_results, index=False)
    pd.DataFrame({"term": ["システム"], "frequency": [2], "doc_frequency": [1], "doc_freq_ratio": [0.5]}).to_csv(term_stats, index=False)
    actions, _ = mine_candidates(parse_mine_args(["--query-results", str(query_results), "--term-stats", str(term_stats), "--config-dir", str(config_dir), "--output-dir", str(tmp_path / "candidates")]))
    assert any((actions["action"] == "keep") & (actions["term"] == "API連携"))


def test_stopword_candidate_excludes_risky_words(tmp_path):
    config_dir = tmp_path / "config"
    query_results = tmp_path / "query_results.csv"
    term_stats = tmp_path / "term_stats.csv"
    pd.DataFrame({"query": ["dummy"], "missing_terms": [""], "hit_text": ["dummy"]}).to_csv(query_results, index=False)
    pd.DataFrame(
        {"term": ["ない", "必要", "対象", "dummy"], "frequency": [10, 10, 10, 10], "doc_frequency": [10, 10, 10, 10], "doc_freq_ratio": [0.9, 0.9, 0.9, 0.9]}
    ).to_csv(term_stats, index=False)
    actions, _ = mine_candidates(parse_mine_args(["--query-results", str(query_results), "--term-stats", str(term_stats), "--config-dir", str(config_dir), "--output-dir", str(tmp_path / "candidates")]))
    stop_terms = set(actions.loc[actions["action"] == "stopword", "term"])
    assert not {"ない", "必要", "対象"} & stop_terms
    assert "dummy" in stop_terms


def validation_and_actions(tmp_path: Path) -> tuple[Path, Path, Path]:
    config_dir = tmp_path / "config"
    ensure_config_xlsx(config_dir)
    actions = tmp_path / "candidate_actions.csv"
    validation = tmp_path / "candidate_validation.csv"
    pd.DataFrame(
        {
            "candidate_id": ["C0001"],
            "action": ["keep"],
            "target_list": ["list_keep.xlsx"],
            "term": ["API連携"],
            "pattern": [""],
            "replacement": [""],
            "canonical": [""],
            "variant": [""],
            "direction": [""],
            "reason": ["technical_missing_term"],
            "priority_score": [4],
            "evidence_count": [2],
            "status": ["candidate"],
            "note": [""],
        }
    ).to_csv(actions, index=False)
    pd.DataFrame(
        {
            "candidate_id": ["C0001"],
            "action": ["keep"],
            "target_list": ["list_keep.xlsx"],
            "term": ["API連携"],
            "score_delta": [0],
            "mrr_delta": [0],
            "recall5_delta": [0],
            "recall1_delta": [0],
            "improved_count": [0],
            "degraded_count": [0],
            "decision": ["approve"],
            "reason": ["test"],
        }
    ).to_csv(validation, index=False)
    return config_dir, actions, validation


def test_apply_lsa_list_updates_dry_run_does_not_modify_config(tmp_path):
    config_dir, actions, validation = validation_and_actions(tmp_path)
    before = pd.read_excel(config_dir / "list_keep.xlsx")
    result = apply_updates(parse_apply_args(["--config-dir", str(config_dir), "--candidate-validation", str(validation), "--candidate-actions", str(actions), "--backup-dir", str(tmp_path / "backup"), "--dry-run"]))
    after = pd.read_excel(config_dir / "list_keep.xlsx")
    assert result["dry_run"] is True
    assert len(before) == len(after)


def test_apply_lsa_list_updates_apply_updates_xlsx_files(tmp_path):
    config_dir, actions, validation = validation_and_actions(tmp_path)
    result = apply_updates(parse_apply_args(["--config-dir", str(config_dir), "--candidate-validation", str(validation), "--candidate-actions", str(actions), "--backup-dir", str(tmp_path / "backup"), "--apply"]))
    keep = pd.read_excel(config_dir / "list_keep.xlsx")
    assert result["updated"] == 1
    assert "API連携" in set(keep["term"])


def test_auto_improve_lsa_lists_dry_run_completes_on_tiny_dataset(tmp_path):
    if importlib.util.find_spec("make_space_v1_r2") is None:
        pytest.skip("make_ce dependencies are not shared on this PC")
    input_csv = tmp_path / "input.csv"
    queries = tmp_path / "queries.csv"
    config_dir = tmp_path / "config"
    pd.DataFrame(
        {
            "file_name": ["manual.pdf", "ops.pdf"],
            "page": [1, 2],
            "text": ["勤務時間の申請には必要な確認がある。", "LINE WORKS クラウドとAPI連携を説明する。"],
        }
    ).to_csv(input_csv, index=False)
    pd.DataFrame({"query": ["勤務時間 申請 必要"], "target_doc": ["manual.pdf"], "page": [1]}).to_csv(queries, index=False)
    result = run_auto_improve(
        parse_auto_args(
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
                str(tmp_path / "run"),
                "--dry-run",
                "--tokenizer",
                "regex",
            ]
        )
    )
    assert Path(result["output_dir"], "leaderboard.csv").exists()


def test_public_query_without_targets_creates_diagnostics_without_recall(tmp_path):
    model_dir = build_model(tmp_path)
    queries = tmp_path / "public_queries.csv"
    pd.DataFrame({"query": ["クラウド 利用"], "target_doc": [""], "page": [""]}).to_csv(queries, index=False)
    metrics = evaluate(parse_eval_args(["--model-dir", str(model_dir), "--queries", str(queries), "--output-dir", str(tmp_path / "public_eval")]))
    assert (tmp_path / "public_eval" / "public_query_diagnostics.csv").exists()
    assert metrics["recall_at_1"] is None


def run_results_fixture(tmp_path: Path, queries: pd.DataFrame, results: pd.DataFrame):
    query_path, result_path, output_dir = tmp_path / "gt.csv", tmp_path / "ranked.csv", tmp_path / "evaluated"
    queries.to_csv(query_path, index=False, encoding="utf-8-sig")
    results.to_csv(result_path, index=False, encoding="utf-8-sig")
    metrics = evaluate(parse_eval_args([
        "--backend", "results_csv", "--queries", str(query_path), "--results-csv", str(result_path),
        "--output-dir", str(output_dir), "--top-k", "20", "--file-col", "hit_doc",
        "--page-col", "hit_page", "--display-col", "hit_text", "--id-col", "hit_id",
        "--docid-col", "hit_docid",
    ]))
    return metrics, pd.read_csv(output_dir / "query_results.csv"), output_dir


def test_ground_truth_doc_page_text_and_evaluation_normalization(tmp_path):
    queries = pd.DataFrame({
        "query": ["q"], "target_doc": ["ＭＡＮＵＡＬ.pdf"], "page": [2],
        "docid": ["old-id"], "expected_text_contains": ["申請 条件。"], "コメント": ["確認"],
    })
    results = pd.DataFrame({
        "query": ["q"], "rank": [1], "score": [0.9], "hit_doc": ["manual.pdf"], "hit_page": [2],
        "hit_text": ["申請\n条件"], "hit_docid": ["changed-id"], "hit_id": ["new-chunk"],
    })
    metrics, summary, output_dir = run_results_fixture(tmp_path, queries, results)
    assert evaluation_normalize("Ａ B。\nC") == "abc"
    assert metrics["recall_at_1"] == 1.0
    assert summary.loc[0, "matched_by"] == "target_doc_page_text"
    assert summary.loc[0, "matched_result_chunk_id"] == "new-chunk"
    assert (output_dir / "error_analysis.csv").exists()


def test_target_doc_page_and_text_mismatches_are_not_hits():
    q = pd.Series({"target_doc": "a.pdf", "page": 2, "expected_text_contains": "正解"})
    wrong_doc = pd.Series({"hit_doc": "b.pdf", "hit_page": 2, "hit_text": "正解", "hit_docid": "", "hit_id": "c"})
    wrong_page = pd.Series({"hit_doc": "a.pdf", "hit_page": 3, "hit_text": "正解", "hit_docid": "", "hit_id": "c"})
    wrong_text = pd.Series({"hit_doc": "a.pdf", "hit_page": 2, "hit_text": "不一致", "hit_docid": "", "hit_id": "c"})
    kwargs = dict(file_col="hit_doc", page_col="hit_page", display_col="hit_text", docid_col="hit_docid", id_col="hit_id")
    assert not match_hit(q, wrong_doc, **kwargs)["is_hit"]
    assert not match_hit(q, wrong_page, **kwargs)["is_hit"]
    assert not match_hit(q, wrong_text, **kwargs)["is_hit"]


def test_docid_is_auxiliary_and_changed_docid_can_hit_by_page_text():
    kwargs = dict(file_col="hit_doc", page_col="hit_page", display_col="hit_text", docid_col="hit_docid", id_col="hit_id")
    docid_only = pd.Series({"target_doc": "a.pdf", "docid": "D-1"})
    hit = pd.Series({"hit_doc": "a.pdf", "hit_page": 9, "hit_text": "x", "hit_docid": "D-1", "hit_id": "new"})
    assert match_hit(docid_only, hit, **kwargs)["matched_by"] == "target_doc_docid"
    changed = pd.Series({"target_doc": "a.pdf", "page": 9, "docid": "OLD", "expected_text_contains": "回答"})
    changed_hit = pd.Series({"hit_doc": "a.pdf", "hit_page": 9, "hit_text": "回答です", "hit_docid": "NEW", "hit_id": "new"})
    assert match_hit(changed, changed_hit, **kwargs)["matched_by"] == "target_doc_page_text"


def test_recall_1_5_20_mrr_and_diagnostic_denominator(tmp_path):
    queries = pd.DataFrame({
        "query": ["q1", "q5", "q20", "qout", "diag"],
        "target_doc": ["d1", "d5", "d20", "missing", ""], "page": [1, 1, 1, 1, ""],
    })
    rows = []
    expected = {"q1": (1, "d1"), "q5": (5, "d5"), "q20": (20, "d20")}
    for query in queries["query"]:
        for rank in range(1, 21):
            doc = expected[query][1] if query in expected and rank == expected[query][0] else f"noise-{query}-{rank}"
            rows.append({"query": query, "rank": rank, "score": 1 / rank, "hit_doc": doc, "hit_page": 1,
                         "hit_text": "x", "hit_docid": "", "hit_id": f"{query}-{rank}"})
    metrics, summary, _ = run_results_fixture(tmp_path, queries, pd.DataFrame(rows))
    assert metrics["evaluated_query_count"] == 4
    assert metrics["diagnostic_only_query_count"] == 1
    assert metrics["recall_at_1"] == 0.25
    assert metrics["recall_at_5"] == 0.5
    assert metrics["recall_at_20"] == 0.75
    assert metrics["mrr"] == pytest.approx((1 + 1 / 5 + 1 / 20) / 4)
    assert dict(zip(summary["query"], summary["best_correct_rank"].fillna(0).astype(int)))["q20"] == 20


def test_converter_uses_lsa_tokens_for_text_and_keeps_chunk_text(tmp_path):
    source = tmp_path / "lsa_ready.csv"
    output = tmp_path / "df0.csv"
    pd.DataFrame({
        "chunk_id_out": ["c1"], "file_name_out": ["a.pdf"], "page_out": [1], "docid": ["doc-1"],
        "lsa_tokens_str": ["申請 条件 abc-123"], "chunk_text": ["申請条件 ABC-123"],
    }).to_csv(source, index=False, encoding="utf-8-sig")
    converted = convert_lsa_ready_to_make_ce_df0(parse_convert_args([
        "--input", str(source), "--output", str(output), "--craw-name", "test",
    ]))
    assert converted.loc[0, "text"] == "申請 条件 abc-123"
    assert converted.loc[0, "chunk_text"] == "申請条件 ABC-123"
    assert converted.loc[0, "text"] != converted.loc[0, "chunk_text"]
    report = json.loads(output.with_suffix(".convert_report.json").read_text(encoding="utf-8"))
    assert report["source_column"] == "lsa_tokens_str"
    assert report["source_destination_hash_match"] is True


def test_preprocessing_profile_mismatch_blocks_measurement(tmp_path):
    model_dir = build_model(tmp_path)
    mismatch = tmp_path / "mismatch.json"
    mismatch.write_text('{"unicode_normalization":"NFC","candidate":{"tokenizer_name":"regex"}}', encoding="utf-8")
    queries = tmp_path / "queries_mismatch.csv"
    pd.DataFrame({"query": ["勤務時間"], "target_doc": ["manual.pdf"], "page": [2]}).to_csv(queries, index=False)
    with pytest.raises(RuntimeError, match="preprocessing_profile_mismatch"):
        evaluate(parse_eval_args(["--model-dir", str(model_dir), "--queries", str(queries),
                                  "--output-dir", str(tmp_path / "blocked"),
                                  "--query-preprocessing-profile", str(mismatch)]))
    lineage = json.loads((tmp_path / "blocked" / "evaluation_lineage.json").read_text(encoding="utf-8"))
    assert lineage["profiles_match"] is False


def test_candidate_gate_requires_gain_and_no_metric_drop():
    baseline = {"evaluated_query_count": 100, "recall_at_1": 0.50, "recall_at_5": 0.70,
                "recall_at_20": 0.90, "mrr": 0.60}
    improved = {**baseline, "recall_at_1": 0.51, "mrr": 0.61}
    decision, _, _ = measured_decision(baseline, improved, min_score_delta=0.01,
                                       rollback_if_score_drops=True, critical_regressions=[], min_evaluated_queries=20)
    assert decision == "approve"
    flat, _, _ = measured_decision(baseline, baseline, min_score_delta=0.01,
                                   rollback_if_score_drops=True, critical_regressions=[], min_evaluated_queries=20)
    assert flat == "needs_more_evidence"
    worse = {**improved, "recall_at_20": 0.89}
    rejected, _, _ = measured_decision(baseline, worse, min_score_delta=0.01,
                                       rollback_if_score_drops=True, critical_regressions=[], min_evaluated_queries=20)
    assert rejected == "reject"


def test_query_rank_delta_counts_and_worst_drop(tmp_path):
    before, after = tmp_path / "before.csv", tmp_path / "after.csv"
    pd.DataFrame({"query_index": [0, 1, 2], "query": ["a", "b", "c"], "best_correct_rank": [5, 1, None]}).to_csv(before, index=False)
    pd.DataFrame({"query_index": [0, 1, 2], "query": ["a", "b", "c"], "best_correct_rank": [1, 10, 20]}).to_csv(after, index=False)
    delta = query_rank_deltas(str(before), str(after))
    assert delta["improved_query_count"] == 2
    assert delta["regressed_query_count"] == 1
    assert delta["worst_rank_drop"] == 9
    assert delta["best_rank_gain"] == 4

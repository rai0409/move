# LSA Auto-Improve Loop

This workflow is only for TF-IDF + TruncatedSVD(LSA). It improves preprocessing, chunking, vectorizer settings, and the Excel config lists used around the LSA pipeline. It does not train neural weights and does not use BERT, sentence-transformers, embedding APIs, or `transformer_text`.

## What It Optimizes

- `config/df_replace.xlsx`
- `config/list_keep.xlsx`
- `config/list_stopword.xlsx`
- `config/list_synonym.xlsx`
- chunk/preprocess/vectorizer settings through measured run reports

Public queries without targets are useful for diagnostics and candidate mining. Targeted eval queries with `target_doc`, `page`, or `expected_text_contains` are required for recall/MRR scoring.

## Run

```bash
python3 auto_improve_lsa_lists.py \
  --input data/input.csv \
  --queries data/eval_queries.csv \
  --text-col text \
  --file-col file_name \
  --page-col page \
  --config-dir config \
  --output-dir runs/auto_improve_001 \
  --max-candidates 20 \
  --dry-run
```

Use `--apply` only after reviewing `candidate_validation.csv` and `recommendations.md`.

## Schemas

`df_replace.xlsx`: `pattern,replacement,match_type,stage,enabled,priority,source,evidence_count,improved_count,degraded_count,last_score_delta,status,note`

`list_keep.xlsx`: `term,term_type,enabled,priority,source,evidence_count,status,note`

`list_stopword.xlsx`: `term,enabled,priority,source,frequency,doc_freq_ratio,total_miss_count,mean_rank_when_degraded,mean_rank_baseline,rank_delta,status,note`

`list_synonym.xlsx`: `canonical,variant,direction,enabled,priority,source,evidence_count,improved_count,degraded_count,last_score_delta,status,note`

`query_results.csv`: `query,hit_text,rank,score,n_query_tokens,n_hit_tokens,jaccard,missing_terms,extra_terms,target_doc,page,hit_doc,hit_page,is_hit,is_target_doc_hit,is_page_hit`

`term_degradation_report.csv`: `term,term_type,frequency,doc_frequency,query_frequency,total_miss_count,affected_query_count,mean_rank_when_missing,mean_rank_baseline,mean_rank_candidate,rank_delta,recall5_delta,mrr_delta,improved_count,degraded_count,priority_score,recommended_action,recommended_list,status`

## Dry Run And Apply

Dry-run is the default operational posture. It builds a baseline model, evaluates queries, mines candidates, validates decisions, and writes a leaderboard without changing config Excel files.

When `--apply` is passed, existing Excel files are backed up first, then approved candidates are appended or updated with `status=approved`.

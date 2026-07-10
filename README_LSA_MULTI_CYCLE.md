# LSA Multi-Cycle Auto-Improve

This runner repeats the LSA auto-improvement loop across multiple cycles. It remains strictly TF-IDF + TruncatedSVD(LSA): no BERT, sentence-transformers, embedding APIs, neural fine-tuning, or `transformer_text`.

## One-Shot Vs Multi-Cycle

`auto_improve_lsa_lists.py` runs one baseline build, mines candidates, validates them, and writes recommendations.

`run_lsa_auto_improve_cycles.py` repeats that process. For each cycle it snapshots config, selects approved candidates, applies them to a temporary config, rebuilds/evaluates, and accepts the temporary config only if the score improves enough.

## Rollback

Before each cycle, the current working config is copied to `cycle_XXX/config_snapshot/`.

If the temporary config does not improve by `--min-score-delta`, the runner restores the snapshot. If `--rollback-if-score-drops` is enabled, a lower `recall_at_5` also forces rollback.

Dry-run never writes accepted changes back to the original `--config-dir`. With `--apply`, the final accepted config is copied back only after backing up the original config.

## Score

The composite score is:

```text
recall_at_1 * 40 + recall_at_3 * 25 + recall_at_5 * 15 + mrr * 20
```

The cycle decision also considers rollback conditions such as recall drop. Preprocess reports expose cut rate and empty-token rows so future policies can add penalties without changing the model type.

## LSA Term-Neighbor Synonyms

When `build_lsa_vector_space.py --export-term-vectors` is enabled, it writes:

- `term_lsa_vectors.npy`
- `term_metadata.csv`

The vectors are `truncated_svd.components_.T`, L2-normalized in vectorizer vocabulary order. The miner can use nearest LSA term neighbors for missing query terms and propose `list_synonym.xlsx` candidates with `reason=lsa_neighbor_missing_term`.

These are not auto-approved semantic guesses. They remain candidate actions; validation and cycle scoring decide whether they should be accepted.

## Why Eval Queries Matter

Public queries without targets can expose missing terms and diagnostics, but they cannot compute trustworthy recall or MRR. Multi-cycle acceptance requires targeted eval queries with `target_doc`, `page`, or `expected_text_contains`.

## Safe Settings

Start with:

```bash
python3 run_lsa_auto_improve_cycles.py \
  --input data/input.csv \
  --queries data/eval_queries.csv \
  --text-col text \
  --file-col file_name \
  --page-col page \
  --config-dir config \
  --output-dir runs/cycles_001 \
  --cycles 5 \
  --max-candidates-per-cycle 20 \
  --max-approved-per-cycle 5 \
  --min-score-delta 0.01 \
  --stop-if-no-approved \
  --rollback-if-score-drops \
  --dry-run
```

Review `cycle_summary.csv`, `approved_history.csv`, and `final/recommendations.md` before running with `--apply`.

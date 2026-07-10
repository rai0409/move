# LSA Preprocess Pipeline

This pipeline prepares an existing CSV for a Japanese TF-IDF + TruncatedSVD(LSA) retrieval model.

It is not a transformer embedding pipeline. It does not create `transformer_text`, BERT inputs, sentence-transformer vectors, or embedding API payloads.

The model path is:

```text
normalized text -> active chunks -> Japanese/technical tokens -> TF-IDF -> TruncatedSVD(LSA)
```

## Output Columns

Use `lsa_tokens_str` as the TF-IDF model input.

Use `chunk_text` for search result display.

Use `file_name_out` and `page_out` for source attribution.

The output always sets `model_target` to `TFIDF_TRUNCATED_SVD_LSA`.

## Preprocess Example

```bash
python3 lsa_preprocess_and_chunk.py \
  --input /path/to/input.csv \
  --output /path/to/output_lsa.csv \
  --text-col text \
  --file-col file_name \
  --page-col page \
  --tokenizer regex \
  --chunk-profile medium \
  --use-noun-compounds \
  --overwrite
```

The default `regex` tokenizer is dependency-light and works without MeCab, fugashi, or Sudachi. If `--tokenizer fugashi` or `--tokenizer sudachi_*` is selected and the package is not installed, the script fails clearly instead of silently falling back.

## Chunk Profiles

- `none`: one output chunk per input row.
- `small`: target 450 chars, 60 overlap, hard max 700, min 120.
- `medium`: target 750 chars, 100 overlap, hard max 1100, min 180.
- `large`: target 1050 chars, 150 overlap, hard max 1500, min 250.

You can override profile values with `--target-chars`, `--overlap`, `--hard-max`, and `--min-chars`.

## Safe Overwrite

To overwrite the input CSV, pass the same path to `--input` and `--output` with `--overwrite`.

```bash
python3 lsa_preprocess_and_chunk.py \
  --input data/chunks.csv \
  --output data/chunks.csv \
  --text-col text \
  --file-col file_name \
  --page-col page \
  --overwrite
```

When input and output paths are equal, the script writes a temporary file first and replaces the original only after CSV generation succeeds.

## Reports

If `--report-dir` is omitted, reports are written next to the output CSV:

- `<output_stem>.lsa_preprocess_report.json`
- `<output_stem>.top_tokens.csv`
- `<output_stem>.suspicious_rows.csv`

## Audit Example

```bash
python3 audit_lsa_csv_columns.py \
  --input /path/to/output_lsa.csv \
  --text-col lsa_tokens_str \
  --display-col chunk_text \
  --file-col file_name_out \
  --page-col page_out \
  --output-report /path/to/audit.json \
  --output-md /path/to/audit.md
```

The audit reports `PASS`, `WARN`, or `FAIL`, a 0-100 score, required-column presence, metrics, failures, and warnings. A `PASS` or low-warning `WARN` CSV is suitable to wire into the current TF-IDF + TruncatedSVD(LSA) model by reading `lsa_tokens_str`.

# LSA Flow Audit

## 2026-07-10 正解付き評価ループ更新（現行仕様）

- vector経路は `lsa_ready.csv[lsa_tokens_str] -> make_ce step0/df0.csv[text] -> TF-IDF -> TruncatedSVD`。`chunk_text` は `df0.csv[chunk_text]` と `original_text` に保持し、表示・引用・正解判定だけに使う。
- converterはsource/destination column hash一致、row count一致、empty token row数、clean OFF、chunk OFF、1入力行1vectorをJSONへ保存する。
- query CSV列は `query,target_doc,page,docid,expected_text_contains,model3の順位,コメント`。target_docとpage/docid/textのいずれかを持つ行だけをrecall分母にする。
- 正解はtarget_doc必須。指定pageとexpected textはそれぞれ必須条件、docidは補助条件。chunk_id/docidが変わってもtarget_doc+page+textで判定できる。
- 評価専用text包含正規化はNFKC、空白/改行除去、ASCII lower、`、。，．,.`差の吸収だけで、検索前処理とは分離する。
- `baseline/eval` にmetrics、query_results、ranked_results、error_analysis、evaluation_lineageを保存する。targetなし行はdiagnostic_onlyである。
- document/query profile SHA-256不一致時は検索評価を行わず `preprocessing_profile_mismatch` とする。
- A/Bは同一query fingerprintを必須にし、Recall@1/5/20、MRRの全非低下、critical regressionなし、最低query数をgateにする。Recall@1またはMRRの+0.01未満はneeds_more_evidence。
- Phase順は辞書のみ、chunkのみ、morphologyのみ、keep/stopword一件ずつ、最後にactive確認済みTF-IDF/SVDであり、複数因子を同時変更しない。

### 本番実行条件

- MeCab + IPA UTF-8 version 102の実辞書pathを `--dictionary-path` で明示する。
- make_ceは `--run_type space` と `vectors` のみ、`--chunk_on n`。cleanを呼ばない。
- 本番query CSVを `--queries` に渡し、`--min-evaluated-queries` を本番最低件数へ設定する。
- 現PCはmake_ce依存、MeCab/Sudachi辞書、本番query実データが未共有のためproduction baseline metricsは未実測。fixtureでrank 1/5/20/圏外、metadata mismatch、profile mismatchを検証した。

本番baseline実行例:

```text
.venv/Scripts/python.exe auto_improve_lsa_lists.py --input <production_df0.csv> --queries <ground_truth_queries.csv> --text-col text --file-col name_org --page-col pageno --config-dir config --output-dir <run_dir> --tokenizer mecab_ipadic_utf8_v102 --dictionary-path <IPA_UTF8_V102_DIR> --morphology-profile content_lemma --chunk-profile current --vector-text-col text --min-evaluated-queries <minimum>
```

この実行は `baseline/eval/metrics.json`, `query_results.csv`, `ranked_results.csv`, `error_analysis.csv`, `evaluation_lineage.json` を生成する。

---

以下は更新前の履歴監査であり、経路が矛盾する場合は上記現行仕様を優先する。

## 今回の前提制約

- 対象repo rootは `/home/rai/move`。
- 現在のファイル構成・ファイル名は変更しない。
- `make_ce_v1.py` の依存ファイル欠損は、このPCへ未送付の可能性があるため、本番不具合とは断定しない。
- `df0.csv` / `lsa_ready.csv` の空列・列崩れは、共有用サンプルの不完全性や削除ミスの可能性があるため、それ単体で商用不可とは断定しない。
- 今回の主目的は、chunk済み成果物が downstream の検索・評価・改善ループで使われるコード構造になっているかの確認と、必要最小限の接続修正。

## 実在する入力ファイル

- input CSV: `df0.csv`
- queries CSV: `KLコンペ公開クエリ_拡張.csv`
  - `KLコンペ公開クエリ.csv` は現在の `/home/rai/move` には存在しないため、指定どおり `KLコンペ公開クエリ_拡張.csv` を使用対象とする。
- config dir: `config`
- make-ce script: `make_ce_v1.py`

補足: 現PCでは `df0.csv` と `lsa_ready.csv` の主要列が空として読まれる箇所があるが、これはサンプル不完全の可能性として扱い、chunk連携コード構造の評価とは分離する。

## chunk済みCSVの保持変数・パス

`auto_improve_lsa_lists.py` で以下のように保持される。

- 変数: `lsa_csv`
- パス: `<output_dir>/baseline/lsa_ready.csv`
- 作成関数: `lsa_preprocess_and_chunk.build_output(prep_args)`
- 書き出し: `safe_write_csv(prep_df, lsa_csv, "utf-8-sig", True, Path(args.input))`
- chunk列: `chunk_text`
- token列: `lsa_tokens_str`
- id列: `chunk_id_out`
- file/page列: `file_name_out`, `page_out`

## chunk済みCSVが使われていた箇所

コード上、chunk済みCSVは make_ce backend へ渡っていた。

1. `auto_improve_lsa_lists.py`
   - `lsa_csv = baseline / "lsa_ready.csv"`
   - `build_make_ce_vectors(parse_make_ce_args(["--lsa-ready", str(lsa_csv), ...]))`
   - `--text-col chunk_text`, `--chunk-id-col chunk_id_out`, `--file-col file_name_out`, `--page-col page_out` を渡す。

2. `make_ce_backend.py`
   - `args.lsa_ready` を `convert_lsa_ready_to_make_ce_df0()` へ渡す。
   - 変換後の出力は `{base_dir}/model_{craw_name}/step0/df0.csv`。
   - `make_ce_v1.py --run_type space` と `--run_type vectors` を呼ぶ。

3. `convert_lsa_ready_to_make_ce_df0.py`
   - `chunk_text -> text`
   - `chunk_text -> wakati_text`
   - `chunk_id_out -> chunk_id`
   - `file_name_out -> name_org`
   - `page_out -> pageno`

4. `evaluate_lsa_retrieval.py`
   - make_ce経路では `--backend make_ce_direct`。
   - `step2/df.pkl` または `step2/df.csv` を metadata として読む。
   - `--vector-text-col text` の列を `vectorizer.transform(docs_text)` に使う。

結論: 「chunk済みCSVを作っているだけで、make_ce build/evaluate が元の `df0.csv` を直接見ている」という構造ではない。chunk済み `lsa_ready.csv[chunk_text]` は make_ce用 `step0/df0.csv[text]` に変換され、その後の評価は `text` を見る設計になっている。

## 使われていなかった・断線していた箇所

修正前に断線していたのは主に2点。

1. `config-dir` 更新が次サイクルの前処理に反映されない
   - `apply_lsa_list_updates.py` は `list_keep.xlsx`, `list_stopword.xlsx`, `list_synonym.xlsx`, `df_replace.xlsx` を更新する。
   - しかし修正前の `auto_improve_lsa_lists.py` は `lsa_preprocess_and_chunk.py` へ `--config-dir` を渡していなかった。
   - そのため cycle 2 以降でも、更新済みconfigが chunk作成・正規化・token化に効かない構造だった。

2. `run_lsa_auto_improve_cycles.py` の `--vector-text-col` 既定値が make_ce metadata とズレていた
   - 修正前の既定値は `lsa_tokens_str`。
   - make_ce変換後の評価metadataで実際に見るべき列は `text`。
   - ユーザー前提CLIの `vector-text-col: text` と合わせる必要があった。

## 修正内容

ファイル構成・ファイル名は変更していない。CSV本体も修復・破壊変更していない。

- `lsa_preprocess_and_chunk.py`
  - `--config-dir` を追加。
  - `config-dir` 配下の `protected_terms.txt`, `remove_line_patterns.txt`, `safe_ja_stopwords.txt` を前処理で優先利用。
  - `list_keep.xlsx` を keep terms に反映。
  - `list_stopword.xlsx` を stopwords に反映。
  - `list_synonym.xlsx` を synonym 展開に反映。
  - `df_replace.xlsx` の enabled な preprocess 置換を正規化前に反映。
  - preprocess report に input path / output path / text column / config dir / config件数を記録。

- `auto_improve_lsa_lists.py`
  - `lsa_preprocess_and_chunk.py` へ `--config-dir args.config_dir` を渡す。
  - preprocess / make_ce / evaluate / mine の stageごとに input path / output path / text column / config dir をログ出力。
  - `baseline/lineage.json` を出力。
  - `--vector-text-col` 既定値は `text`。

- `run_lsa_auto_improve_cycles.py`
  - `run_one_cycle_pipeline()` のログに input / queries / config_dir / output_dir / text_col / vector_text_col を出力。
  - `--vector-text-col` 既定値を `text` に修正。
  - final metrics に input/query/text/config lineage を追加。
  - cycle 2以降の `temp_config` を `run_one_cycle_pipeline(args, temp_config, after_run)` へ渡す既存構造に、前処理側の `--config-dir` 接続が加わったため、更新configが次サイクルの chunk作成へ反映される。

- `make_ce_backend.py`
  - `lsa_ready.csv` から make_ce用 `step0/df0.csv` への変換ログと report に lineage を追加。

- `evaluate_lsa_retrieval.py`
  - make_ce評価時に model_dir / metadata path / vectorizer path / queries / output_dir / vector_text_col をログ出力。

- `build_lsa_vector_space.py`
  - legacy LSA build時に input / output_dir / text_col / id_col / display_col をログ出力。

## vector-text-col 既定値ズレ

- 修正前: `run_lsa_auto_improve_cycles.py` の既定値は `lsa_tokens_str`。
- 問題: make_ce経路の評価metadataは `text` を見る前提で、`lsa_tokens_str` は make_ce標準metadata列ではない。
- 修正後: `run_lsa_auto_improve_cycles.py` と `auto_improve_lsa_lists.py` の既定値を `text` に統一。
- 評価側: `evaluate_lsa_retrieval.py` は `choose_vector_text_col(metadata, args.vector_text_col)` で明示列 `text` を検証し、`metadata[text]` を `vectorizer.transform()` に渡す。

## config更新が次サイクルに反映されるか

修正後は反映されるコード構造になっている。

- `run_lsa_auto_improve_cycles.py`
  - before: `run_one_cycle_pipeline(args, working_config, before_run)`
  - approved適用: `apply_selected_to_temp(working_config, actions, validation, temp_config, cycle_dir)`
  - after: `run_one_cycle_pipeline(args, temp_config, after_run)`
  - accepted時: `copy_config(temp_config, working_config)`

- `auto_improve_lsa_lists.py`
  - `run_one_cycle_pipeline()` から渡された `config_dir` を `--config-dir` として受ける。
  - その `config_dir` を `lsa_preprocess_and_chunk.py` に渡す。

- `lsa_preprocess_and_chunk.py`
  - `config_dir` から txt設定と Excel設定を読み、正規化・置換・stopword・keep・synonymに反映する。

したがって、承認された config 更新は、次の `lsa_ready.csv` 生成に反映され、その後の make_ce build/evaluate に渡る。

## 修正後のデータ lineage

`df0.csv[text]`
-> `run_lsa_auto_improve_cycles.py`
-> `auto_improve_lsa_lists.py: lsa_csv = <cycle>/before|after/baseline/lsa_ready.csv`
-> `lsa_preprocess_and_chunk.py --input df0.csv --text-col text --config-dir <working_config|temp_config>`
-> `lsa_ready.csv[chunk_text, lsa_tokens_str, chunk_id_out, file_name_out, page_out]`
-> `make_ce_backend.py --lsa-ready <lsa_ready.csv> --text-col chunk_text`
-> `convert_lsa_ready_to_make_ce_df0.py`
-> `{base_dir}/model_{craw_name}/step0/df0.csv[text=chunk_text, wakati_text=chunk_text, chunk_id=chunk_id_out]`
-> `make_ce_v1.py --run_type space`
-> `make_ce_v1.py --run_type vectors`
-> `{base_dir}/model_{craw_name}/step2/df.pkl|df.csv`
-> `evaluate_lsa_retrieval.py --backend make_ce_direct --vector-text-col text`
-> `query_results.csv`, `metrics.json`
-> `mine_lsa_improvement_candidates.py`
-> `candidate_actions.csv`
-> `validate_lsa_candidate_actions.py`
-> `apply_lsa_list_updates.py`
-> next cycle config
-> next cycle `lsa_preprocess_and_chunk.py --config-dir <updated_config>`

## build_lsa_vector_space.py との関係

`build_lsa_vector_space.py` は legacy LSA backend 用の単体buildスクリプトで、現在の `run_lsa_auto_improve_cycles.py` の make_ce経路では直接呼ばれていない。単体で使う場合は `--input` と `--text-col` に渡されたCSV/列を読むため、chunk済みCSVを使うには `--input lsa_ready.csv --text-col lsa_tokens_str` のように明示する設計。

今回の主経路は make_ce backend なので、chunk済みCSVの downstream 利用は `make_ce_backend.py` と `evaluate_lsa_retrieval.py --backend make_ce_direct` で確認した。

## 現PCでの再現実行性スコア

55 / 100

理由:

- `py_compile` と `.venv/bin/python run_lsa_auto_improve_cycles.py --help` は成功。
- full run は、現PCに `make_ce_v1.py` の依存ファイルが無い可能性があるため未確認。
- 現PCの `df0.csv` / `lsa_ready.csv` は共有サンプル不完全の可能性があり、実データ品質の再現性評価には使わない。

このスコアは現PC上での実行再現性だけを表す。本番コード不具合の断定ではない。

## コード構造・chunk連携設計スコア

82 / 100

評価:

- chunk済みCSVの生成パスと downstream への受け渡しは追跡可能。
- `lsa_ready.csv[chunk_text] -> make_ce step0/df0.csv[text] -> evaluate[text]` の経路は明示された。
- `config-dir` 更新が次サイクルの前処理へ接続された。
- stageログと `baseline/lineage.json` により、使用path/column/config dirを追跡しやすくなった。

減点:

- `validate_lsa_candidate_actions.py` は候補を実際に適用して再検索評価する実測validationではない。
- make_ce成果物側の metadata に、元 `lsa_ready.csv` の全lineage列が保持される保証は限定的。
- `build_lsa_vector_space.py` は main multi-cycle 経路には未統合で、make_ce経路と legacy LSA経路が併存している。

## 商用プロダクト基準での評価

依存欠損やサンプルCSV不完全だけを理由に単純不可とはしない。chunk lineage / 再現ログ / validation の観点では、今回の修正後に「検証可能な構造」へ改善した。

商用基準で残る本質的課題:

1. validationが実測改善ループになっていない
   - 現在の `validate_lsa_candidate_actions.py` は候補を適用した検索再実行で score_delta を測る実装ではない。
   - 商用では、候補適用前後の同一query set評価と rollback根拠が必要。

2. lineageの永続性
   - `baseline/lineage.json` とログで追跡性は改善した。
   - ただし make_ce `step2/df.pkl|df.csv` に元chunkの `source_row`, `chunk_index`, `chunk_id_out` 等が確実に保持されるかは make_ce実装依存。

3. 環境再現性
   - `base-dir` 既定値が Windows path で、Linux上では明示指定が実質必須。
   - 本番相当の make_ce依存ファイル込みで smoke test を定義する必要がある。

4. public query評価の妥当性
   - target列がある場合は recall/mrr を計算できる。
   - targetが弱い/無い query set では、改善判定用の proxy metric を別途定義する必要がある。

5. 入力schema検証
   - `df0.csv` の必須列 `text`, `name_org`, `pageno` が存在し、非空率が許容範囲かを本番実行前に検証する軽量checkが必要。

## validate_lsa_candidate_actions.py の現状診断

修正前の `validate_lsa_candidate_actions.py` は実測検索validationではなかった。

- 候補アクションを candidate config に適用していなかった。
- before / after の検索・評価を再実行していなかった。
- `candidate_actions.csv` の `action`, `priority_score`, `evidence_count`, `term` と、baseline config の keep list だけで静的に `approve` / `reject` / `needs_review` を決めていた。
- `score_delta`, `mrr_delta`, `recall5_delta`, `recall1_delta` は常に `0.0` だった。
- `baseline_metrics.json` を読むコードはあったが、`output_dir/baseline_metrics.json` 固定で、呼び出し元の実評価 `baseline/eval/metrics.json` には接続されていなかった。

結論: 修正前は「候補改善アクションが検索結果を改善したか」を検証できない静的validationだった。

## validation 修正内容

`validate_lsa_candidate_actions.py` を、実測metricsが渡された場合だけ measured 判定する構造へ変更した。

追加・変更:

- `--baseline-metrics`
- `--candidate-metrics`
- `--baseline-results`
- `--candidate-results`
- `--candidate-config-dir`
- `--vector-text-col`
- `--min-score-delta`
- `--rollback-if-score-drops` / `--no-rollback-if-score-drops`

出力:

- 既存: `candidate_validation.csv`
- 追加: `candidate_validation_result.json`

`candidate_validation_result.json` に含める主な項目:

- `validation_mode`: `static_only` / `measured` / `insufficient_evidence`
- `baseline_config_path`
- `candidate_config_path`
- `query_path`
- `vector_text_col`
- `before_metrics`
- `after_metrics`
- `metric_delta`
- `before_proxy_metrics`
- `after_proxy_metrics`
- `proxy_metric_delta`
- `decision`
- `reason`
- `environment_blockers`

判定方針:

- before / after の target-based metrics が両方ある場合のみ `validation_mode=measured`。
- measured で composite score が改善し、必要なら recall_at_5 が落ちていない場合のみ `decision=approve`。
- metricsが片側だけ、または public query 等で recall/mrr が `None` の場合は `insufficient_evidence`。
- metrics/resultsが無い場合は `static_only`。
- 実測未実行の候補は `approve` しない。静的に有望な候補は `needs_real_eval` または `insufficient_evidence` として扱う。

`auto_improve_lsa_lists.py` からは、baseline の `metrics.json` と `query_results.csv`、および `vector_text_col` を validation に渡すようにした。candidate metrics はこの段階では存在しないため、通常は `insufficient_evidence` として記録される。これは「実測していないのに改善と判定しない」ための挙動。

## candidate config A/B validation 構造

今回、candidate action から candidate config を生成し、baseline/candidate を比較できる最小構造を追加した。

candidate action の生成:

- `auto_improve_lsa_lists.py` が baseline 評価結果 `baseline/eval/query_results.csv` を `mine_lsa_improvement_candidates.py` に渡す。
- `mine_lsa_improvement_candidates.py` が `candidates/candidate_actions.csv` を作る。

candidate config の生成:

- 生成場所: `<output_dir>/candidate_config`
- `auto_improve_lsa_lists.py` が baseline config dir を `candidate_config` にコピーする。
- `apply_lsa_list_updates.py` に追加した `apply_candidate_actions_for_eval()` が `candidate_actions.csv` の上位 `--max-candidates` 件を isolated candidate config に反映する。
- これは評価用config生成であり、元の `config` へは反映しない。承認も意味しない。

candidate metrics の生成:

- baseline と同じ `run_retrieval_measurement()` 経路を使う。
- 入力は同じ `args.input`, `args.queries`, `args.vector_text_col`。
- config dir だけを baseline config から `<output_dir>/candidate_config` に差し替える。
- 成果物:
  - baseline metrics: `<output_dir>/baseline/eval/metrics.json`
  - baseline results: `<output_dir>/baseline/eval/query_results.csv`
  - candidate metrics: `<output_dir>/candidate_measurement/eval/metrics.json`
  - candidate results: `<output_dir>/candidate_measurement/eval/query_results.csv`

現PC環境制約で candidate measurement が失敗した場合:

- `validation/candidate_measurement_error.json` を出す。
- `validate_lsa_candidate_actions.py` には candidate metrics を渡さない。
- measured 扱いにはせず、`insufficient_evidence` とする。

## A/B measured 判定できる条件

`validate_lsa_candidate_actions.py` が `validation_mode=measured` と判定する条件:

1. baseline metrics JSON が存在する。
2. candidate metrics JSON が存在する。
3. `recall_at_1`, `recall_at_3`, `recall_at_5`, `mrr` の少なくとも一部が before/after で数値として読める。

measured判定の主指標:

- composite score
- `recall_at_1`
- `recall_at_5`
- `mrr`

reject条件の入口:

- composite score が `--min-score-delta` 以上改善しない。
- `--rollback-if-score-drops` 有効時に `recall_at_5` が低下する。
- `critical_query_regression=true` になる。

`critical_query_regression`:

- baseline/candidate の `query_results.csv` に `query`, `rank`, `is_hit` がある場合に判定する。
- baselineでhitしていたqueryがcandidateでhitを失った場合、または `--critical-rank-drop` 以上rank悪化した場合に true。
- 現時点では入口実装であり、critical query の重み付けや業務重要度リストは未実装。

## 実測できない場合の扱い

現PCでは make_ce依存ファイル未送付やサンプルCSV不完全の可能性があるため、full run できない場合がある。この場合も本番不具合とは断定せず、`candidate_validation_result.json` の `environment_blockers` に不足情報を残す設計にした。

実測validationを行うには、少なくとも以下が必要:

1. baseline config で検索評価した `metrics.json`
2. candidate config を適用した状態で検索評価した `metrics.json`
3. 同一 query set
4. 同一 `vector_text_col`

public query で target-based metrics が無い場合、`query_results.csv` から top1 score / jaccard の proxy は算出できる。ただし scoreの較正や妥当性に限界があるため、自動 `approve` には使わず、`insufficient_evidence` として扱う。

## validation設計スコア

78 / 100

加点:

- 静的validationと実測validationを区別できる。
- before / after metrics がある場合のみ改善判定する。
- 実測未実行で `approve` しない。
- JSONで validation mode, metrics, deltas, blockers を追跡できる。
- 呼び出し元から baseline metrics/results と `vector_text_col` が渡る。
- candidate action から isolated candidate config を生成できる。
- baseline/candidate を同じ retrieval measurement 経路で評価できる。
- candidate measurement 失敗時に error JSON を残し、measured 扱いしない。
- `critical_query_regression` の入口がある。

減点:

- 候補ごとの個別A/Bではなく、候補セット単位のA/B。
- proxy metric は記録のみで、自動承認には使わない。
- make_ce依存込みの実測runは現PC環境制約により未確認。
- critical query の業務重要度リストや重み付けは未実装。

## コード構造・chunk連携設計スコア再評価

86 / 100

前回 84 / 100 から微増。candidate config 生成と candidate measurement の lineage が `auto_improve_lsa_lists.py` に追加され、baseline/candidate metrics path が validation に接続されたため。ただし個別候補A/Bや本番依存込みのfull run確認は未完了。

## validation観点で残る商用課題

1. candidate config の自動作成と検索再実行
   - 候補セット単位では実装済み。
   - 商用では candidateごとの個別評価が必要。

2. 候補単位のA/B評価
   - 複数候補をまとめて評価すると、どの候補が改善・劣化に寄与したか分離できない。

3. rollback根拠の強化
   - recall_at_5 drop だけでなく、重要query群・doc/page hit・業務上critical queryの保護が必要。

4. public query向け評価設計
   - targetが無い場合に、安全に自動承認できる指標はまだ無い。
   - proxy metric は診断用に留めるのが妥当。

## 確認コマンド

- `python -m py_compile run_lsa_auto_improve_cycles.py lsa_preprocess_and_chunk.py build_lsa_vector_space.py evaluate_lsa_retrieval.py auto_improve_lsa_lists.py apply_lsa_list_updates.py make_ce_v1.py make_ce_backend.py`
  - 成功。

- `.venv/bin/python run_lsa_auto_improve_cycles.py --help`
  - 成功。
  - `--input`, `--queries`, `--text-col`, `--file-col`, `--page-col`, `--config-dir`, `--output-dir`, `--base-dir`, `--craw-name`, `--make-ce-script`, `--top-k`, `--chunk-profile`, `--vector-text-col`, `--cycles`, `--skip-space`, `--skip-vectors` 等を確認。

- `python -m py_compile validate_lsa_candidate_actions.py auto_improve_lsa_lists.py evaluate_lsa_retrieval.py run_lsa_auto_improve_cycles.py`
  - 成功。

- `.venv/bin/python validate_lsa_candidate_actions.py --help`
  - 成功。
  - measured validation 用の `--baseline-metrics`, `--candidate-metrics`, `--baseline-results`, `--candidate-results`, `--candidate-config-dir`, `--vector-text-col` を確認。

- `.venv/bin/python validate_lsa_candidate_actions.py ... --baseline-metrics /tmp/lsa_validate_smoke/before_metrics.json --candidate-metrics /tmp/lsa_validate_smoke/after_metrics.json --max-candidates 1`
  - 成功。
  - `/tmp/lsa_validate_smoke/out/candidate_validation_result.json` に `validation_mode: measured`, `metric_delta.composite_score: 100.0`, `decision: approve` が出力されることを確認。
  - これは make_ce full run ではなく、validation CLI 単体の軽量確認。

- `.venv/bin/python auto_improve_lsa_lists.py --help`
  - 成功。

- `.venv/bin/python validate_lsa_candidate_actions.py ... --baseline-results /tmp/lsa_validate_ab_smoke/before_results.csv --candidate-results /tmp/lsa_validate_ab_smoke/after_results.csv`
  - 成功。
  - `candidate_validation_result.json` に `validation_mode: measured`, `critical_query_regression: false`, `proxy_metric_delta`, `decision: approve` が出力されることを確認。

full run / pytest:

- 現PCでは make_ce依存ファイル未送付の可能性とサンプルCSV不完全の可能性があるため、chunk連携コード評価とは分離し、「環境制約による未確認」とする。

## 2026-07-10 tuning / recall@20 追加

追加内容:

- `evaluate_lsa_retrieval.py` が `metrics.json` に `recall_at_20` を出力する。
- `validate_lsa_candidate_actions.py` が `recall_at_20` を `metric_delta` と `missing_metrics` に含める。
- measured 以外では approve しない仕様は維持。
- `critical_query_regression=true`、`recall_at_5` 悪化、`mrr` 悪化は reject する安全側判定を維持・強化。
- `auto_improve_lsa_lists.py` に `--enable-param-tuning`, `--tuning-profile minimal`, `--max-tuning-candidates`, `--tokenizer` を追加。
- tuning候補の lineage は `<output_dir>/tuning/tuning_validation_result.json` と `<output_dir>/tuning/tuning_leaderboard.csv` に保存する。

chunk lineage / config lineage:

- `lsa_ready.csv[chunk_text] -> make_ce step0/df0.csv[text] -> evaluate[text]` は維持。
- `vector-text-col=text` は維持。
- `config-dir` は baseline/candidate/tuning candidate の前処理に渡る。
- tuning candidate config は `<output_dir>/tuning/<tuning_candidate_id>/candidate_config/` に isolated copy として生成し、元 `config` は変更しない。

パラメータ接続:

- `tokenizer` は `lsa_preprocess_and_chunk.py --tokenizer` に接続済み。
- `chunk_profile` は `lsa_preprocess_and_chunk.py --chunk-profile` に接続済み。
- `min_df`, `max_df`, `svd_dim` は legacy `build_lsa_vector_space.py` では効くが、現在の make_ce backend 主経路では `make_ce_backend.py` から `make_ce_v1.py` へ渡すCLIが無いため inactive として記録する。
- inactive params は measured improvement の原因として扱わない。

確認結果:

- `/tmp/lsa_tuning_smoke` の小規模 `results_csv` 評価で `recall_at_20` が出ることを確認。
- 同 before/after metrics を `validate_lsa_candidate_actions.py` に渡し、`validation_mode=measured`, `decision=approve`, `missing_metrics=[]` を確認。
- tuning result の失敗系 smoke で、候補測定不可時に `validation_mode=insufficient_evidence`, `decision=needs_measurement`, `environment_blockers` が出ることを確認。

スコア再評価:

- validation設計スコア: 84 / 100
- tuning設計スコア: 74 / 100
- chunk連携設計スコア: 87 / 100

残課題:

1. make_ce主経路で `min_df`, `max_df`, `svd_dim` を実際に制御する接続。
2. tuning候補ごとの `critical_query_regression` 実計算。
3. full run を本番相当 make_ce依存込みで確認する smoke test。
4. tokenizer依存欠損時の候補skipと環境診断の強化。

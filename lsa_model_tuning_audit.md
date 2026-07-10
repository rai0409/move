# LSA Model Tuning Audit

## 2026-07-10 実装・実測更新（この節を現行仕様とする）

### 結論

- Windows本番baselineは `mecab_ipadic_utf8_v102`（MeCab + IPA辞書、charset UTF-8、dictionary version 102）に固定した。Python経路は fugashi wrapper を使うが、fugashiは別モデルではない。同一辞書・同一MeCab option・同一品詞profileならMeCab directとの検索性能候補は重複させない。
- regexは後方互換テスト用コードだけ残し、既定baseline・production candidate・tuning candidateから除外した。MeCab/Sudachiのimport、辞書path、辞書packageが欠けてもregexへfallbackしない。
- 現PCには fugashi/MeCab/SudachiPy/sudachidict-small/sudachidict-coreがなく、IPA v102 pathも未設定である。自動downloadは行っていないため、全形態素candidateは `not_measured`、`no_candidate_approved` である。
- `lsa_ready.csv` と公開query CSVはヘッダーと空行のみでtargetがない。recall@1/5/20、MRRを計算可能なfull評価データがないため、指標はnullであり、未実測approveはない。

### 確定した実フロー

1. `lsa_preprocess_and_chunk.py::build_output()` が指定CSVを読み、`--text-col` を `normalize_text()` でNFKC・制御文字・空白・改行正規化する。
2. config-dirの `protected_terms.txt`、`remove_line_patterns.txt`、`safe_ja_stopwords.txt` および存在時の `list_keep.xlsx`、`list_stopword.xlsx`、`list_synonym.xlsx`、`df_replace.xlsx` を毎回読む。このため次cycleへ設定変更が反映される。
3. 正規化後、tokenize前に `chunk_text()` がchunk化する。`chunk_text`を形態素処理して `lsa_tokens_str` を作る。
4. legacy経路は `build_lsa_vector_space.py` が `lsa_tokens_str` をTF-IDFへ渡し、その疎行列をTruncatedSVDへ渡す。candidateごとにvectorizerをfitし直す。
5. make_ce主経路は `make_ce_backend.py` → `convert_lsa_ready_to_make_ce_df0.py` で `lsa_tokens_str -> df0.csv[text]` とする。`chunk_text -> chunk_text/original_text`、`docid -> docid`、`chunk_id_out -> chunk_id`、`file_name_out -> name_org`、`page_out -> pageno` は表示・正解判定・lineage用に保持する。CSV本体のin-place変更はしない。
6. `make_ce_v1.py` は `make_clean_df`、`make_space_v1_r2`、`make_vectors_v1_r2` が未共有のため現PCで起動不能。`vector-text-col=text` は維持されている。make_ceの保存済みvectorizerが実際に参照するTF-IDF/SVD詳細は推測で変更していない。
7. legacy評価はmodel directoryの `preprocessing_profile.json` を読み、queryにも文書と同じnormalize/tokenizeを適用する。profileのない旧modelだけは「既に分かち書き済みquery」と明示的に扱う。make_ce_directは文書 `text` とqueryの双方を同じ保存済みvectorizerへ渡す。

### 形態素解析・品詞・原形

- Phase 1候補は `mecab_ipadic_utf8_v102`、`sudachi_small_b/c`、`sudachi_core_b/c`、profileはすべて `content_lemma`、chunkは `current` 固定。
- MeCabは `--dictionary-path` を必須にし、`-d <path>` と `--mecab-options` をlineageへ記録する。version 102・UTF-8以外はblocker。暗黙system dictionary選択は禁止。
- Sudachiはsmall/core package存在を別々に検査し、SplitMode B/C、package version/path、normalized_form、OOV surfaceを記録する。
- `content_lemma`: 一般/固有/サ変接続を含む名詞はsurface、動詞・形容詞はlemma、助詞・助動詞・副詞・記号は除外、未知語はsurface。`noun_only`、`content_surface`、`surface_plus_lemma` はPhase 3候補として実装したがPhase 1未完了のため未比較。
- 数詞は削除しない。URL、email、`ABC-123`型番、第10条、2026年度、3ページ、100万円を形態素列に加えてcompound tokenとして注入する。ASCII部分は文書/queryともlower化する。単独記号は除外し、型番内hyphen/slashは保持する。
- 表記揺れはNFKCとSudachi normalized_form（Sudachi候補時）のみを自動適用する。「ユーザー/ユーザ」「申込み/申し込み/申込」「問い合わせ/問合せ」は無根拠な一括置換をせず、config候補をquery単位A/B後に採用する。

### keep words / stopwords

- 優先順位は keep/protected phrase → 品詞選択 → 最小token長 → stopwords。keep wordsは品詞除外、1文字除外、stopword除外より常に優先する。
- 複合keep語は通常の形態素unigramに加え、空白をunderscoreにしたprotected phraseも注入する。元textに実在するphraseだけを注入する。
- 現在の `safe_ja_stopwords.txt` は22語、すべて日本語、英語stopwordは0。keepとの重複はload時に差し引く。現CSVが空なので文書頻度、query頻度、正解頻度、削除時query悪化は未実測であり、新stopwordは追加していない。

### chunkの旧問題と現行profile

- 旧実装はtarget超過時は文/読点単位だったが、hard max時とoverlap時は固定文字位置で切り、文途中切断が起こり得た。offset、文番号、段落番号、見出しlineageもなかった。
- 現実装は固定文字sliceを廃止し、長い単一文は切らず `oversized=true` で可視化する。overlapは直前chunk末尾の完全な1〜2文、paragraph profileは完全な1段落を使う。
- `sentence_boundary_medium` (750/900/100)、`paragraph_boundary_medium` (800/1000/100)、`heading_aware` (750/900/100)、`compact` (500/650/65)、`broad` (1000/1300/150) を追加した。数値はtarget/max/overlap文字数。
- 見出しprofileは見出しだけのchunkを作らず子chunkへ複製する。入力1行がpage単位なのでpageを跨がず、name/page/chunk_idを維持する。`source_char_start/end`、sentence/paragraph start/end、source_heading、overlap_charsをlineageへ追加した。
- 反復文を含む1文書smokeの統計は `chunk_quality.csv` に保存した。これは構造smokeであり検索性能比較ではない。compactのnear-duplicateが高いのは反復fixtureとoverlapの影響で、approve材料にはしていない。

### TF-IDF / SVD 接続状況

legacyで有効: `min_df`、`max_df`、`sublinear_tf`、`norm`、`use_idf`、`smooth_idf`、`binary`、`ngram_range`、100次元baseline、`random_state=42`、`n_iter=5`、`algorithm=randomized`。空白tokenizer、`preprocessor=None`、`token_pattern=None`、`lowercase=False` を明示し、1文字日本語とhyphen型番をvectorizerが再分割しない。説明分散、語彙数、build時間を保存する。n_componentsは `min(requested, vocabulary-1, documents-1)` に制限する。

make_ce主経路で未接続/未確認: `min_df`、`max_df`、上記TF-IDF flags、ngram、SVD次元、seed、n_iter、algorithm、explained variance。`--n-clusters` はSVD次元ではない。このためPhase 5候補へ含めずinactive paramsとする。本番共有後はmake_space/make_vectorsのCLIまたは最小adapter入口へ50/100/150/200を接続し、300は規模確認後に限る。

2026-07-10の評価ループ更新で `lsa_tokens_str -> df0[text]` を正式接続したため、外側のMeCab/Sudachi・morphology変更はmake_ce主経路でもactiveである。converterはsource/destination SHA-256、row count、empty token rowsを記録し、`chunk_text` がvector入力へ入っていないことをtestする。make_ce clean/chunkはともにOFFである。

### 正解付きquery評価（2026-07-10追加）

- schemaは `query`, `target_doc`, `page`, `docid`, `expected_text_contains`, `model3の順位`, `コメント`。列として `query` と `target_doc` を必須、さらにpage/docid/textの少なくとも1列を必須とする。行単位でtarget_docと補助正解がない場合はdiagnostic_onlyでrecall分母から除外する。
- target_doc一致は必須。pageがあればexactまたは明示range一致、expected textがあれば評価専用NFKC・空白改行除去・ASCII lower・最小句読点除去後の包含を必須とする。docidはpage/textがない場合の補助根拠で、chunk変更後のdocid不一致だけでは失敗にしない。
- `query_results.csv` は1 query 1行でbest rank、Recall@1/5/20 hit、RR、matched metadata、matched_by、top1、failureを保存する。全rankは `ranked_results.csv`、失敗は `error_analysis.csv` に保存する。
- recall分母はground truthを満たす行だけ。target_doc_onlyは弱い診断で正解に数えない。failure categoryはno target doc、wrong page、text not contained、tokenizer mismatch、empty query tokens、ambiguous docs、metadata mismatch、no ground truth、other。
- baseline/candidateはquery CSV SHA-256を比較し、document/query preprocessing fingerprintが一致しない場合は `preprocessing_profile_mismatch` で評価をblockする。
- approveはmeasured、最低query数以上、Recall@1/5/20とMRRの全て非低下、critical regressionなしを必須とし、Recall@1またはMRRが0.01以上改善した場合のみ。無悪化でも閾値未満はneeds_more_evidence。

### 段階A/Bと選定gate

- Phase 1: current chunk + content_lemmaを固定し辞書だけ比較。
- Phase 2: Phase 1 winnerを固定しcurrent + 5 chunk profileを比較。
- Phase 3: tokenizer/chunk固定で4 morphology profileを比較。
- Phase 4: stopword/keep候補を一件ずつA/B。
- Phase 5: make_ce本体接続確認後のみTF-IDF/SVDを比較。
- 必須出力はrecall@1/5/20、MRR、composite、mean/median rank、no-hit数、query別rank。targetなしqueryはdiagnostic_onlyでrecall分母から除外する。
- `critical_query_regression=true`、recall@1低下、recall@5低下、MRR低下はreject。recall@20だけの上昇ではapproveしない。未実測、proxyのみ、targetなしもapproveしない。

### 商用プロダクト基準スコアと次の優先改善

- 前処理設計: 86/100（共通profileと保護patternは接続済み。実辞書でのNFKC/normalized_form回帰が未実測）。
- chunk設計: 84/100（境界・overlap・lineage・統計は実装済み。実コーパスの回答span/page/tableで未評価）。
- tokenizer/dictionary設計: 78/100（候補・辞書固定・no fallbackは実装済み。全辞書が環境blocker）。
- tuning設計: 82/100（段階gate・target分離・critical rejectは実装済み。make_ce内部paramsとground-truth評価が未接続）。
- 次の最優先はWindows本番でIPA v102の実path/charset/versionを取得してPhase 1 baselineを測ること。その後、利用を承認された既存Sudachi packageだけを同一入力/queryで測る。続いて実文書でtable/箇条書き/見出し判定、stopword DF/query/target頻度を出す。

---

以下は更新前の履歴分析であり、上記現行仕様と矛盾する箇所は上記を優先する。

## 分析前提

- 分析専用。コード変更はしない。
- 既存モデルの大枠は TF-IDF -> TruncatedSVD による LSA 検索。
- embedding, BM25, 外部LLM, FAISS 等への置き換えは対象外。
- 既存の chunk lineage、`vector-text-col=text`、candidate A/B validation 構造は維持する。
- 現PCでの make_ce 依存欠損やサンプルCSV不完全は環境制約として扱い、モデル改善余地の評価とは分ける。

## 分析したファイル

- `run_lsa_auto_improve_cycles.py`
- `lsa_preprocess_and_chunk.py`
- `build_lsa_vector_space.py`
- `evaluate_lsa_retrieval.py`
- `auto_improve_lsa_lists.py`
- `validate_lsa_candidate_actions.py`
- `apply_lsa_list_updates.py`
- `make_ce_backend.py`
- `convert_lsa_ready_to_make_ce_df0.py`
- `config/safe_ja_stopwords.txt`
- `config/protected_terms.txt`
- `config/remove_line_patterns.txt`

## 現在の検索モデル構造

主経路は make_ce backend。

1. `lsa_preprocess_and_chunk.py`
   - `df0.csv[text]` を正規化。
   - chunk化。
   - token化して `lsa_tokens_str` を作る。
   - `chunk_text`, `chunk_id_out`, `file_name_out`, `page_out` を含む `lsa_ready.csv` を作る。
2. `make_ce_backend.py`
   - `lsa_ready.csv` を `convert_lsa_ready_to_make_ce_df0.py` へ渡す。
3. `convert_lsa_ready_to_make_ce_df0.py`
   - `chunk_text -> text`
   - `chunk_text -> wakati_text`
   - `chunk_id_out -> chunk_id`
   - `file_name_out -> name_org`
   - `page_out -> pageno`
4. `make_ce_v1.py`
   - `--run_type space` / `--run_type vectors` を実行。
5. `evaluate_lsa_retrieval.py --backend make_ce_direct`
   - `step2/vectorizer.pkl` と `step2/df.pkl|df.csv` を読む。
   - `--vector-text-col text` で `metadata[text]` を `vectorizer.transform()` する。

legacy経路として `build_lsa_vector_space.py` も存在する。こちらは `lsa_ready.csv[lsa_tokens_str]` を直接 TF-IDF -> SVD 化する構造。

## MeCab / tokenizer の現状

`lsa_preprocess_and_chunk.py` には tokenizer として `regex`, `fugashi`, `sudachi_a/b/c` がある。

現状の `auto_improve_lsa_lists.py` は前処理を呼ぶ際に `--tokenizer regex` を固定している。つまり main auto-improve 経路では MeCab/fugashi は使われていない。

`fugashi_tokenize()` の仕様:

- `fugashi.Tagger()` を使う。
- `str(word.feature).split(",")` の先頭を品詞として見る。
- 採用品詞は `名詞`, `動詞`, `形容詞` のみ。
- 固有名詞、副詞、記号、接頭辞、未知語などの細分類制御はない。
- 原形化はしていない。surface をそのまま採用する。

`sudachi_*` の仕様:

- SplitMode A/B/C を選べる。
- 採用品詞は同じく `名詞`, `動詞`, `形容詞` のみ。
- 原形化はしていない。

`regex_tokenize()` の仕様:

- ASCII英数字、数字混じり、カタカナ2文字以上、漢字2文字以上、日本語2文字以上を拾う。
- ASCIIを含むtokenは lower 化。
- `_`, `-`, `/`, `.`, camelCase, 英数字境界で追加分割する。

recall低下リスク:

- main経路が `regex` 固定なので、MeCab分かち書き前提のデータでない場合、日本語複合語が過大tokenになる可能性がある。
- fugashi/sudachiを使っても原形化がないため、活用差が残る。
- 名詞/動詞/形容詞のみなので、固有名詞が名詞扱いで取れればよいが、未知語や記号混じりの業務IDが落ちる可能性がある。
- ASCII完全一致のみ lower 化なので、英数字混在語 `API連携` のような語は lower 化されない。検索側と本文側で `API連携` / `api連携` が揺れると落ちる可能性がある。
- token dedupe がchunk内頻度を消すため、TFの強弱が弱くなる。recall@20にはプラスの場合もあるが、recall@1には重要語の繰り返しシグナルを失う可能性がある。

## 前処理の現状

正規化:

- Unicode NFKC。
- Unicode空白を通常空白へ。
- zero-width文字を削除。
- `[NL]`, `\\n`, CRLF を改行へ。
- 英単語のハイフン改行を結合。
- 日本語文字間の空白を除去。
- 日本語と英数字の境界に空白を挿入。
- 連続空白と過剰改行を整理。
- `remove_line_patterns` で空行、数字だけ、ページ番号、Copyright等を削除。

設定反映:

- `config/protected_terms.txt`
- `config/remove_line_patterns.txt`
- `config/safe_ja_stopwords.txt`
- `list_keep.xlsx`
- `list_stopword.xlsx`
- `list_synonym.xlsx`
- `df_replace.xlsx`

recall低下リスク:

- NFKCは基本的に有効。ただし型番や記号付きIDで意味ある記号を丸めすぎる可能性がある。
- 日本語と英数字の境界に空白を入れるため、`API連携` は `API 連携` として扱われる場合がある。クエリ・本文の両方で同じなら良いが、語として保持したい場合は compound/keep が必要。
- URLやメール、ファイルパスは専用処理がない。`/`, `.`, `-` で分割されるため、完全一致が必要な識別子は弱い。
- ひらがな4文字以上を keep 以外で落とすため、「できない」「ください」等のノイズは減るが、業務上の状態語・否定語を落とす危険がある。`RISKY_KEEP_WORDS` に否定・範囲語は一部保護されている。

## chunk分割ロジック

profiles:

- `none`: chunkなし。
- `small`: target 450 chars, overlap 60, hard_max 700, min 120。
- `medium`: target 750, overlap 100, hard_max 1100, min 180。
- `large`: target 1050, overlap 150, hard_max 1500, min 250。
- `auto`: Q75 * 0.9 を target にし、320-900にclip。overlapはtargetの15%、40-120にclip。hard_maxは max(Q90*1.1, target*1.6), floor 800。

分割単位:

- 改行/段落で分ける。
- `。！？!?、，,;：:` を境界にする。
- target超過時にchunk確定。
- hard_max超過時は文字数slice。
- 前chunk末尾から overlap 文字を次chunk先頭へ付与。

lineage:

- `chunk_text` は `lsa_ready.csv` に保存される。
- make_ce経路で `chunk_text -> text` に変換される。
- 評価側は `vector-text-col=text` を見る。

recall観点:

- recall@1: chunkが大きすぎると不要語が混ざり、top1がぼやける。450-750 chars付近が候補。
- recall@5/20: chunkが小さすぎると文脈語が不足し、別表現・同義語のSVD近傍が弱くなる。750-1050 charsやoverlap増が候補。
- overlapは現在15%相当。境界またぎの回答を拾うには有効だが、重複chunkが増えすぎると上位を同一文書近傍が占有してrecall@5/20を悪化させる可能性がある。

## TF-IDF設定

legacy `build_lsa_vector_space.py`:

- `TfidfVectorizer(analyzer=analyzer)`
- `analyzer(text) = text.split()`
- `min_df=3`
- `max_df=0.95`
- `svd_dim=150`
- tokenizer/preprocessor/token_pattern は無効。
- `ngram_range`, `sublinear_tf`, `norm`, `use_idf`, `smooth_idf` は明示なし。scikit-learn defaultに依存。
- chunk内tokenは前処理由来で dedupe 済み。

make_ce経路:

- `make_ce_v1.py` の依存本体は現repoに揃っていないため詳細は未確認。
- `evaluate_lsa_retrieval.py` は make_ce の保存済み vectorizer を読み、metadataの `text` を transform して評価する。

recall低下リスク:

- `min_df=3` は小規模・固有名詞・型番・英数字略語を落としやすい。recall@1/5では特に痛い。
- `max_df=0.95` は妥当だが、業務コーパスでほぼ全chunkに出る「申請」「対象」「確認」などは残りすぎる可能性がある。
- token dedupe によりTFが0/1に近くなり、`sublinear_tf` の恩恵以前に頻度情報が消える。
- unigramのみだと「操作 PC」「LINE WORKS」「Google Workspace」などのphraseが弱い。既存の `use_noun_compounds` はあるが main経路で有効化されていない。

## TruncatedSVD設定

legacy経路:

- default `svd_dim=150`。
- 実使用は `min(requested, n_features-1, n_docs-1)`。
- `random_state=42`。
- vectorsはL2 normalize。
- explained varianceは build report に出ていない。

make_ce経路:

- `run_lsa_auto_improve_cycles.py` / `auto_improve_lsa_lists.py` の `--n-clusters` はクラスタ数であり、SVD次元制御ではない。
- make_ce側のSVD次元制御は現repoからは確認できない。

recall観点:

- 次元が低すぎると固有名詞・型番・略語が潰れ、recall@1が落ちる。
- 次元が高すぎるとノイズ語やOCR揺れを拾い、recall@5/20は上がるがtop1安定性が落ちる場合がある。
- 100前後は一般的候補だが、chunk数・語彙数・専門語密度に応じて 50/100/150/200/300 のA/Bが必要。
- explained variance、query別rank変動、seed安定性を記録すべき。

## stopwords / keep words

現在の `config/safe_ja_stopwords.txt`:

- 助詞・指示語・汎用語中心。
- 比較的安全。

現在の `config/protected_terms.txt`:

- `LINE WORKS`, `Microsoft Teams`, `Google Workspace`, `RFIDリーダー`, `操作用PC`, `三菱UFJ`, `UFJ銀行`, `KIBIT`

リスク:

- stopword候補を自動追加する場合、`必要`, `対象`, `不可`, `未満` など業務判定語を落とすとrecallだけでなく正誤が崩れる。
- 英語略語や英数字混在語は keep 側へ積極的に入れるべき。
- protected_terms は flags に使われているが、tokenizer上で複合語として必ず保持する実装には見えない。`LINE WORKS` は分割後 `line`, `works` になる可能性が高い。

stopword候補の出し方:

- 高doc_freqかつmiss_termsに出ない。
- top1/top5 hit_textにもtarget判定に寄与しない。
- A/Bで recall@1/5/20 と critical query regression が悪化しない。

keep候補の出し方:

- missing_termsに頻出。
- 英字大文字、数字、記号、カタカナ、漢字+英字混在。
- target_doc/pageが外れたqueryで、query側にはあるがhit_text側にない。
- protected_termsや商品名/部門名/システム名。

## ボトルネック候補

recall@1/5/20を上げるうえで効きそうな順。

1. main経路の tokenizer が `regex` 固定
   - 日本語本文ではMeCab/fugashi/Sudachiの方が語彙品質が上がる可能性が高い。
   - ただし辞書差・未知語分割で固有名詞が壊れる危険もあるためA/B必須。

2. `min_df` が固有語・略語を落とす可能性
   - legacyでは default 3。
   - make_ce側も同等の語彙絞りがあるなら最重要チューニング対象。

3. phrase/compound不足
   - `LINE WORKS`, `操作用PC`, `RFIDリーダー`, `API連携` などが単語分割だけだと弱い。
   - `--use-noun-compounds` や protected term injection をA/Bしたい。

4. chunk粒度
   - top1重視なら `small` / `medium`。
   - recall@20重視なら `medium` / `large` / overlap増。
   - 現在 run_lsa は `auto`, auto_improve単体は `none` default。entrypointで差がある点は評価時に注意。

5. SVD次元
   - 固有語を保持したいなら 150/200/300。
   - ノイズが多いなら 50/100。

6. query側前処理とdoc側前処理の非対称
   - evaluateではqueryを保存済み vectorizer に直接 transform している。doc作成時の正規化・同義語展開・tokenizerと完全一致している保証が弱い。
   - 特に make_ce_direct では query preprocessing が make_ce vectorizer内部依存。

## 変更してよい範囲 / 変更してはいけない範囲

よい範囲:

- tokenizer選択のA/B。
- chunk profile / target / overlap / hard_max のA/B。
- min_df / max_df / SVD次元のA/B。
- stopwords / keep words / synonyms / df_replace の候補生成とA/B。
- `recall@20` メトリクス追加。
- report/log/lineage拡張。

禁止範囲:

- モデルをembedding/BM25/FAISS/LLMに置き換える。
- ファイル構成・ファイル名変更。
- CSV本体の破壊的修正。
- 既存chunk lineageやcandidate A/B validationを壊す変更。

## 推奨パラメータ候補

tokenizer:

- 現状維持: `regex`
- 候補: `fugashi`
- 候補: `sudachi_b`
- 候補: `sudachi_c`

chunk:

- `small`: target 450, overlap 60
- `medium`: target 750, overlap 100
- `large`: target 1050, overlap 150
- custom候補:
  - 600 / overlap 90 / hard_max 900
  - 900 / overlap 150 / hard_max 1300
  - 1200 / overlap 180 / hard_max 1700

TF-IDF:

- `min_df`: 1, 2, 3, 5
- `max_df`: 0.90, 0.95, 0.98
- `sublinear_tf`: False / True
- `ngram_range`: unigramのみ維持を基本。導入するなら word bigram ではなく前処理側 compound を優先。

SVD:

- `n_components`: 50, 100, 150, 200, 300
- 採用判定は recall@1/5/20, mrr, critical regression, explained variance。

stopwords:

- 候補抽出条件:
  - doc_freq_ratio >= 0.6
  - missing_terms頻度が低い
  - keep/protectedに含まれない
  - A/Bで recall@1/5/20 が悪化しない

keep words:

- 英語略語: API, CSV, PDF, PC, RFID, URL, ID
- 業務語: 申請, 承認, 対象, 対象外, 必要, 不要, 例外, 期限, 権限
- 複合語: API連携, 操作用PC, RFIDリーダー, LINE WORKS, Google Workspace, Microsoft Teams
- 組織/製品/システム固有名詞。

## 自動改善ループに組み込む順番

1. 評価メトリクス拡張
   - `recall@20` を追加。
   - query別 rank before/after を保存。

2. 安全な語彙候補
   - keep words候補。
   - protected/compound候補。
   - synonym/replace候補。

3. chunk A/B
   - small/medium/large/auto。
   - overlap 10/15/20%。

4. min_df / max_df A/B
   - min_df 1/2/3/5。
   - 固有語が落ちていないかterm_statsで確認。

5. tokenizer A/B
   - regex vs fugashi vs sudachi_b/c。
   - unsegmented warning, avg_token_len, unique_token_count, OOV的missing_termsを比較。

6. SVD次元A/B
   - 50/100/150/200/300。
   - explained varianceとrank安定性を確認。

7. stopword追加
   - 最後に回す。
   - critical query regression がゼロのものだけ採用。

## 実測A/Bで採用判定すべき指標

必須:

- recall@1
- recall@5
- recall@20
- MRR
- critical_query_regression
- query別 first_hit_rank delta

補助:

- top1 score / margin
- jaccard missing_terms
- failed_queries count
- unique_token_count
- empty_token_rows
- avg_token_count
- SVD explained variance

採用条件案:

- recall@1 が改善、または recall@5/20 が改善し recall@1 が悪化しない。
- MRRが悪化しない。
- critical_query_regression がゼロ。
- failed_queries が増えない。

## 実測A/Bに回すべき候補セット

Set A: 低リスク語彙補強

- current chunk profile
- current tokenizer
- keep words追加: protected_terms + missing technical terms
- synonyms/replace候補のみ

Set B: chunk粒度

- auto
- small
- medium
- large
- custom 600/90

Set C: min_df

- min_df 1
- min_df 2
- min_df 3
- min_df 5

Set D: tokenizer

- regex
- fugashi
- sudachi_b
- sudachi_c

Set E: SVD次元

- 50
- 100
- 150
- 200
- 300

Set F: stopword

- high doc frequency terms only
- keep/protected除外
- critical regressionゼロなら採用

## 商用プロダクト基準での現状評価

更新後評価: 78 / 100

前回分析時点: 72 / 100

良い点:

- chunk lineage が明確。
- config更新が次サイクルに反映される。
- candidate A/B validation の入口がある。
- tokenizer/chunk/TF-IDF/SVDにA/B可能なCLIや構造が一部ある。
- `recall_at_20` が `metrics.json` と validation delta に追加された。
- minimal tuning profile として Set A〜E を候補化し、tuning lineage をJSON/CSVで追跡できる。

弱い点:

- main経路の既定は `regex` だが、任意の `--tokenizer` と tuning候補で `fugashi` / `sudachi_b` / `sudachi_c` を試せる構造になった。
- make_ce内部のTF-IDF/SVDパラメータは現repoから制御できない。
- query前処理とdoc前処理の対称性が保証されていない。
- stopword/keep/synonym候補の採用がまだ実測自動最適化まで到達していない。
- SVD explained variance / rank安定性レポートが無い。

## 2026-07-10 実装更新

今回の実装で、既存の TF-IDF -> TruncatedSVD / make_ce 経路を維持したまま、自動A/B tuning の最小入口を追加した。

### recall@20

- `evaluate_lsa_retrieval.py` の `metrics.json` に `recall_at_20` を追加。
- target列が無い public query 診断時も、後方互換のため `recall_at_20: null` を出す。
- `query_results.csv` の `rank` / `is_hit` は従来どおり出力され、`rank <= 20` の first hit が `recall_at_20` 判定に使われる。
- `auto_improve_lsa_lists.py` の測定経路では `top_k` を最低20に引き上げ、`recall_at_20` を評価可能にした。

### validation 判定

- `validate_lsa_candidate_actions.py` の metric delta に `recall_at_20` を追加。
- `missing_metrics` に `recall_at_1`, `recall_at_5`, `recall_at_20`, `mrr` の不足を記録。
- 古いmetricsに `recall_at_20` が無くても、既存指標があれば measured 判定は後方互換で継続する。
- measured 以外では approve しない仕様は維持。
- safety gate:
  - `critical_query_regression=true` は reject。
  - `recall_at_1` が閾値を超えて悪化した場合は reject。
  - `recall_at_5` 悪化は reject。
  - `mrr` 悪化は reject。
  - `recall_at_20` だけ改善しても、`recall_at_1` / `recall_at_5` / `mrr` が悪化する候補は approve しない。

### tuning candidate A/B 構造

`auto_improve_lsa_lists.py` に任意CLIを追加。

- `--enable-param-tuning`
- `--tuning-profile minimal`
- `--max-tuning-candidates`
- `--tokenizer {regex,fugashi,sudachi_a,sudachi_b,sudachi_c}`

`minimal` profile は以下の Set A〜E。

- Set A: `regex + medium + min_df=2 + svd_dim=150 + max_df=0.95`
- Set B: `regex + small + min_df=2 + svd_dim=150 + max_df=0.95`
- Set C: `sudachi_b + medium + min_df=2 + svd_dim=150 + max_df=0.95`
- Set D: `regex + medium + min_df=1 + svd_dim=200 + max_df=0.95`
- Set E: `regex + auto + min_df=2 + svd_dim=200 + max_df=0.95`

出力:

- `<output_dir>/tuning/tuning_validation_result.json`
- `<output_dir>/tuning/tuning_leaderboard.csv`
- `<output_dir>/tuning/<tuning_candidate_id>/candidate_config/`
- `<output_dir>/tuning/<tuning_candidate_id>/tuning_validation_result.json`

各 candidate には以下を記録する。

- `tuning_candidate_id`
- `params`
- `active_params`
- `inactive_params`
- `baseline_config_dir`
- `candidate_config_dir`
- `baseline_metrics_path`
- `candidate_metrics_path`
- `baseline_results_path`
- `candidate_results_path`
- `vector_text_col`
- `query_path`
- `measurement_status`
- `environment_blockers`

### 実際に効く経路

make_ce backend 経路で今回コード上接続済みのパラメータ:

- `tokenizer`
  - `auto_improve_lsa_lists.py -> lsa_preprocess_and_chunk.py --tokenizer`
  - `regex`, `fugashi`, `sudachi_a/b/c` を前処理に渡せる。
- `chunk_profile`
  - `auto_improve_lsa_lists.py -> lsa_preprocess_and_chunk.py --chunk-profile`
  - `small`, `medium`, `large`, `auto`, `none` が chunk作成に効く。
- `vector_text_col`
  - `evaluate_lsa_retrieval.py --vector-text-col text`
  - make_ce metadata の `text` を評価に使う構造を維持。

make_ce backend 経路で未接続・inactive として記録するパラメータ:

- `min_df`
- `max_df`
- `svd_dim`

理由:

- `build_lsa_vector_space.py` には `--min-df`, `--max-df`, `--svd-dim` が存在し、legacy LSA build では効く。
- しかし現在の主経路は `make_ce_backend.py -> make_ce_v1.py`。
- `make_ce_backend.py` から `make_ce_v1.py` へ渡すCLIに `min_df`, `max_df`, `svd_dim` は存在しない。
- そのため今回の tuning result では、これらを `inactive_params` として記録し、効いていないパラメータによる改善とは扱わない。

### measured 判定できる条件

- baseline/candidate の `metrics.json` が両方存在する。
- 少なくとも既存の target-based metrics が数値として読める。
- `recall_at_20` がある場合は delta と採用判断に含まれる。
- candidate measurement が失敗した場合は `validation_mode=insufficient_evidence` または `measurement_status=insufficient_evidence` とし、approveしない。

### 現PC環境制約で未確認の範囲

- make_ce依存込みの full run は未確認。
- `sudachi_b` / `sudachi_c` は環境に SudachiPy と辞書が無い場合に失敗する。この場合は `environment_blockers` に記録し、measured扱いしない。
- make_ce内部のTF-IDF/SVDパラメータ制御は、現repoの `make_ce_backend.py` 経由では未接続。

## 次に実装すべき改善

1. make_ce_v1.py 側で `min_df`, `max_df`, `svd_dim` をCLI制御できるか確認し、可能なら `make_ce_backend.py` から渡す。
2. tuning candidateごとに `critical_query_regression` を実計算する。
3. `sudachi_b/c` が無い環境では自動skipし、候補失敗をノイズにしない。
4. `use_noun_compounds` と protected term phrase injection を tuning候補に追加する。
5. SVD explained variance と query別 first_hit_rank delta を tuning summary に追加する。

## 次に実装すべきPrompt案

```
目的:
/home/rai/move の既存LSA検索構造を維持したまま、recall@20を評価指標に追加し、tokenizer/chunk/min_df/svd_dimのA/B候補を実測validationへ接続してください。

制約:
- モデル置き換え禁止。TF-IDF -> TruncatedSVDのまま。
- ファイル名・フォルダ構成変更禁止。
- CSV本体の破壊的変更禁止。
- 既存chunk lineage、vector-text-col=text、candidate A/B validation構造を壊さない。

実装:
1. evaluate_lsa_retrieval.py に recall_at_20 を追加。
2. build/evaluate report に query別 first_hit_rank と recall@1/5/20 を保存。
3. auto_improve_lsa_lists.py に dry-run A/B候補生成だけを追加。
   - chunk_profile: small/medium/large/auto
   - tokenizer: regex/fugashi/sudachi_b
   - min_df: 1/2/3/5
   - svd_dim: 50/100/150/200
4. full run不可の場合は environment_blockers に記録し、measured扱いしない。
5. lsa_flow_audit.md と新規/既存チューニングレポートを更新。

確認:
- py_compile
- --help
- 可能なら /tmp の小規模LSA CSVで recall@20 が出ることを確認。
```

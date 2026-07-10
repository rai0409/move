import argparse
import logging
import re
from pathlib import Path
import pandas as pd

# ===== 設定 =====
BASE_DIR_DEFAULT = r"C:\project\document_viewer\_data_assets"
FILE_NAME = r"df0.csv"
WAKATI_COL = "wakati_text"

LOG_LEVEL = logging.INFO
LOGGER = logging.getLogger("validator")


def setup_logging() -> None:
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def validate_statistics(df: pd.DataFrame, wakati_col: str) -> bool:
    """現在のCSVの値と、wakati_textから再計算した値にズレがないか検証する"""
    required_cols = ["形態素数", "一文字形態素数", "一文字形態素数割合"]
    for col in required_cols:
        if col not in df.columns:
            LOGGER.error(
                f"❌ 検証不可: CSVに必須カラム '{col}' が存在しません。先に特徴量を算出してください。"
            )
            return False

    LOGGER.info("🔍 データの整合性チェック（ズレの検証）を開始します...")

    # 1. 現実の wakati_text から「真のトークン」を抽出
    series_clean = df[wakati_col].fillna("").astype(str).str.strip()
    token_series = series_clean.apply(
        lambda x: [t for t in re.split(r"[ \t\u3000]+", x) if t]
    )

    # 2. 真の実態を再計算（検証用テンポラリ）
    true_morph_count = token_series.apply(len)
    true_single_count = token_series.apply(
        lambda tokens: sum(1 for t in tokens if len(t) == 1)
    )
    true_ratio = pd.Series(
        [
            float(s) / float(m) if m > 0 else 0.0
            for m, s in zip(true_morph_count, true_single_count)
        ]
    )

    # 3. 既存のCSVデータとの差分（絶対値）を計算
    # 浮動小数点数（割合）の比較のため、微小な誤差（1e-6）を許容する判定を行います
    mismatched_morph = df[df["形態素数"] != true_morph_count]
    mismatched_single = df[df["一文字形態素数"] != true_single_count]
    mismatched_ratio = df[
        (df["一文字形態素数割合"] - true_ratio).abs() > 1e-6
    ]

    # 4. 結果レポートの出力
    has_error = False
    print("\n" + "=" * 60)
    print("📊 データのズレ・不整合 検証レポート")
    print("=" * 60)

    if len(mismatched_morph) > 0:
        LOGGER.error(
            f"❌ 【異常】『形態素数』がズレている行: {len(mismatched_morph)} 件"
        )
        has_error = True
    else:
        LOGGER.info("✅ 『形態素数』は100%一致しています。")

    if len(mismatched_single) > 0:
        LOGGER.error(
            f"❌ 【異常】『一文字形態素数』がズレている行: {len(mismatched_single)} 件"
        )
        has_error = True
    else:
        LOGGER.info("✅ 『一文字形態素数』は100%一致しています。")

    if len(mismatched_ratio) > 0:
        LOGGER.error(
            f"❌ 【異常】『一文字形態素数割合』がズレている行: {len(mismatched_ratio)} 件"
        )
        has_error = True
    else:
        LOGGER.info("✅ 『一文字形態素数割合』は100%一致しています。")

    print("-" * 60)

    # 不整合がある場合は、サンプルの行（最大3件）を視覚的に表示
    if has_error:
        print("💡 【原因】チャンク分割後に特徴量の再計算が行われていません。")
        print("    以下は不整合が起きているデータの先頭サンプルです：\n")

        # ズレている行のインデックスを特定して表示
        error_indices = set(mismatched_morph.index) | set(
            mismatched_single.index
        )
        for idx in list(error_indices)[:3]:
            print(f"--- 行番号(DataFrame Index): {idx} ---")
            print(f" [テキスト抜粋] : {str(df.loc[idx, wakati_col])[:50]}...")
            print(
                f" [CSV内の値]   : 形態素数={df.loc[idx, '形態素数']}, 一文字={df.loc[idx, '一文字形態素数']}, 割合={df.loc[idx, '一文字形態素数割合']:.4f}"
            )
            print(
                f" [真の実態値] : 形態素数={true_morph_count[idx]}, 一文字={true_single_count[idx]}, 割合={true_ratio[idx]:.4f}"
            )
            print()
    else:
        print("🎉 【合格】すべての統計値が現在のチャンクテキストと完全に一致しています！")

    print("=" * 60 + "\n")
    return not has_error


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(
        description="チャンク後データのメタデータ不整合検算スクリプト"
    )
    parser.add_argument(
        "--craw_name", type=str, default=None, help="クロール名（必須）"
    )
    parser.add_argument(
        "--base_dir", type=str, default=BASE_DIR_DEFAULT, help="ベースディレクトリ"
    )
    args = parser.parse_args()

    craw_name = args.craw_name
    while not craw_name:
        craw_name = input(
            "craw_name を入力してください（必須）: "
        ).strip()

    input_dir = Path(args.base_dir) / f"model_{craw_name}" / "step0"
    input_path = input_dir / FILE_NAME

    LOGGER.info(f"🚀 検証用データ（CSV）を読み込みます: {input_path}")

    if not input_path.exists():
        LOGGER.error(f"❌ エラー: 指定された CSV ファイルが存在しません。")
        return

    # CSVの読み込み
    df = pd.read_csv(input_path, encoding="utf-8-sig")

    # 検証実行
    validate_statistics(df, wakati_col=WAKATI_COL)


if __name__ == "__main__":
    main()
import argparse
import logging
import atexit
from pathlib import Path
from datetime import datetime

from make_clean_df import make_clean_df
from make_space_v1_r2 import make_space
from make_vectors_v1_r2 import make_vectors


def now_iso():
    """現在時刻をISO形式で返す"""
    return datetime.now().isoformat()


def setup_logging(log_dir: str, craw_name: str) -> logging.Logger:
    """
    ログ設定を行う

    Args:
        log_dir (str): ログファイルの保存先ディレクトリ
        craw_name (str): クロール名（ログファイル名に使用）

    Returns:
        logging.Logger: 設定済みのロガー
    """
    # ログディレクトリの作成
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # ログファイル名（日時とcraw_nameを含む）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"make_ce_v1_{craw_name}_{timestamp}.log"

    # フォーマッター
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # コンソールハンドラ
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # ファイルハンドラ
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # Root logger の設定（すべてのモジュールのログを統合）
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    # 既存のハンドラをクリア（重複を防ぐ）
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # メインロガーの設定
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    logger.info(f"ログファイル: {log_file}")

    return logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="make_clean_df, make_space, make_vectors を実行するツール（v1_r2版）"
    )
    parser.add_argument("--base_dir", type=str, default=None, help="ベースディレクトリ (例: ../_data_assets)")
    parser.add_argument("--craw_name", type=str, default=None, help="必須: クロール名")
    parser.add_argument(
        "--chunk_on",
        type=str,
        choices=["y", "n", "Y", "N"],
        default=None,
        help="chunk_on を有効にするか [Y/n]",
    )
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=None,
        help="クラスタ数 [100] (データ数が少ない場合は自動調整)",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=None,
        help="チャンク分割サイズ [500]",
    )
    parser.add_argument(
        "--run_type",
        type=str,
        choices=["all", "clean", "space", "vectors"],
        default=None,
        help="実行対象: all (全て実行), clean (データクリーニングのみ), space (ベクトル空間構築のみ), vectors (ドキュメントベクトル生成のみ)",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="log",
        help="ログファイルの保存先ディレクトリ (デフォルト: base_dir/log)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # --- 引数が一つでも指定されていたら yn_args=1 ---
    yn_args = 1 if any([
        args.base_dir,
        args.craw_name,
        args.chunk_on,
        args.chunk_size is not None,
        args.n_clusters is not None,
        args.run_type,
        args.log_dir != "log",  # デフォルトから変更されている場合
    ]) else 0

    # base_dir（ログディレクトリのデフォルトパスを決定するために最初に取得）
    if args.base_dir:
        base_dir = args.base_dir
    elif yn_args:
        base_dir = r"C:\project\document_viewer\_data_assets"
    else:
        base_dir_input = input("base_dir を入力してください（デフォルト: C:\\project\\document_viewer\\_data_assets）: ").strip()
        base_dir = base_dir_input if base_dir_input else r"C:\project\document_viewer\_data_assets"

    # craw_name
    craw_name = args.craw_name
    while not craw_name:
        craw_name = input("craw_name を入力してください（必須）: ").strip()
        if not craw_name:
            print("craw_name は必須です。値を入力してください。")

    # log_dir（デフォルトは base_dir/log）
    default_log_dir = str(Path(base_dir) / "log")
    if args.log_dir and args.log_dir != "log":
        # 明示的に指定された場合
        log_dir = args.log_dir
    elif yn_args and args.log_dir == "log":
        # 引数モードでlog_dirが指定されていない場合
        log_dir = default_log_dir
    elif yn_args:
        # 引数モードでlog_dirが明示的に指定された場合
        log_dir = args.log_dir
    else:
        # 対話モード
        log_dir_input = input(f"ログディレクトリを入力してください（デフォルト: {default_log_dir}）: ").strip()
        log_dir = log_dir_input if log_dir_input else default_log_dir

    # ログ設定
    logger = setup_logging(log_dir, craw_name)
    logger.info("=" * 60)
    logger.info("make_ce_v1.py 実行開始")
    logger.info("=" * 60)

    # セッション開始ファイル
    try:
        ts_start = datetime.now().strftime("%Y%m%d_%H%M%S")
        Path(log_dir, f"make_ce_start_{ts_start}.txt").write_text(now_iso() + "\n", encoding="utf-8")
        logger.info(f"セッション開始ファイルを作成しました: make_ce_start_{ts_start}.txt")
    except Exception as e:
        logger.error(f"セッション開始ファイルの作成に失敗しました: {e}")

    # セッション終了ファイル（atexitで保証）
    end_status = {"msg": "OK"}

    def _write_end_file():
        try:
            ts_end = datetime.now().strftime("%Y%m%d_%H%M%S")
            with open(Path(log_dir) / f"make_ce_end_{ts_end}.txt", "w", encoding="utf-8") as f:
                f.write(now_iso() + "\n")
                # 1行化して安全に出力
                msg = str(end_status.get("msg", "OK")).replace("\n", " ")
                f.write(msg + "\n")
            logger.info(f"セッション終了ファイルを作成しました: make_ce_end_{ts_end}.txt")
        except Exception as e:
            # 終了処理の例外はログだけ残す
            logger.error(f"セッション終了ファイルの作成に失敗しました: {e}")

    atexit.register(_write_end_file)

    # chunk_on
    if args.chunk_on:
        chunk_on = not (args.chunk_on.lower() == "n")
    elif yn_args:
        chunk_on = True   # デフォルト: 有効
    else:
        chunk_on_input = input("chunk_on を有効にしますか？ [Y/n]（デフォルト: Y）: ").strip().lower()
        chunk_on = not (chunk_on_input == "n")

    if chunk_on:
        logger.info("chunk_on は引数で有効指定されていますが、本スクリプトでは強制的に無効化します。")
    chunk_on = False

    #chunk_size
    if chunk_on:
        if args.chunk_size is not None:
            chunk_size = args.chunk_size
        elif yn_args:
            chunk_size = 100000
        else:
            chunk_size_input = input(
                "チャンク分割サイズを入力してください [500]: "
            ).strip()
            try:
                chunk_size = int(chunk_size_input) if chunk_size_input else 500
            except ValueError:
                print("整数を入力してください。デフォルトの 500 を使用します。")
                chunk_size = 100000

    # n_clusters
    if args.n_clusters is not None:
        n_clusters = args.n_clusters
    elif yn_args:
        n_clusters = 100
    else:
        n_clusters_input = input(
            "クラスタ数を決定してください [100]（データ数がクラスタ数より少ない場合はデータ数になります）: "
        ).strip()
        try:
            n_clusters = int(n_clusters_input) if n_clusters_input else 100
        except ValueError:
            print("整数を入力してください。デフォルトの 100 を使用します。")
            n_clusters = 100

    if args.n_clusters is None or args.n_clusters <= 0:
        logger.info("CLUSTERING DISABLED (n_clusters <= 0)")
        n_clusters = 0

    # chunk_size
    if args.chunk_size is not None:
        chunk_size = args.chunk_size
    elif yn_args:
        chunk_size = 100000
    else:
        chunk_size_input = input(
            "チャンク分割サイズを入力してください [500]: "
        ).strip()
        try:
            chunk_size = int(chunk_size_input) if chunk_size_input else 500
        except ValueError:
            print("整数を入力してください。デフォルトの 500 を使用します。")
            chunk_size = 100000

    # run_type
    if args.run_type:
        run_type = args.run_type.lower()
    elif yn_args:
        run_type = "all"   # 引数指定されていれば省略時は all
    else:
        run_type_input = input("実行形式を選んでください [all/clean/space/vectors]（デフォルト: all）: ").strip().lower()
        run_type = run_type_input if run_type_input in ["all", "clean", "space", "vectors"] else "all"

    # 入力ファイルのパスを構築
    file_name = "df0.csv"
    input_dir = Path(base_dir) / f"model_{craw_name}" / "step0"
    input_path = input_dir / file_name

    # 実行処理
    logger.info("-" * 60)
    logger.info(f"入力ファイル: {input_path}")
    logger.info(f"ベースディレクトリ: {base_dir}")
    logger.info(f"クロール名: {craw_name}")
    logger.info(f"チャンク分割: {'有効' if chunk_on else '無効'}")
    logger.info(f"チャンクサイズ: {chunk_size}")
    logger.info(f"クラスタ数: {n_clusters}")
    logger.info(f"実行タイプ: {run_type}")
    logger.info("-" * 60)

    if run_type == "clean":
        logger.info("=== Step0: データクリーニング ===")
        result = make_clean_df(base_dir, craw_name)
        if result:
            logger.info("✅ データクリーニングが完了しました")
        else:
            logger.error("❌ データクリーニングに失敗しました")
            end_status["msg"] = "ERROR: データクリーニング失敗"
    elif run_type == "space":
        logger.info("=== Step1: ベクトル空間の構築 ===")
        result = make_space(base_dir, str(input_path), craw_name, chunk_on, chunk_size=chunk_size)
        if result:
            logger.info("✅ ベクトル空間の構築が完了しました")
        else:
            logger.error("❌ ベクトル空間の構築に失敗しました")
            end_status["msg"] = "ERROR: ベクトル空間の構築失敗"
    elif run_type == "vectors":
        logger.info("=== Step2: ドキュメントベクトルの生成 ===")
        result = make_vectors(base_dir, str(input_path), craw_name, chunk_on, n_clusters=n_clusters, chunk_size=chunk_size)
        if result:
            logger.info("✅ ドキュメントベクトルの生成が完了しました")
        else:
            logger.error("❌ ドキュメントベクトルの生成に失敗しました")
            end_status["msg"] = "ERROR: ドキュメントベクトルの生成失敗"
    else:  # all
        logger.info("=== Step0: データクリーニング ===")
        result_step0 = make_clean_df(base_dir, craw_name)
        if result_step0:
            logger.info("✅ Step0 が完了しました")
            logger.info("")
            logger.info("=== Step1: ベクトル空間の構築 ===")
            result_step1 = make_space(base_dir, str(input_path), craw_name, chunk_on, chunk_size=chunk_size)
            if result_step1:
                logger.info("✅ Step1 が完了しました")
                logger.info("")
                logger.info("=== Step2: ドキュメントベクトルの生成 ===")
                result_step2 = make_vectors(base_dir, str(input_path), craw_name, chunk_on, n_clusters=n_clusters, chunk_size=chunk_size)
                if result_step2:
                    logger.info("✅ すべての処理が正常に完了しました")
                else:
                    logger.error("❌ Step2 の処理が失敗しました")
                    end_status["msg"] = "ERROR: Step2 の処理が失敗しました"
            else:
                logger.error("❌ Step1 の処理が失敗しました")
                end_status["msg"] = "ERROR: Step1 の処理が失敗しました"
        else:
            logger.error("❌ Step0 の処理が失敗しました")
            end_status["msg"] = "ERROR: Step0 の処理が失敗しました"

    logger.info("=" * 60)
    logger.info(f"実行完了: {run_type}")
    logger.info("=" * 60)

import argparse
import csv
from pathlib import Path
from typing import Any

from prompt_adapter.ai_model_runner import (
    CsvConnectionRepository,
    CsvModelRepository,
    LiteLLMClient,
    ModelCatalog,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_QUESTIONS_CSV = (
    REPO_ROOT / "instance/datasets/model_output_tendency_questions.csv"
)
DEFAULT_MODELS_JSON = REPO_ROOT / "instance/models.local.json"
DEFAULT_CONNECTIONS_JSON = REPO_ROOT / "instance/model_connections.local.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "instance/experiments_result/model_output_tendency"

OUTPUT_HEADERS = [
    "quetion_id",
    "回答",
]


def _load_questions(
    questions_csv_path: Path, limit: int | None
) -> list[dict[str, str]]:
    """質問CSVを読み込み、必要に応じて件数を制限して返す。

    Parameters
    ----------
    questions_csv_path : Path
        読み込む質問CSVのパス。
    limit : int | None
        読み込む件数上限。未指定時は全件。

    Returns
    -------
    list[dict[str, str]]
        質問データ行の一覧。

    Raises
    ------
    ValueError
        必須列が不足している場合。
    """
    with questions_csv_path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        field_names = reader.fieldnames
        if field_names is None:
            raise ValueError("質問CSVのヘッダーを読み込めませんでした")
        if "question_id" not in field_names or "質問内容" not in field_names:
            raise ValueError("質問CSVには question_id 列と 質問内容 列が必要です")
        rows = list(reader)

    if limit is not None:
        return rows[:limit]
    return rows


def _resolve_target_model_aliases(
    model_repository: CsvModelRepository,
    requested_model_aliases: str | None,
) -> list[str]:
    """実行対象のモデル別名一覧を解決する。

    Parameters
    ----------
    model_repository : CsvModelRepository
        有効モデル一覧を取得するリポジトリ。
    requested_model_aliases : str | None
        ユーザー指定のモデル別名（カンマ区切り）。

    Returns
    -------
    list[str]
        実行対象のモデル別名一覧。

    Raises
    ------
    ValueError
        指定形式が不正、または無効なモデル別名を含む場合。
    """
    enabled_aliases = [model.model_alias for model in model_repository.list_all()]
    if requested_model_aliases is None:
        return enabled_aliases

    requested_aliases = [
        alias.strip()
        for alias in requested_model_aliases.split(",")
        if alias.strip() != ""
    ]
    if not requested_aliases:
        raise ValueError("--model-aliases には1件以上のモデル別名を指定してください")

    enabled_aliases_set = set(enabled_aliases)
    unknown_aliases = [
        alias for alias in requested_aliases if alias not in enabled_aliases_set
    ]
    if unknown_aliases:
        unknown_aliases_str = ", ".join(unknown_aliases)
        raise ValueError(
            f"指定されたモデル別名が models.json に存在しないか無効です: {unknown_aliases_str}"
        )

    return requested_aliases


def run_model_output_tendency(
    questions_csv_path: Path,
    models_json_path: Path,
    connections_json_path: Path,
    output_dir: Path,
    model_aliases: str | None,
    limit: int | None,
    max_tokens: int | None,
) -> None:
    """モデル出力傾向の質問を各モデルで実行し、モデル別CSVへ保存する。

    Parameters
    ----------
    questions_csv_path : Path
        質問CSVのパス。
    models_json_path : Path
        モデル定義JSONのパス。
    connections_json_path : Path
        接続定義JSONのパス。
    output_dir : Path
        出力先ディレクトリ。
    model_aliases : str | None
        対象モデル別名のカンマ区切り指定。未指定時は有効モデル全件。
    limit : int | None
        実行する質問数の上限。未指定時は全件。
    max_tokens : int | None
        LLM呼び出し時に付与するmax_tokens。未指定時は付与しない。
    """
    questions = _load_questions(questions_csv_path=questions_csv_path, limit=limit)

    connection_repository = CsvConnectionRepository(connections_json_path)
    model_repository = CsvModelRepository(models_json_path)
    model_catalog = ModelCatalog(
        model_repository=model_repository,
        connection_repository=connection_repository,
    )
    llm_client = LiteLLMClient()

    target_model_aliases = _resolve_target_model_aliases(
        model_repository=model_repository,
        requested_model_aliases=model_aliases,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    for model_alias in target_model_aliases:
        output_path = output_dir / f"model_output_tendency_{model_alias}.csv"
        with output_path.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=OUTPUT_HEADERS)
            writer.writeheader()

            for question in questions:
                question_text = question["質問内容"]

                try:
                    request_kwargs: dict[str, Any] = {}
                    if max_tokens is not None:
                        request_kwargs["max_tokens"] = max_tokens

                    response = llm_client.run_by_alias(
                        model_catalog=model_catalog,
                        model_alias=model_alias,
                        messages=[{"role": "user", "content": question_text}],
                        **request_kwargs,
                    )
                    answer_text = llm_client.extract_response_text(response)
                except Exception as exception:  # noqa: BLE001
                    answer_text = f"ERROR: {type(exception).__name__}: {exception}"

                writer.writerow(
                    {
                        "quetion_id": question.get("question_id", ""),
                        "回答": answer_text,
                    }
                )

        print(f"保存完了: {output_path}")


def _build_argument_parser() -> argparse.ArgumentParser:
    """CLI引数パーサーを構築して返す。

    Returns
    -------
    argparse.ArgumentParser
        本スクリプト用の引数パーサー。
    """
    parser = argparse.ArgumentParser(
        description=(
            "model_output_tendency_questions.csv を対象に、"
            "モデル別の出力結果CSVを作成します"
        )
    )
    parser.add_argument(
        "--questions-csv",
        type=Path,
        default=DEFAULT_QUESTIONS_CSV,
        help="質問CSVのパス",
    )
    parser.add_argument(
        "--models-json",
        type=Path,
        default=DEFAULT_MODELS_JSON,
        help="モデル定義JSONのパス",
    )
    parser.add_argument(
        "--connections-json",
        type=Path,
        default=DEFAULT_CONNECTIONS_JSON,
        help="接続定義JSONのパス",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="結果CSVの出力先ディレクトリ",
    )
    parser.add_argument(
        "--model-aliases",
        type=str,
        default=None,
        help="対象モデル別名をカンマ区切りで指定 (例: model_a,model_b)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="実行する質問数の上限",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="LLM呼び出しで付与するmax_tokens",
    )
    return parser


def main() -> None:
    """CLI引数を解釈してモデル出力傾向の実行処理を開始する。"""
    parser = _build_argument_parser()
    args = parser.parse_args()

    run_model_output_tendency(
        questions_csv_path=args.questions_csv,
        models_json_path=args.models_json,
        connections_json_path=args.connections_json,
        output_dir=args.output_dir,
        model_aliases=args.model_aliases,
        limit=args.limit,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()

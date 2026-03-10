from pathlib import Path

import pytest

from prompt_adapter.ai_model_runner import (
    AIModelProvider,
    ConnectionNotFoundError,
    CsvConnectionRecord,
    CsvConnectionRepository,
    CsvModelRecord,
    CsvModelRepository,
)

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "instance"


def test_CSVリポジトリからサンプル定義を読み込める() -> None:
    # Given: 接続定義CSVとモデル定義CSVを読み込むリポジトリを用意する
    connection_repository = CsvConnectionRepository(
        FIXTURE_DIR / "model_connections.example.csv"
    )
    model_repository = CsvModelRepository(FIXTURE_DIR / "models.example.csv")

    # When: OpenAI用の接続定義とサンプルモデル定義を取得する
    connection = connection_repository.get("openai_main")
    model_record = model_repository.get("gpt5mini_openai")

    # Then: CSVから期待した件数と内容で定義を取得できる
    assert len(connection_repository.list_all()) == 3
    assert len(model_repository.list_all()) == 6
    assert isinstance(connection, CsvConnectionRecord)
    assert isinstance(model_record, CsvModelRecord)
    assert connection_repository.get_default(
        AIModelProvider.OPENAI
    ).connection_name == ("openai_main")
    assert connection.api_key.get_secret_value() == "sk-your-openai-api-key"
    assert model_record.model_name == "gpt-5-mini"


def test_存在しない接続名を指定すると例外になる() -> None:
    # Given: 接続定義CSVを読み込むリポジトリを用意する
    connection_repository = CsvConnectionRepository(
        FIXTURE_DIR / "model_connections.example.csv"
    )

    # Then: 存在しない接続名の取得では例外になる
    with pytest.raises(ConnectionNotFoundError, match="見つかりません"):
        # When: 未定義の接続名を指定して取得する
        connection_repository.get("unknown_connection")

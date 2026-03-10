import os
from pathlib import Path

import pytest
from pydantic import SecretStr

from prompt_adapter.ai_model_runner import (
    AIModelConfig,
    AIModelConnection,
    AIModelProvider,
    CsvConnectionRepository,
    CsvModelRepository,
    LiteLLMRunner,
    ModelCatalog,
)

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "instance"


def test_モデル別名から設定を解決して実行できる() -> None:
    # Given: completion呼び出しをモックしたRunnerとモデルカタログを用意する
    captured_kwargs: dict[str, object] = {}

    def fake_completion(**kwargs: object) -> dict[str, object]:
        captured_kwargs.update(kwargs)
        return {"ok": True}

    model_catalog = ModelCatalog(
        model_repository=CsvModelRepository(FIXTURE_DIR / "models.example.csv"),
        connection_repository=CsvConnectionRepository(
            FIXTURE_DIR / "model_connections.example.csv"
        ),
    )
    runner = LiteLLMRunner(completion_client=fake_completion)

    # When: モデル別名を使ってRunner経由で実行する
    response = runner.run_by_alias(
        model_catalog=model_catalog,
        model_alias="gpt5mini_openai",
        messages=[{"role": "user", "content": "hello"}],
    )

    # Then: モックされたcompletionに期待した引数が渡されレスポンスも返る
    assert response == {"ok": True}
    assert captured_kwargs["model"] == "gpt-5-mini"
    assert captured_kwargs["api_key"] == "sk-your-openai-api-key"


def test_接続定義を実行時接続へ解決するとSecretStrでAPIキーを保持する() -> None:
    # Given: CSVの接続定義を読み込むリポジトリを用意する
    connection_repository = CsvConnectionRepository(
        FIXTURE_DIR / "model_connections.example.csv"
    )

    # When: 接続定義をドメインモデルへ変換する
    connection = connection_repository.get("openai_main").to_domain()

    # Then: APIキーはSecretStrとして保持され秘匿表示される
    assert connection.api_key.get_secret_value() == "sk-your-openai-api-key"
    assert "sk-your-openai-api-key" not in str(connection.api_key)


@pytest.mark.skipif(
    os.getenv("CI") == "true" or not os.getenv("OPENAI_API_KEY"),
    reason="CIでは実リクエストを行わず、ローカルでOPENAI_API_KEYがある場合のみ実行するため",
)
def test_OpenAIへ実際にリクエストできる() -> None:
    # Given: 実際のOpenAI APIキーを使うRunnerと接続設定を用意する
    runner = LiteLLMRunner()
    connection = AIModelConnection(
        connection_name="openai_integration",
        provider_type=AIModelProvider.OPENAI,
        api_key=SecretStr(os.environ["OPENAI_API_KEY"]),
        is_default=False,
        enabled=True,
    )
    model_config = AIModelConfig(
        model_name="gpt-5-mini",
        provider_type=AIModelProvider.OPENAI,
    )

    # When: OpenAIへ最小限の実リクエストを送る
    response = runner.run(
        model_config=model_config,
        connection=connection,
        messages=[{"role": "user", "content": "Reply with OK only."}],
        max_tokens=10,
    )

    # Then: レスポンスが返ってきて疎通できる
    assert response is not None

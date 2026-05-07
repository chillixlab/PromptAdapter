import os
from pathlib import Path

import pytest
from pydantic import SecretStr

from prompt_adapter.ai_model_runner import (
    AIModelCallMode,
    AIModelConfig,
    AIModelConnection,
    AIModelProvider,
    CsvConnectionRepository,
    CsvModelRepository,
    LiteLLMClient,
    ModelCatalog,
)

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "instance"


def test_モデル別名から設定を解決して実行できる() -> None:
    # Given: completion呼び出しをモックしたClientとモデルカタログを用意する
    captured_kwargs: dict[str, object] = {}

    def fake_completion(**kwargs: object) -> dict[str, object]:
        captured_kwargs.update(kwargs)
        return {"ok": True}

    model_catalog = ModelCatalog(
        model_repository=CsvModelRepository(FIXTURE_DIR / "models.example.json"),
        connection_repository=CsvConnectionRepository(
            FIXTURE_DIR / "model_connections.example.json"
        ),
    )
    llm_client = LiteLLMClient(completion_client=fake_completion)

    # When: モデル別名を使ってClient経由で実行する
    response = llm_client.run_by_alias(
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
        FIXTURE_DIR / "model_connections.example.json"
    )

    # When: 接続定義をドメインモデルへ変換する
    connection = connection_repository.get("openai_main").to_domain()

    # Then: APIキーはSecretStrとして保持され秘匿表示される
    assert connection.api_key.get_secret_value() == "sk-your-openai-api-key"
    assert "sk-your-openai-api-key" not in str(connection.api_key)


def test_未対応のLiteLLMモードを指定すると例外になる() -> None:
    # Given: 未対応モードを持つモデル設定を生成する
    llm_client = LiteLLMClient(completion_client=lambda **_: {"ok": True})
    connection = AIModelConnection(
        connection_name="openai_integration",
        provider_type=AIModelProvider.OPENAI,
        api_key=SecretStr("sk-dummy"),
        is_default=False,
        enabled=True,
    )
    model_config = AIModelConfig(
        model_name="gpt-5-mini",
        provider_type=AIModelProvider.OPENAI,
        litellm_mode=AIModelCallMode.COMPLETION,
    )

    # When/Then: 実行時に未対応モードへ書き換えると例外になる
    model_config.__dict__["litellm_mode"] = "unknown_mode"
    with pytest.raises(ValueError, match="未対応のLiteLLMモード"):
        llm_client.run(
            model_config=model_config,
            connection=connection,
            messages=[{"role": "user", "content": "hello"}],
        )


def test_Responsesモードでresponsesクライアントを呼び出せる() -> None:
    # Given: responsesモードのモデル設定とモックしたresponsesクライアントを用意する
    captured_kwargs: dict[str, object] = {}

    def fake_responses(**kwargs: object) -> dict[str, object]:
        captured_kwargs.update(kwargs)
        return {"ok": True, "mode": "responses"}

    llm_client = LiteLLMClient(
        completion_client=lambda **_: {"ok": False},
        responses_client=fake_responses,
    )
    connection = AIModelConnection(
        connection_name="openai_integration",
        provider_type=AIModelProvider.OPENAI,
        api_key=SecretStr("sk-dummy"),
        is_default=False,
        enabled=True,
    )
    model_config = AIModelConfig(
        model_name="gpt-4.1",
        provider_type=AIModelProvider.OPENAI,
        litellm_mode=AIModelCallMode.RESPONSES,
    )

    # When: responsesモードで実行する
    response = llm_client.run(
        model_config=model_config,
        connection=connection,
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=32,
    )

    # Then: responsesクライアントが呼ばれinput/max_output_tokensが渡る
    assert response == {"ok": True, "mode": "responses"}
    assert captured_kwargs["model"] == "gpt-4.1"
    assert captured_kwargs["input"] == [{"role": "user", "content": "hello"}]
    assert captured_kwargs["max_output_tokens"] == 32


@pytest.mark.skipif(
    os.getenv("CI") == "true" or not os.getenv("OPENAI_API_KEY"),
    reason="CIでは実リクエストを行わず、ローカルでOPENAI_API_KEYがある場合のみ実行するため",
)
def test_OpenAIへ実際にリクエストできる() -> None:
    # Given: 実際のOpenAI APIキーを使うClientと接続設定を用意する
    llm_client = LiteLLMClient()
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
    response = llm_client.run(
        model_config=model_config,
        connection=connection,
        messages=[{"role": "user", "content": "Reply with OK only."}],
        max_tokens=10,
    )

    # Then: レスポンスが返ってきて疎通できる
    assert response is not None


def test_レスポンスから回答本文を抽出できる() -> None:
    # Given: output_text属性を持つレスポンスとClientを用意する
    llm_client = LiteLLMClient(completion_client=lambda **_: {"ok": True})
    response = type("Response", (), {"output_text": "抽出対象の本文"})()

    # When: レスポンス本文を抽出する
    extracted_text = llm_client.extract_response_text(response)

    # Then: 本文をそのまま取得できる
    assert extracted_text == "抽出対象の本文"

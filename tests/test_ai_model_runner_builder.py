from pathlib import Path

from prompt_adapter.ai_model_runner import (
    AIModelConfig,
    AIModelProvider,
    CsvConnectionRepository,
    LiteLLMRequestBuilder,
)

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "instance"


def test_OpenAI向けのリクエスト引数を組み立てられる() -> None:
    # Given: OpenAI用の接続設定とモデル設定、リクエストビルダーを用意する
    connection_repository = CsvConnectionRepository(
        FIXTURE_DIR / "model_connections.example.csv"
    )
    connection = connection_repository.get("openai_main").to_domain()
    builder = LiteLLMRequestBuilder()
    model_config = AIModelConfig(
        model_name="gpt-5-mini",
        provider_type=AIModelProvider.OPENAI,
    )

    # When: OpenAI向けのLiteLLMリクエスト引数を組み立てる
    request = builder.build(
        model_config=model_config,
        connection=connection,
        messages=[{"role": "user", "content": "hello"}],
    )

    # Then: OpenAI向けに期待したmodel名と認証情報が入る
    assert request["model"] == "gpt-5-mini"
    assert request["api_key"] == "sk-your-openai-api-key"
    assert connection.api_key.get_secret_value() == "sk-your-openai-api-key"
    assert "api_version" not in request


def test_AzureOpenAI向けのリクエスト引数を組み立てられる() -> None:
    # Given: Azure OpenAI用の接続設定とモデル設定、リクエストビルダーを用意する
    connection_repository = CsvConnectionRepository(
        FIXTURE_DIR / "model_connections.example.csv"
    )
    connection = connection_repository.get("aoai_main").to_domain()
    builder = LiteLLMRequestBuilder()
    model_config = AIModelConfig(
        model_name="gpt-4o-deploy",
        provider_type=AIModelProvider.AZURE_OPENAI,
    )

    # When: Azure OpenAI向けのLiteLLMリクエスト引数を組み立てる
    request = builder.build(
        model_config=model_config,
        connection=connection,
        messages=[{"role": "user", "content": "hello"}],
    )

    # Then: Azure OpenAI向けのmodel名、api_base、api_versionが入る
    assert request["model"] == "azure/gpt-4o-deploy"
    assert request["api_base"] == "https://your-resource.openai.azure.com/"
    assert request["api_version"] == "2024-08-01-preview"


def test_Foundry向けのリクエスト引数を組み立てられる() -> None:
    # Given: Foundry用の接続設定とモデル設定、リクエストビルダーを用意する
    connection_repository = CsvConnectionRepository(
        FIXTURE_DIR / "model_connections.example.csv"
    )
    connection = connection_repository.get("foundry_main").to_domain()
    builder = LiteLLMRequestBuilder()
    model_config = AIModelConfig(
        model_name="command-r-plus",
        provider_type=AIModelProvider.MICROSOFT_FOUNDRY,
    )

    # When: Foundry向けのLiteLLMリクエスト引数を組み立てる
    request = builder.build(
        model_config=model_config,
        connection=connection,
        messages=[{"role": "user", "content": "hello"}],
    )

    # Then: Foundry向けのmodel名とapi_baseが入りapi_versionは不要のままになる
    assert request["model"] == "azure_ai/command-r-plus"
    assert request["api_base"] == "https://your-project.inference.ai.azure.com/"
    assert "api_version" not in request

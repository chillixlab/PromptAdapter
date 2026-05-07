from pathlib import Path

from prompt_adapter.ai_model_runner import (
    AIModelCallMode,
    CsvConnectionRepository,
    CsvModelRepository,
    ModelCatalog,
)

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "instance"


def test_モデル別名から実行用モデル設定と接続設定を解決できる() -> None:
    # Given: モデル別名から設定を引けるカタログを用意する
    model_catalog = ModelCatalog(
        model_repository=CsvModelRepository(FIXTURE_DIR / "models.example.json"),
        connection_repository=CsvConnectionRepository(
            FIXTURE_DIR / "model_connections.example.json"
        ),
    )

    # When: モデル別名を使って実行用のモデル設定と接続設定を解決する
    model_config, connection = model_catalog.resolve("gpt5mini_openai")

    # Then: モデル別名に対応するドメインモデルへ変換された設定が取得できる
    assert model_config.model_name == "gpt-5-mini"
    assert model_config.litellm_mode == AIModelCallMode.COMPLETION
    assert connection.connection_name == "openai_main"
    assert connection.api_key.get_secret_value() == "sk-your-openai-api-key"

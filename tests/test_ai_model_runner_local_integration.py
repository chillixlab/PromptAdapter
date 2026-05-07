import os
from pathlib import Path

import pytest

from prompt_adapter.ai_model_runner import (
    CsvConnectionRepository,
    CsvModelRepository,
    LiteLLMClient,
    ModelCatalog,
)

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "instance"
LOCAL_CONNECTIONS_JSON = FIXTURE_DIR / "model_connections.local.json"
LOCAL_MODELS_JSON = FIXTURE_DIR / "models.local.json"


@pytest.mark.skipif(
    os.getenv("CI") == "true",
    reason="CIでは実リクエストを行わず、ローカルでも明示指定時のみ実行するため",
)
def test_ローカル接続先JSONを使ってLLMへ実リクエストできる() -> None:
    # GIVEN: ローカル専用の接続定義JSONとモデル定義JSONが存在する
    if not LOCAL_CONNECTIONS_JSON.exists() or not LOCAL_MODELS_JSON.exists():
        pytest.skip(
            "ローカル接続先JSONが見つからないためスキップします: "
            "instance/model_connections.local.json, instance/models.local.json"
        )

    connection_repository = CsvConnectionRepository(LOCAL_CONNECTIONS_JSON)
    model_repository = CsvModelRepository(LOCAL_MODELS_JSON)
    model_catalog = ModelCatalog(
        model_repository=model_repository,
        connection_repository=connection_repository,
    )
    llm_client = LiteLLMClient()

    local_model_alias = os.getenv("LOCAL_MODEL_ALIAS")
    if local_model_alias:
        model_alias = local_model_alias
    else:
        models = model_repository.list_all()
        if not models:
            pytest.skip("有効なモデル定義が存在しないためスキップします")
        model_alias = models[0].model_alias

    # WHEN: モデル別名から設定を解決して実リクエストを送信する
    response = llm_client.run_by_alias(
        model_catalog=model_catalog,
        model_alias=model_alias,
        messages=[
            {
                "role": "user",
                "content": "接続確認です。No, I am your father. とだけ返してください。",
            }
        ],
        max_tokens=16,
    )

    # THEN: 実リクエストのレスポンス本文が期待どおり返る
    response_text = llm_client.extract_response_text(response)
    assert response_text.strip() != ""
    assert "No, I am your father." in response_text

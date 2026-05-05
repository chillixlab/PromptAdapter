from typing import Any

from prompt_adapter.ai_model_runner.domain import (
    AIModelCallMode,
    AIModelConfig,
    AIModelConnection,
    AIModelProvider,
)


class LiteLLMRequestBuilder:
    """モデル設定と接続設定からLiteLLM呼び出し引数を組み立てる。"""

    def build(
        self,
        model_config: AIModelConfig,
        connection: AIModelConnection,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """provider差分を吸収してLiteLLM呼び出し用kwargsを返す。

        Parameters
        ----------
        model_config : AIModelConfig
            実行対象のモデル設定。
        connection : AIModelConnection
            接続先の認証・endpoint設定。
        messages : list[dict[str, Any]]
            LiteLLMへ渡すメッセージ配列。
        **kwargs : Any
            呼び出し時に追加で上書きしたいLiteLLMパラメータ。

        Returns
        -------
        dict[str, Any]
            `litellm.completion()` に渡せる引数辞書。

        Raises
        ------
        ValueError
            providerの不一致、またはAzure OpenAIの`api_version`不足時。
        """
        if model_config.provider_type != connection.provider_type:
            raise ValueError(
                "モデル設定のprovider_typeと接続設定のprovider_typeが一致していません"
            )

        request_kwargs: dict[str, Any] = {
            "model": model_config.litellm_model_name,
            "messages": messages,
            "api_key": connection.api_key.get_secret_value(),
        }

        api_version = model_config.api_version or connection.api_version

        match connection.provider_type:
            case AIModelProvider.OPENAI:
                if connection.organization:
                    request_kwargs["organization"] = connection.organization
                if connection.api_base:
                    request_kwargs["api_base"] = connection.api_base
            case AIModelProvider.AZURE_OPENAI:
                request_kwargs["api_base"] = connection.api_base
                if not api_version:
                    raise ValueError("Azure OpenAIではapi_versionの指定が必須です")
                request_kwargs["api_version"] = api_version
            case AIModelProvider.MICROSOFT_FOUNDRY:
                request_kwargs["api_base"] = connection.api_base
                if api_version:
                    request_kwargs["api_version"] = api_version

        request_kwargs.update(model_config.litellm_model_params)
        request_kwargs.update(kwargs)

        if model_config.litellm_mode == AIModelCallMode.RESPONSES:
            messages_input = request_kwargs.pop("messages")
            if "input" not in request_kwargs:
                request_kwargs["input"] = messages_input
            if (
                "max_output_tokens" not in request_kwargs
                and "max_tokens" in request_kwargs
            ):
                request_kwargs["max_output_tokens"] = request_kwargs.pop("max_tokens")

        return request_kwargs

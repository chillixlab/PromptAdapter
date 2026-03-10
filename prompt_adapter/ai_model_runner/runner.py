from collections.abc import Callable
from typing import Any

import litellm

from prompt_adapter.ai_model_runner.builder import LiteLLMRequestBuilder
from prompt_adapter.ai_model_runner.catalog import ModelCatalog
from prompt_adapter.ai_model_runner.domain import AIModelConfig, AIModelConnection

CompletionClient = Callable[..., Any]


class LiteLLMRunner:
    """組み立て済み設定を使ってLiteLLMへ実リクエストを送る実行器。

    Notes
    -----
    `litellm.completion()` を直接呼んでもよいが、providerごとのパラメータ差分や
    テスト時の差し替えを1か所へ寄せるために薄いラッパーとして置いている。
    """

    def __init__(
        self,
        request_builder: LiteLLMRequestBuilder | None = None,
        completion_client: CompletionClient | None = None,
    ):
        """リクエストビルダーとcompletion関数を注入して初期化する。

        Parameters
        ----------
        request_builder : LiteLLMRequestBuilder | None, optional
            LiteLLM用引数を構築するビルダー。未指定時は標準ビルダーを使う。
        completion_client : CompletionClient | None, optional
            実際に呼び出すcompletion関数。未指定時は`litellm.completion`を使う。
        """
        self.request_builder = request_builder or LiteLLMRequestBuilder()
        self.completion_client: CompletionClient = (
            completion_client or litellm.completion
        )

    def run(
        self,
        model_config: AIModelConfig,
        connection: AIModelConnection,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> Any:
        """モデル設定と接続設定を使ってLiteLLMへリクエストする。

        Parameters
        ----------
        model_config : AIModelConfig
            実行対象のモデル設定。
        connection : AIModelConnection
            使用する接続先設定。
        messages : list[dict[str, Any]]
            LiteLLMへ渡すメッセージ配列。
        **kwargs : Any
            実行時に追加で渡すLiteLLMパラメータ。

        Returns
        -------
        Any
            completion関数が返すレスポンス。
        """
        request_kwargs = self.request_builder.build(
            model_config=model_config,
            connection=connection,
            messages=messages,
            **kwargs,
        )
        return self.completion_client(**request_kwargs)

    def run_by_alias(
        self,
        model_catalog: ModelCatalog,
        model_alias: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> Any:
        """モデル別名から設定を解決してLiteLLMへリクエストする。

        Parameters
        ----------
        model_catalog : ModelCatalog
            モデル別名から設定を解決するカタログ。
        model_alias : str
            実行したいモデルの別名。
        messages : list[dict[str, Any]]
            LiteLLMへ渡すメッセージ配列。
        **kwargs : Any
            実行時に追加で渡すLiteLLMパラメータ。

        Returns
        -------
        Any
            completion関数が返すレスポンス。

        Notes
        -----
        `model_alias` をキーにして `ModelCatalog` からモデル定義と接続定義を取得し、
        それを `run()` に渡して実行するためのショートカットメソッド。
        """
        model_config, connection = model_catalog.resolve(model_alias)
        return self.run(
            model_config=model_config,
            connection=connection,
            messages=messages,
            **kwargs,
        )

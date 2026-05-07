from collections.abc import Callable
from typing import Any, cast

import litellm
from litellm.types.utils import ModelResponse, ResponsesAPIResponse

from prompt_adapter.ai_model_runner.builder import LiteLLMRequestBuilder
from prompt_adapter.ai_model_runner.catalog import ModelCatalog
from prompt_adapter.ai_model_runner.domain import (
    AIModelCallMode,
    AIModelConfig,
    AIModelConnection,
)
from prompt_adapter.ai_model_runner.response_extractor import (
    extract_response_text as _extract_response_text,
)

CompletionClient = Callable[..., Any]
ResponsesClient = Callable[..., Any]
type LiteLLMResponse = ModelResponse | ResponsesAPIResponse


class LiteLLMClient:
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
        responses_client: ResponsesClient | None = None,
    ):
        """リクエストビルダーとLiteLLM呼び出し関数を注入して初期化する。

        Parameters
        ----------
        request_builder : LiteLLMRequestBuilder | None, optional
            LiteLLM用引数を構築するビルダー。未指定時は標準ビルダーを使う。
        completion_client : CompletionClient | None, optional
            実際に呼び出すcompletion関数。未指定時は`litellm.completion`を使う。
        responses_client : ResponsesClient | None, optional
            実際に呼び出すresponses関数。未指定時は`litellm.responses`を使う。
        """
        self.request_builder = request_builder or LiteLLMRequestBuilder()
        self.completion_client: CompletionClient = (
            completion_client or litellm.completion
        )
        self.responses_client: ResponsesClient = responses_client or litellm.responses

    def run(
        self,
        model_config: AIModelConfig,
        connection: AIModelConnection,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> LiteLLMResponse:
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
        LiteLLMResponse
            指定モードのLiteLLM関数が返すレスポンス。

        Raises
        ------
        ValueError
            未対応のLiteLLMモードが指定された場合。
        """
        request_kwargs = self.request_builder.build(
            model_config=model_config,
            connection=connection,
            messages=messages,
            **kwargs,
        )
        match model_config.litellm_mode:
            case AIModelCallMode.COMPLETION:
                return cast(LiteLLMResponse, self.completion_client(**request_kwargs))
            case AIModelCallMode.RESPONSES:
                return cast(LiteLLMResponse, self.responses_client(**request_kwargs))
            case _:
                raise ValueError("未対応のLiteLLMモードが指定されました")

    def run_by_alias(
        self,
        model_catalog: ModelCatalog,
        model_alias: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> LiteLLMResponse:
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
        LiteLLMResponse
            指定モードのLiteLLM関数が返すレスポンス。

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

    def extract_response_text(self, response: Any) -> str:
        """LiteLLMレスポンスから回答本文を抽出する。

        Parameters
        ----------
        response : Any
            LiteLLMから返却されたレスポンス。

        Returns
        -------
        str
            抽出した回答本文。抽出できない場合は文字列化した値。
        """
        return _extract_response_text(response)

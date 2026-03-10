from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator


class AIModelProvider(StrEnum):
    """AIモデルの提供元種別。

    Notes
    -----
    LiteLLMへ渡す`model`値の組み立て方法はプロバイダーごとに異なる。
    その差分をこの列挙型で表現する。
    """

    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"
    MICROSOFT_FOUNDRY = "microsoft_foundry"

    def to_litellm_model_name(self, model_name: str) -> str:
        """プロバイダーごとのLiteLLM用モデル名に変換する。

        Parameters
        ----------
        model_name : str
            OpenAIのモデル名、Azure OpenAIのdeployment名、
            またはFoundry上のモデル名。

        Returns
        -------
        str
            LiteLLMへ渡す`model`文字列。
        """
        match self:
            case AIModelProvider.OPENAI:
                return model_name
            case AIModelProvider.AZURE_OPENAI:
                return f"azure/{model_name}"
            case AIModelProvider.MICROSOFT_FOUNDRY:
                return f"azure_ai/{model_name}"


class AIModelConfig(BaseModel):
    """実行時に使用するAIモデル設定。

    Notes
    -----
    接続先情報そのものは持たず、どのモデルをどのような生成条件で
    呼び出すかだけを表現する。
    """

    model_name: Annotated[
        str, Field(title="モデル名", description="使用するAIモデルの名前")
    ]
    provider_type: Annotated[
        AIModelProvider,
        Field(title="プロバイダータイプ", description="AIモデルの提供元タイプ"),
    ]
    connection_name: Annotated[
        str | None,
        Field(title="接続名", description="使用する接続設定名"),
    ] = None
    api_version: Annotated[
        str | None,
        Field(title="APIバージョン", description="使用するAPIのバージョン"),
    ] = None
    temperature: Annotated[
        float | None,
        Field(title="temperature", description="生成温度"),
    ] = None
    max_tokens: Annotated[
        int | None,
        Field(title="max_tokens", description="最大トークン数"),
    ] = None

    @property
    def litellm_model_name(self) -> str:
        """LiteLLMに渡す`model`値を返す。

        Returns
        -------
        str
            providerに応じて変換済みのLiteLLM用モデル名。
        """
        return self.provider_type.to_litellm_model_name(self.model_name)

    @property
    def litellm_model_params(self) -> dict[str, str | float | int]:
        """LiteLLMにそのまま渡せるモデル固有パラメータを返す。

        Returns
        -------
        dict[str, str | float | int]
            `api_version` や `temperature` など、モデル設定由来の追加パラメータ。
        """
        params: dict[str, str | float | int] = {}
        if self.api_version:
            params["api_version"] = self.api_version
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        return params


class AIModelConnection(BaseModel):
    """実行時に使用する接続先情報。

    Notes
    -----
    リポジトリ層から解決された後にアプリケーション層で利用するドメインモデル。
    APIキーは`SecretStr`として保持する。
    """

    connection_name: Annotated[str, Field(title="接続名", description="接続先の識別子")]
    provider_type: Annotated[
        AIModelProvider,
        Field(title="プロバイダータイプ", description="接続先のプロバイダータイプ"),
    ]
    api_key: Annotated[
        SecretStr,
        Field(title="APIキー", description="実行時に利用するAPIキー"),
    ]
    api_base: Annotated[
        str | None,
        Field(title="API base", description="接続先のbase URL"),
    ] = None
    api_version: Annotated[
        str | None,
        Field(title="APIバージョン", description="既定のAPIバージョン"),
    ] = None
    organization: Annotated[
        str | None,
        Field(title="Organization", description="OpenAI organization ID"),
    ] = None
    is_default: Annotated[
        bool,
        Field(default=False, title="既定接続", description="既定接続かどうか"),
    ]
    enabled: Annotated[
        bool,
        Field(default=True, title="有効フラグ", description="使用可否"),
    ]
    notes: Annotated[
        str | None,
        Field(title="備考", description="任意のメモ"),
    ] = None

    @field_validator("api_base", "api_version", "organization", "notes", mode="before")
    @classmethod
    def empty_string_to_none(cls, value: str | None) -> str | None:
        """入力由来の空文字を`None`へ正規化する。

        Parameters
        ----------
        value : str | None
            入力から読み込んだ値。

        Returns
        -------
        str | None
            空文字なら`None`、それ以外は入力値を返す。
        """
        if value == "":
            return None
        return value

    @model_validator(mode="after")
    def validate_provider_specific_fields(self) -> "AIModelConnection":
        """プロバイダーごとの必須接続項目を検証する。

        Returns
        -------
        AIModelConnection
            検証済みの自身のインスタンス。

        Raises
        ------
        ValueError
            Azure系プロバイダーで`api_base`が未設定の場合。
        """
        if (
            self.provider_type
            in {AIModelProvider.AZURE_OPENAI, AIModelProvider.MICROSOFT_FOUNDRY}
            and not self.api_base
        ):
            raise ValueError("Azure系プロバイダーではapi_baseの指定が必須です")
        return self

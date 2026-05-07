import json
from pathlib import Path
from typing import Annotated, cast

from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator

from prompt_adapter.ai_model_runner.domain import (
    AIModelCallMode,
    AIModelConfig,
    AIModelConnection,
    AIModelProvider,
)


class ConnectionNotFoundError(LookupError):
    """接続定義が見つからないときに送出する例外。"""


class ModelDefinitionNotFoundError(LookupError):
    """モデル定義が見つからないときに送出する例外。"""


class CsvConnectionRecord(BaseModel):
    """接続先JSONの1件を表すリポジトリ用スキーマ。

    Notes
    -----
    JSONとの入出力に閉じたスキーマであり、アプリケーション本体では
    `AIModelConnection` に変換した結果を扱う。
    """

    connection_name: Annotated[str, Field(title="接続名", description="接続先の識別子")]
    provider_type: Annotated[
        AIModelProvider,
        Field(title="プロバイダータイプ", description="接続先のプロバイダータイプ"),
    ]
    api_key: Annotated[
        SecretStr,
        Field(title="APIキー", description="JSONに保存されたAPIキー"),
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
        """JSON由来の空文字を`None`へ正規化する。

        Parameters
        ----------
        value : str | None
            JSONから読み込んだ値。

        Returns
        -------
        str | None
            空文字なら`None`、それ以外は入力値を返す。
        """
        if value == "":
            return None
        return value

    @model_validator(mode="after")
    def validate_provider_specific_fields(self) -> "CsvConnectionRecord":
        """プロバイダーごとの必須接続項目を検証する。

        Returns
        -------
        CsvConnectionRecord
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

    def to_domain(self) -> AIModelConnection:
        """JSONレコードをドメインモデルへ変換する。

        Returns
        -------
        AIModelConnection
            実行時に利用する接続先情報。
        """
        return AIModelConnection(
            connection_name=self.connection_name,
            provider_type=self.provider_type,
            api_key=self.api_key,
            api_base=self.api_base,
            api_version=self.api_version,
            organization=self.organization,
            is_default=self.is_default,
            enabled=self.enabled,
            notes=self.notes,
        )


class CsvModelRecord(BaseModel):
    """モデル定義JSONの1件を表すリポジトリ用スキーマ。"""

    model_alias: Annotated[
        str, Field(title="モデル別名", description="アプリケーションから参照する論理名")
    ]
    provider_type: Annotated[
        AIModelProvider,
        Field(title="プロバイダータイプ", description="モデルの提供元タイプ"),
    ]
    connection_name: Annotated[
        str, Field(title="接続名", description="参照する接続先名")
    ]
    model_name: Annotated[
        str,
        Field(
            title="モデル名", description="プロバイダー上のモデル名またはdeployment名"
        ),
    ]
    api_version_override: Annotated[
        str | None,
        Field(
            title="APIバージョン上書き",
            description="接続設定より優先するAPIバージョン",
        ),
    ] = None
    litellm_mode: Annotated[
        AIModelCallMode,
        Field(
            default=AIModelCallMode.COMPLETION,
            title="LiteLLMモード",
            description="LiteLLMで使用するモード",
        ),
    ]
    temperature: Annotated[
        float | None,
        Field(title="temperature", description="モデル既定の生成温度"),
    ] = None
    max_tokens: Annotated[
        int | None,
        Field(title="max_tokens", description="モデル既定の最大トークン数"),
    ] = None
    enabled: Annotated[
        bool,
        Field(default=True, title="有効フラグ", description="使用可否"),
    ]
    notes: Annotated[
        str | None,
        Field(title="備考", description="任意のメモ"),
    ] = None

    @field_validator(
        "api_version_override", "temperature", "max_tokens", "notes", mode="before"
    )
    @classmethod
    def empty_optional_values_to_none(cls, value: str | None) -> str | None:
        """JSON上の空文字を任意項目の`None`として扱う。

        Parameters
        ----------
        value : str | None
            JSONから読み込んだ値。

        Returns
        -------
        str | None
            空文字なら`None`、それ以外は入力値を返す。
        """
        if value == "":
            return None
        return value

    def to_domain(self) -> AIModelConfig:
        """JSONレコードを実行用モデル設定へ変換する。

        Returns
        -------
        AIModelConfig
            `LiteLLMClient` がそのまま利用できる実行用設定。
        """
        return AIModelConfig(
            model_name=self.model_name,
            provider_type=self.provider_type,
            connection_name=self.connection_name,
            api_version=self.api_version_override,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            litellm_mode=self.litellm_mode,
        )


def _read_rows(file_path: str | Path) -> list[dict[str, object]]:
    """JSONファイルを辞書リストとして読み込む。

    Parameters
    ----------
    file_path : str | Path
        読み込むJSONファイルのパス。

    Returns
    -------
    list[dict[str, object]]
        1行を1辞書として表現した行データ一覧。

    Raises
    ------
    ValueError
        JSON構造が不正、または拡張子が`.json`でない場合。
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".json":
        with path.open(encoding="utf-8") as json_file:
            rows = json.load(json_file)
        if not isinstance(rows, list):
            raise ValueError("JSONファイルは配列形式である必要があります")
        validated_rows: list[dict[str, object]] = []
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ValueError(
                    f"JSONファイルの{index + 1}件目がオブジェクト形式ではありません"
                )
            validated_rows.append(cast(dict[str, object], row))
        return validated_rows

    raise ValueError("接続・モデル定義ファイルは .json のみ対応しています")


def _is_enabled(row: dict[str, object]) -> bool:
    """enabled値を真偽値として解釈する。"""
    enabled_value = row.get("enabled", True)
    if isinstance(enabled_value, bool):
        return enabled_value
    if enabled_value is None:
        return True
    return str(enabled_value).strip().lower() != "false"


class CsvConnectionRepository:
    """接続先JSONから有効な接続定義を読み込むリポジトリ。"""

    def __init__(self, file_path: str | Path):
        """接続先JSONを読み込み、利用可能な接続一覧を保持する。

        Parameters
        ----------
        file_path : str | Path
            接続定義JSONのパス。
        """
        self.file_path = Path(file_path)
        self._connections = [
            CsvConnectionRecord.model_validate(row)
            for row in _read_rows(self.file_path)
            if _is_enabled(row)
        ]

    def list_all(self) -> list[CsvConnectionRecord]:
        """有効な接続定義をすべて返す。

        Returns
        -------
        list[CsvConnectionRecord]
            有効フラグが立っている接続定義一覧。
        """
        return list(self._connections)

    def get(self, connection_name: str) -> CsvConnectionRecord:
        """接続名から接続定義を1件取得する。

        Parameters
        ----------
        connection_name : str
            取得対象の接続名。

        Returns
        -------
        CsvConnectionRecord
            条件に一致した接続定義。

        Raises
        ------
        ConnectionNotFoundError
            一致する接続名が存在しない場合。
        """
        for connection in self._connections:
            if connection.connection_name == connection_name:
                return connection
        raise ConnectionNotFoundError(
            f"接続名 '{connection_name}' に対応する接続定義が見つかりません"
        )

    def get_default(self, provider_type: AIModelProvider) -> CsvConnectionRecord:
        """プロバイダーごとの既定接続を取得する。

        Parameters
        ----------
        provider_type : AIModelProvider
            対象のプロバイダー種別。

        Returns
        -------
        CsvConnectionRecord
            指定プロバイダーの既定接続。

        Raises
        ------
        ConnectionNotFoundError
            既定接続が見つからない場合。
        """
        for connection in self._connections:
            if connection.provider_type == provider_type and connection.is_default:
                return connection
        raise ConnectionNotFoundError(
            f"プロバイダー '{provider_type}' の既定接続が見つかりません"
        )


class CsvModelRepository:
    """モデル定義JSONから有効なモデル一覧を読み込むリポジトリ。"""

    def __init__(self, file_path: str | Path):
        """モデル定義JSONを読み込み、利用可能なモデル一覧を保持する。

        Parameters
        ----------
        file_path : str | Path
            モデル定義JSONのパス。
        """
        self.file_path = Path(file_path)
        self._models = [
            CsvModelRecord.model_validate(row)
            for row in _read_rows(self.file_path)
            if _is_enabled(row)
        ]

    def list_all(self) -> list[CsvModelRecord]:
        """有効なモデル定義をすべて返す。

        Returns
        -------
        list[CsvModelRecord]
            有効フラグが立っているモデル定義一覧。
        """
        return list(self._models)

    def get(self, model_alias: str) -> CsvModelRecord:
        """モデル別名からモデル定義を1件取得する。

        Parameters
        ----------
        model_alias : str
            取得対象のモデル別名。

        Returns
        -------
        CsvModelRecord
            条件に一致したモデル定義。

        Raises
        ------
        ModelDefinitionNotFoundError
            一致するモデル別名が存在しない場合。
        """
        for model in self._models:
            if model.model_alias == model_alias:
                return model
        raise ModelDefinitionNotFoundError(
            f"モデル別名 '{model_alias}' に対応するモデル定義が見つかりません"
        )

from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, Field


class AIModelProvider(StrEnum):
    """AIモデルの提供元種別。litellmのコールの仕組みに影響する"""

    AZURE_OPENAI = "azure_openai"
    MICROSOFT_FOUNDRY = "microsoft_foundry"


class AIModelConfig(BaseModel):
    """実行したいAIモデルの設定情報"""

    model_name: Annotated[
        str, Field(title="モデル名", description="使用するAIモデルの名前")
    ]
    api_version: Annotated[
        str, Field(title="APIバージョン", description="使用するAPIのバージョン")
    ]
    provider_type: Annotated[
        AIModelProvider,
        Field("auto", title="プロバイダータイプ", description="AIモデルの提供元タイプ"),
    ]

    @property
    def litellm_model_params(self) -> dict[str, str]:

        match self.provider_type:
            case AIModelProvider.AZURE_OPENAI, AIModelProvider.MICROSOFT_FOUNDRY:
                return {
                    "api_version": self.api_version,
                }
            case _:
                return {}

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_PATH = Path(__file__).parent.parent / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ENV_PATH),
        extra="ignore",
    )

    # MS Foundry
    project_api_endpoint: str = Field(..., description="MS FoundryのAPIエンドポイント")

    # AOAI
    azure_openai_api_key: str = Field(..., description="Azure OpenAIのAPIキー")
    azure_openai_endpoint: str = Field(..., description="Azure OpenAIのエンドポイント")
    azure_openai_api_version: str = Field(
        ..., description="Azure OpenAIのAPIバージョン"
    )
    azure_embedding_deployment_name: str = Field(
        "text-embedding-3-large", description="Azure OpenAIの埋め込みデプロイメント名"
    )

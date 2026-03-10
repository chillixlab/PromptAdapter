from prompt_adapter.ai_model_runner.domain import AIModelConfig, AIModelConnection
from prompt_adapter.ai_model_runner.repositories import (
    CsvConnectionRepository,
    CsvModelRepository,
)


class ModelCatalog:
    """モデル定義と接続定義をまとめて解決するユーティリティ。"""

    def __init__(
        self,
        model_repository: CsvModelRepository,
        connection_repository: CsvConnectionRepository,
    ):
        """モデルリポジトリと接続リポジトリを束ねて保持する。

        Parameters
        ----------
        model_repository : CsvModelRepository
            モデル定義を解決するリポジトリ。
        connection_repository : CsvConnectionRepository
            接続定義を解決するリポジトリ。
        """
        self.model_repository = model_repository
        self.connection_repository = connection_repository

    def resolve(self, model_alias: str) -> tuple[AIModelConfig, AIModelConnection]:
        """モデル別名から実行用のモデル設定と接続設定を取得する。

        Parameters
        ----------
        model_alias : str
            解決対象のモデル別名。

        Returns
        -------
        tuple[AIModelConfig, AIModelConnection]
            実行用モデル設定と接続設定の組。

        Raises
        ------
        ModelDefinitionNotFoundError
            一致するモデル別名が存在しない場合。
        """
        model_record = self.model_repository.get(model_alias)
        connection_record = self.connection_repository.get(model_record.connection_name)
        return model_record.to_domain(), connection_record.to_domain()

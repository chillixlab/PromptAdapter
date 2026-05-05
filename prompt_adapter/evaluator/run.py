import litellm
from datasets import Dataset
from langchain_openai import AzureOpenAIEmbeddings
from pydantic import SecretStr
from ragas import evaluate
from ragas.dataset_schema import EvaluationResult
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import llm_factory
from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

from prompt_adapter.config import Settings
from prompt_adapter.evaluator.eval_models import EvalResult
from prompt_adapter.logger import logger


class Evaluator:
    def __init__(self, evaluate_llm_name: str = "gpt-5.2-chat"):
        """
        Evaluatorの初期化

        Parameters
        ----------
        evaluate_llm_name : str
            RAGAS評価に使用するLLMのモデル名（例：gpt-5.2-chat）
        """

        logger.info("RAGAS Evaluatorを初期化します")

        # litellmの設定: サポートされていないパラメータを自動的に削除
        litellm.drop_params = True

        self.settings = Settings()

        self.azure_llm = llm_factory(
            f"azure/{evaluate_llm_name}",
            provider="litellm",
            client=litellm.completion,
            api_base=self.settings.azure_openai_endpoint,
            api_key=self.settings.azure_openai_api_key,
            api_version=self.settings.azure_openai_api_version,
            # system_prompt="オプションで設定可能"
            temperature=None,  # temperatureを無効化
        )

        # embed_query()が必要な為、仕方なくLangChainを使用
        # Litellmembeddingsにはメソッドが無かった
        langchain_embeddings = AzureOpenAIEmbeddings(
            model=self.settings.azure_embedding_deployment_name,
            azure_endpoint=self.settings.azure_openai_endpoint,
            api_key=SecretStr(self.settings.azure_openai_api_key),
            api_version=self.settings.azure_openai_api_version,
        )

        # RAGASのLangchainEmbeddingsWrapperでラップ
        self.azure_embeddings = LangchainEmbeddingsWrapper(langchain_embeddings)

    def evaluate(self, model_name: str, dataset: Dataset) -> EvalResult:
        """
        RAGAS評価を実行

        Parameters
        ----------
        model_name : str
            評価対象の回答生成モデル名
        dataset : Dataset
            評価に使用するデータセット

        Returns
        -------
        EvalResult
            各評価指標のスコアを含む辞書
        """

        # 評価メトリクスの定義
        metrics = [
            ContextPrecision(llm=self.azure_llm),
            ContextRecall(llm=self.azure_llm),
            Faithfulness(llm=self.azure_llm),
            AnswerRelevancy(llm=self.azure_llm, embeddings=self.azure_embeddings),
        ]

        # 評価の実行
        results = evaluate(dataset, metrics=metrics)
        if not isinstance(results, EvaluationResult):
            raise ValueError(
                "RAGASの評価結果がEvaluationResultのインスタンスではありません"
            )

        # EvalResultモデルに変換
        eval_result = EvalResult.from_ragas_result(
            ragas_result=results,
            model_name=model_name,
            evaluation_model_name=model_name,
        )

        return eval_result

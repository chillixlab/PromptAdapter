from typing import Annotated, Any

from pydantic import BaseModel, Field


class EvalResult(BaseModel):
    """RAGAS評価結果のデータモデル"""

    model_name: Annotated[
        str,
        Field(title="回答生成モデル名", description="回答生成を行ったLLMのモデル名"),
    ]

    answer_relevancy: Annotated[
        float,
        Field(
            title="Answer Relevancy", description="データセット全体の回答の関連性スコア"
        ),
    ]
    context_precision: Annotated[
        float,
        Field(
            title="Context Precision",
            description="データセット全体のコンテキストの精度スコア",
        ),
    ]
    context_recall: Annotated[
        float,
        Field(
            title="Context Recall",
            description="データセット全体のコンテキストの再現率スコア",
        ),
    ]
    faithfulness: Annotated[
        float,
        Field(title="Faithfulness", description="データセット全体の回答忠実度スコア"),
    ]

    evaluation_model_name: Annotated[
        str,
        Field(title="評価実行モデル名", description="RAGAS評価を行ったLLMのモデル名"),
    ]

    each_metrics: Annotated[
        list[dict],
        Field(
            title="各データの評価結果",
            description="各データポイントごとの詳細な評価結果を含む辞書",
        ),
    ]

    @classmethod
    def from_ragas_result(
        cls,
        ragas_result,
        model_name: str,
        evaluation_model_name: str,
    ) -> "EvalResult":
        """
        RAGAS評価結果からEvalResultインスタンスを生成するファクトリメソッド

        Parameters
        ----------
        ragas_result : ragas.evaluation.Result
            RAGASによる評価結果
        model_name : str
            評価対象のモデル名
        evaluation_model_name : str
            評価に使用したモデル名

        Returns
        -------
        EvalResult
            変換されたEvalResultインスタンス
        """
        # RAGAS結果をDataFrameに変換して統計値を計算
        df = ragas_result.to_pandas()

        return cls(
            model_name=model_name,
            answer_relevancy=float(df["answer_relevancy"].mean()),
            context_precision=float(df["context_precision"].mean()),
            context_recall=float(df["context_recall"].mean()),
            faithfulness=float(df["faithfulness"].mean()),
            evaluation_model_name=evaluation_model_name,
            each_metrics=df.to_dict(orient="records"),
        )

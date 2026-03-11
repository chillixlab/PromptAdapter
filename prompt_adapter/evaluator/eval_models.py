from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, Field
from ragas.dataset_schema import EvaluationResult


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
        ragas_result: EvaluationResult,
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


class JaQuADDatasetRecord(BaseModel):
    """JaQuAD の評価用データセット行を表すデータモデル"""

    question_id: Annotated[
        int,
        Field(title="質問ID", description="JaQuAD 内での質問識別子"),
    ]
    context_id: Annotated[
        int,
        Field(title="コンテキストID", description="JaQuAD 内でのコンテキスト識別子"),
    ]
    answer: Annotated[
        str,
        Field(title="正解回答", description="データセットに含まれる正解回答文字列"),
    ]


class JaQuADLLMResultRecord(BaseModel):
    """JaQuAD のモデル単位生成結果行を表すデータモデル"""

    question_id: Annotated[
        int,
        Field(title="質問ID", description="JaQuAD 内での質問識別子"),
    ]
    context_id: Annotated[
        int,
        Field(title="コンテキストID", description="JaQuAD 内でのコンテキスト識別子"),
    ]
    answer: Annotated[
        str,
        Field(title="モデル回答", description="LLM が生成した回答文字列"),
    ]
    system_prompt: Annotated[
        str,
        Field(title="システムプロンプト", description="回答生成時に使用したシステムプロンプト",),
    ]


class JaQuADExactMatchMetricResult(BaseModel):
    """JaQuAD の完全一致評価結果を表すデータモデル"""

    score: Annotated[
        float,
        Field(title="スコア", description="評価指標のスコア"),
    ]
    correct_count: Annotated[
        int | None,
        Field(default=None, title="正解件数", description="指標に応じた正解件数"),
    ]
    incorrect_count: Annotated[
        int | None,
        Field(default=None, title="不正解件数", description="指標に応じた不正解件数"),
    ]


class JaQuADEvaluationResult(BaseModel):
    """JaQuAD の単一モデル評価結果を表すデータモデル"""

    dataset_name: Annotated[
        str,
        Field(title="データセット名", description="評価対象データセット名"),
    ]
    dataset_split: Annotated[
        str,
        Field(title="データ分割", description="train や dev などのデータ分割名"),
    ]
    source_dataset_file: Annotated[
        str,
        Field(title="元データセットCSV名", description="評価元のデータセット CSV ファイル名"),
    ]
    source_llm_result_file: Annotated[
        str,
        Field(title="元生成結果CSV名", description="評価元の生成結果 CSV ファイル名"),
    ]
    model_name: Annotated[
        str,
        Field(title="モデル名", description="評価対象のモデル名"),
    ]
    generated_at: Annotated[
        datetime | None,
        Field(default=None, title="生成日時", description="評価結果 JSON の生成日時"),
    ]
    total_questions: Annotated[
        int,
        Field(title="総質問数", description="評価対象となった質問数"),
    ]
    metrics: Annotated[
        dict[str, JaQuADExactMatchMetricResult],
        Field(title="評価指標結果", description="評価指標名をキーにした評価結果一覧"),
    ]

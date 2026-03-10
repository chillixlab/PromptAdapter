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


class JaQuADLLMAnswer(BaseModel):
    """JaQuAD の各モデル回答を表すデータモデル"""

    answer: Annotated[
        str,
        Field(title="モデル回答", description="LLM が生成した回答文字列"),
    ]
    is_exact_match: Annotated[
        bool | None,
        Field(
            default=None,
            title="完全一致判定",
            description="正解文字列と完全一致したかどうかの判定結果",
        ),
    ]


class JaQuADResultRecord(BaseModel):
    """JaQuAD の1問分の実行結果を表すデータモデル"""

    question_id: Annotated[
        int,
        Field(title="質問ID", description="JaQuAD 内での質問識別子"),
    ]
    context_id: Annotated[
        int,
        Field(title="コンテキストID", description="JaQuAD 内でのコンテキスト識別子"),
    ]
    title: Annotated[
        str,
        Field(title="記事タイトル", description="元記事のタイトル"),
    ]
    question_type: Annotated[
        str,
        Field(title="質問種別", description="JaQuAD が持つ質問タイプ"),
    ]
    question: Annotated[
        str,
        Field(title="質問文", description="LLM に与えた質問文"),
    ]
    ground_truth_answer: Annotated[
        str,
        Field(title="正解文字列", description="JaQuAD に含まれる正解回答"),
    ]
    llm_answers: Annotated[
        dict[str, JaQuADLLMAnswer],
        Field(
            title="モデル回答一覧",
            description="モデル名をキーに持つ回答結果辞書",
        ),
    ]


class JaQuADExecutionResult(BaseModel):
    """JaQuAD の複数モデル実行結果を表すデータモデル"""

    dataset_name: Annotated[
        str,
        Field(title="データセット名", description="評価対象データセット名"),
    ]
    dataset_split: Annotated[
        str,
        Field(title="データ分割", description="train や dev などのデータ分割名"),
    ]
    source_file: Annotated[
        str,
        Field(title="元 CSV ファイル名", description="実行元となった JaQuAD CSV ファイル名"),
    ]
    generated_at: Annotated[
        datetime | None,
        Field(
            default=None,
            title="生成日時",
            description="実行結果 JSON を生成した日時",
        ),
    ]
    models: Annotated[
        list[str],
        Field(title="対象モデル一覧", description="実行対象としたモデル名一覧"),
    ]
    records: Annotated[
        list[JaQuADResultRecord],
        Field(title="実行結果一覧", description="各質問ごとの実行結果一覧"),
    ]


class ExactMatchEvaluationMethod(BaseModel):
    """完全一致評価のルールを表すデータモデル"""

    name: Annotated[
        str,
        Field(title="評価手法名", description="評価手法の識別名"),
    ]
    description: Annotated[
        str,
        Field(title="評価手法説明", description="評価手法の説明文"),
    ]


class ModelExactMatchMetric(BaseModel):
    """モデル単位の完全一致評価結果を表すデータモデル"""

    model_name: Annotated[
        str,
        Field(title="モデル名", description="評価対象のモデル名"),
    ]
    correct_count: Annotated[
        int,
        Field(title="正解件数", description="完全一致した件数"),
    ]
    incorrect_count: Annotated[
        int,
        Field(title="不正解件数", description="完全一致しなかった件数"),
    ]
    exact_match_accuracy: Annotated[
        float,
        Field(title="完全一致正解率", description="完全一致した割合"),
    ]
    matched_question_ids: Annotated[
        list[int],
        Field(title="一致質問 ID 一覧", description="完全一致した質問 ID の一覧"),
    ]
    mismatched_question_ids: Annotated[
        list[int],
        Field(title="不一致質問 ID 一覧", description="完全一致しなかった質問 ID の一覧"),
    ]


class JaQuADExactMatchEvalResult(BaseModel):
    """JaQuAD の完全一致評価結果を表すデータモデル"""

    dataset_name: Annotated[
        str,
        Field(title="データセット名", description="評価対象データセット名"),
    ]
    dataset_split: Annotated[
        str,
        Field(title="データ分割", description="train や dev などのデータ分割名"),
    ]
    source_file: Annotated[
        str,
        Field(title="元 CSV ファイル名", description="元になった JaQuAD CSV ファイル名"),
    ]
    source_result_file: Annotated[
        str,
        Field(title="元結果 JSON 名", description="評価元の実行結果 JSON ファイル名"),
    ]
    generated_at: Annotated[
        datetime | None,
        Field(default=None, title="生成日時", description="評価結果 JSON の生成日時"),
    ]
    evaluation_method: Annotated[
        ExactMatchEvaluationMethod,
        Field(title="評価手法情報", description="完全一致評価の定義情報"),
    ]
    total_questions: Annotated[
        int,
        Field(title="総質問数", description="評価対象となった質問数"),
    ]
    model_metrics: Annotated[
        list[ModelExactMatchMetric],
        Field(title="モデル別評価結果", description="モデル単位の完全一致評価結果一覧"),
    ]

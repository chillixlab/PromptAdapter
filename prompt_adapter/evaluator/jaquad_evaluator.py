import json
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

from prompt_adapter.evaluator.eval_models import (
    ExactMatchEvaluationMethod,
    JaQuADExecutionResult,
    JaQuADExactMatchEvalResult,
    ModelExactMatchMetric,
)


class _ModelExactMatchStat(NamedTuple):
    """完全一致評価の途中集計を保持する。"""

    correct_count: int
    incorrect_count: int
    matched_question_ids: list[int]
    mismatched_question_ids: list[int]


class JaQuADEvaluator:
    """JaQuAD の完全一致評価を行う evaluator"""

    def evaluate_exact_match(
        self,
        execution_result: JaQuADExecutionResult,
        source_result_file: str,
    ) -> JaQuADExactMatchEvalResult:
        """
        JaQuAD の実行結果に対して完全一致評価を行う。

        Parameters
        ----------
        execution_result : JaQuADExecutionResult
            複数モデルの回答結果を含む実行結果
        source_result_file : str
            評価元になった実行結果 JSON ファイル名

        Returns
        -------
        JaQuADExactMatchEvalResult
            モデル単位の完全一致評価結果
        """
        total_questions = len(execution_result.records)
        if total_questions == 0:
            raise ValueError("JaQuAD の評価対象レコードが存在しません")

        model_stats: dict[str, _ModelExactMatchStat] = {
            model_name: _ModelExactMatchStat(
                correct_count=0,
                incorrect_count=0,
                matched_question_ids=[],
                mismatched_question_ids=[],
            )
            for model_name in execution_result.models
        }

        for record in execution_result.records:
            normalized_ground_truth_answer = self._normalize_answer(
                record.ground_truth_answer
            )

            for model_name in execution_result.models:
                llm_answer = record.llm_answers.get(model_name)
                model_answer = "" if llm_answer is None else llm_answer.answer
                is_exact_match = (
                    self._normalize_answer(model_answer) == normalized_ground_truth_answer
                )
                current_stat = model_stats[model_name]

                if is_exact_match:
                    current_stat.matched_question_ids.append(record.question_id)
                    model_stats[model_name] = _ModelExactMatchStat(
                        correct_count=current_stat.correct_count + 1,
                        incorrect_count=current_stat.incorrect_count,
                        matched_question_ids=current_stat.matched_question_ids,
                        mismatched_question_ids=current_stat.mismatched_question_ids,
                    )
                else:
                    current_stat.mismatched_question_ids.append(record.question_id)
                    model_stats[model_name] = _ModelExactMatchStat(
                        correct_count=current_stat.correct_count,
                        incorrect_count=current_stat.incorrect_count + 1,
                        matched_question_ids=current_stat.matched_question_ids,
                        mismatched_question_ids=current_stat.mismatched_question_ids,
                    )

        model_metrics = [
            ModelExactMatchMetric(
                model_name=model_name,
                correct_count=model_stats[model_name].correct_count,
                incorrect_count=model_stats[model_name].incorrect_count,
                exact_match_accuracy=(
                    model_stats[model_name].correct_count / total_questions
                ),
                matched_question_ids=model_stats[model_name].matched_question_ids,
                mismatched_question_ids=model_stats[
                    model_name
                ].mismatched_question_ids,
            )
            for model_name in execution_result.models
        ]

        return JaQuADExactMatchEvalResult(
            dataset_name=execution_result.dataset_name,
            dataset_split=execution_result.dataset_split,
            source_file=execution_result.source_file,
            source_result_file=source_result_file,
            generated_at=datetime.now().astimezone(),
            evaluation_method=ExactMatchEvaluationMethod(
                name="完全一致",
                description="各モデルの回答文字列が ground_truth_answer と完全一致した件数と正解率を集計",
            ),
            total_questions=total_questions,
            model_metrics=model_metrics,
        )

    def evaluate_exact_match_from_file(
        self,
        result_file_path: str | Path,
    ) -> JaQuADExactMatchEvalResult:
        """
        JaQuAD の実行結果 JSON を読み込み、完全一致評価を行う。

        Parameters
        ----------
        result_file_path : str | Path
            評価対象の実行結果 JSON パス

        Returns
        -------
        JaQuADExactMatchEvalResult
            モデル単位の完全一致評価結果
        """
        path = Path(result_file_path)
        with path.open(encoding="utf-8") as file:
            execution_result = JaQuADExecutionResult.model_validate(json.load(file))

        return self.evaluate_exact_match(
            execution_result=execution_result,
            source_result_file=path.name,
        )

    def save_exact_match_evaluation(
        self,
        evaluation_result: JaQuADExactMatchEvalResult,
        output_file_path: str | Path,
    ) -> None:
        """
        完全一致評価結果を JSON ファイルとして保存する。

        Parameters
        ----------
        evaluation_result : JaQuADExactMatchEvalResult
            保存対象の完全一致評価結果
        output_file_path : str | Path
            出力先の JSON ファイルパス
        """
        path = Path(output_file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            json.dump(
                evaluation_result.model_dump(mode="json"),
                file,
                ensure_ascii=False,
                indent=2,
            )

    @staticmethod
    def _normalize_answer(answer: str) -> str:
        """完全一致評価用に回答文字列を正規化する。"""
        return answer.strip()

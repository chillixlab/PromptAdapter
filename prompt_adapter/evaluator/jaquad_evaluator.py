import csv
import json
from datetime import datetime
from pathlib import Path

from prompt_adapter.evaluator.eval_models import (
    JaQuADDatasetRecord,
    JaQuADEvaluationMetricResult,
    JaQuADEvaluationResult,
    JaQuADLLMResultRecord,
)


class JaQuADEvaluator:
    """JaQuAD の完全一致評価を行う evaluator"""

    def evaluate_exact_match(
        self,
        dataset_name: str,
        dataset_split: str,
        source_dataset_file: str,
        source_llm_result_file: str,
        model_name: str,
        dataset_records: list[JaQuADDatasetRecord],
        llm_result_records: list[JaQuADLLMResultRecord],
    ) -> JaQuADEvaluationResult:
        """
        JaQuAD の単一モデル生成結果に対して完全一致評価を行う。

        Parameters
        ----------
        dataset_name : str
            評価対象データセット名。
        dataset_split : str
            評価対象データの split 名。
        source_dataset_file : str
            評価元のデータセット CSV ファイル名。
        source_llm_result_file : str
            評価元の生成結果 CSV ファイル名。
        model_name : str
            評価対象のモデル名。
        dataset_records : list[JaQuADDatasetRecord]
            正解データ一覧。
        llm_result_records : list[JaQuADLLMResultRecord]
            モデル生成結果一覧。

        Returns
        -------
        JaQuADEvaluationResult
            単一モデルの評価結果。
        """
        if not dataset_records:
            raise ValueError("JaQuAD の評価対象データセットが存在しません")

        dataset_record_map = {
            (record.question_id, record.context_id): record for record in dataset_records
        }
        llm_result_record_map = {
            (record.question_id, record.context_id): record for record in llm_result_records
        }

        if len(dataset_record_map) != len(dataset_records):
            raise ValueError("データセット CSV に重複した question_id と context_id の組み合わせがあります")

        if len(llm_result_record_map) != len(llm_result_records):
            raise ValueError("生成結果 CSV に重複した question_id と context_id の組み合わせがあります")

        correct_count = 0
        incorrect_count = 0

        for record_key, dataset_record in dataset_record_map.items():
            llm_record = llm_result_record_map.get(record_key)
            if llm_record is None:
                raise ValueError(
                    "生成結果 CSV に対応する question_id と context_id の組み合わせが存在しません"
                )

            # 正規化して想定回答とあっているかを確認する
            is_exact_match = (
                self._normalize_answer(llm_record.answer)
                == self._normalize_answer(dataset_record.answer)
            )
            if is_exact_match:
                correct_count += 1
            else:
                incorrect_count += 1

        extra_record_keys = set(llm_result_record_map) - set(dataset_record_map)
        if extra_record_keys:
            raise ValueError(
                "生成結果 CSV にデータセット CSV に存在しない question_id と context_id の組み合わせがあります"
            )

        total_questions = len(dataset_records)
        score = correct_count / total_questions

        return JaQuADEvaluationResult(
            dataset_name=dataset_name,
            dataset_split=dataset_split,
            source_dataset_file=source_dataset_file,
            source_llm_result_file=source_llm_result_file,
            model_name=model_name,
            generated_at=datetime.now().astimezone(),
            total_questions=total_questions,
            metrics={
                "exact_match": JaQuADEvaluationMetricResult(
                    score=score,
                    correct_count=correct_count,
                    incorrect_count=incorrect_count,
                )
            },
        )

    def evaluate_exact_match_from_file(
        self,
        dataset_file_path: str | Path,
        llm_result_file_path: str | Path,
    ) -> JaQuADEvaluationResult:
        """
        データセット CSV と生成結果 CSV を読み込み、完全一致評価を行う。

        Parameters
        ----------
        dataset_file_path : str | Path
            評価対象のデータセット CSV パス。
        llm_result_file_path : str | Path
            評価対象の生成結果 CSV パス。

        Returns
        -------
        JaQuADEvaluationResult
            単一モデルの評価結果。
        """
        dataset_path = Path(dataset_file_path)
        llm_result_path = Path(llm_result_file_path)

        dataset_records = self._load_dataset_records(dataset_path)
        llm_result_records = self._load_llm_result_records(llm_result_path)

        return self.evaluate_exact_match(
            dataset_name="JaQuAD",
            dataset_split=self._extract_dataset_split(dataset_path.name),
            source_dataset_file=dataset_path.name,
            source_llm_result_file=llm_result_path.name,
            model_name=self._extract_model_name(llm_result_path.name),
            dataset_records=dataset_records,
            llm_result_records=llm_result_records,
        )

    def save_exact_match_evaluation(
        self,
        evaluation_result: JaQuADEvaluationResult,
        output_file_path: str | Path,
    ) -> None:
        """
        完全一致評価結果を JSON ファイルとして保存する。

        Parameters
        ----------
        evaluation_result : JaQuADEvaluationResult
            保存対象の評価結果。
        output_file_path : str | Path
            出力先の JSON ファイルパス。
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
    def _load_dataset_records(file_path: Path) -> list[JaQuADDatasetRecord]:
        """データセット CSV を読み込む。"""
        with file_path.open(encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            return [
                JaQuADDatasetRecord(
                    question_id=int(row["question_id"]),
                    context_id=int(row["context_id"]),
                    answer=row["answer"],
                )
                for row in reader
            ]

    @staticmethod
    def _load_llm_result_records(file_path: Path) -> list[JaQuADLLMResultRecord]:
        """生成結果 CSV を読み込む。"""
        with file_path.open(encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            return [
                JaQuADLLMResultRecord(
                    question_id=int(row["question_id"]),
                    context_id=int(row["context_id"]),
                    answer=row["answer"],
                )
                for row in reader
            ]

    @staticmethod
    def _extract_dataset_split(dataset_file_name: str) -> str:
        """データセットファイル名から split 名を抽出する。"""
        file_stem_parts = Path(dataset_file_name).stem.split("_")
        if len(file_stem_parts) < 3:
            raise ValueError("データセット CSV ファイル名から split を判定できません")
        return file_stem_parts[1]

    @staticmethod
    def _extract_model_name(llm_result_file_name: str) -> str:
        """生成結果ファイル名からモデル名を抽出する。"""
        file_stem_parts = Path(llm_result_file_name).stem.split("_")
        if len(file_stem_parts) < 6:
            raise ValueError("生成結果 CSV ファイル名からモデル名を判定できません")
        return "_".join(file_stem_parts[5:])

    @staticmethod
    def _normalize_answer(answer: str) -> str:
        """完全一致評価用に回答文字列を正規化する。"""
        return answer.strip()

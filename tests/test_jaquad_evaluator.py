import json

from prompt_adapter.evaluator.jaquad_evaluator import JaQuADEvaluator


def test_完全一致評価結果が単一モデルJSON形式で生成されること(tmp_path):
    """完全一致評価結果が単一モデルJSON形式で生成されることを確認する。

    Parameters
    ----------
    tmp_path : Path
        テスト用の一時ディレクトリ。
    """
    # GIVEN: データセットCSVとsystem_promptを含む生成結果CSVを用意する
    dataset_file = tmp_path / "jaquad_dev_0000.csv"
    dataset_file.write_text(
        "question_id,context_id,title,context,question,answer,question_type\n"
        "1,1,記事A,文脈A,質問A,奈良,type_a\n"
        "2,1,記事A,文脈A,質問B,約15メートル,type_a\n",
        encoding="utf-8",
    )
    llm_result_file = tmp_path / "llm_result_jaquad_dev_0000_gpt-4.1.csv"
    llm_result_file.write_text(
        "question_id,context_id,answer,system_prompt\n"
        "1,1,奈良,簡潔に答えてください\n"
        "2,1,15メートル,簡潔に答えてください\n",
        encoding="utf-8",
    )
    output_file = tmp_path / "evaluation_result_jaquad_dev_0000_gpt-4.1.json"
    evaluator = JaQuADEvaluator()

    # WHEN: 完全一致評価を実行して評価結果JSONを保存する
    result = evaluator.evaluate_exact_match_from_file(dataset_file, llm_result_file)
    llm_records = evaluator._load_llm_result_records(llm_result_file)
    evaluator.save_exact_match_evaluation(result, output_file)
    saved_result = json.loads(output_file.read_text(encoding="utf-8"))

    # THEN: system_promptを保持したまま完全一致評価結果が保存される
    assert llm_records[0].system_prompt == "簡潔に答えてください"
    assert result.dataset_name == "JaQuAD"
    assert result.dataset_split == "dev"
    assert result.source_dataset_file == "jaquad_dev_0000.csv"
    assert result.source_llm_result_file == "llm_result_jaquad_dev_0000_gpt-4.1.csv"
    assert result.model_name == "gpt-4.1"
    assert result.total_questions == 2
    assert result.metrics["exact_match"].score == 0.5
    assert result.metrics["exact_match"].correct_count == 1
    assert result.metrics["exact_match"].incorrect_count == 1
    assert saved_result["model_name"] == "gpt-4.1"
    assert saved_result["metrics"]["exact_match"]["score"] == 0.5


def test_system_prompt列がない生成結果CSVでは例外になること(tmp_path):
    """system_prompt列がない生成結果CSVでは例外になることを確認する。

    Parameters
    ----------
    tmp_path : Path
        テスト用の一時ディレクトリ。
    """
    # GIVEN: system_prompt列を持たない生成結果CSVを用意する
    llm_result_file = tmp_path / "llm_result_jaquad_dev_0000_gpt-4.1.csv"
    llm_result_file.write_text(
        "question_id,context_id,answer\n1,1,奈良\n",
        encoding="utf-8",
    )

    # WHEN: 生成結果CSVを読み込む
    try:
        JaQuADEvaluator()._load_llm_result_records(llm_result_file)
    except KeyError as error:
        # THEN: system_prompt列不足を示す例外が発生する
        assert str(error) == "'system_prompt'"
    else:
        raise AssertionError("KeyError が発生しませんでした")


def test_生成結果CSVに不足行がある場合は例外になること(tmp_path):
    """生成結果CSVに不足行がある場合は例外になることを確認する。

    Parameters
    ----------
    tmp_path : Path
        テスト用の一時ディレクトリ。
    """
    # GIVEN: データセットCSVより少ない行数の生成結果CSVを用意する
    dataset_file = tmp_path / "jaquad_dev_0000.csv"
    dataset_file.write_text(
        "question_id,context_id,title,context,question,answer,question_type\n"
        "1,1,記事A,文脈A,質問A,奈良,type_a\n"
        "2,1,記事A,文脈A,質問B,約15メートル,type_a\n",
        encoding="utf-8",
    )
    llm_result_file = tmp_path / "llm_result_jaquad_dev_0000_gpt-4.1.csv"
    llm_result_file.write_text(
        "question_id,context_id,answer,system_prompt\n1,1,奈良,簡潔に答えてください\n",
        encoding="utf-8",
    )
    evaluator = JaQuADEvaluator()

    # WHEN: 完全一致評価を実行する
    try:
        evaluator.evaluate_exact_match_from_file(dataset_file, llm_result_file)
    except ValueError as error:
        # THEN: 対応する生成結果不足を示す例外が発生する
        assert (
            str(error)
            == "生成結果 CSV に対応する question_id と context_id の組み合わせが存在しません"
        )
    else:
        raise AssertionError("ValueError が発生しませんでした")

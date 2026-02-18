import pandas as pd
import requests

from prompt_adapter.logger import logger


class JaQuADPreparer:
    def __init__(self):
        self.jaquad_base_url = (
            "https://api.github.com/repos/SkelterLabsInc/JaQuAD/contents/data"
        )
        self.path_train = self.jaquad_base_url + "/train/"
        self.path_dev = self.jaquad_base_url + "/dev/"

        self.output_folder = "../instance/datasets/JaQuAD/"

    def load_jaquad_all(self) -> None:
        logger.info(
            "JaQuADデータセットの全てのJsonファイルに対するCSV構築を開始します。"
        )
        response = requests.get(self.path_train)
        files = response.json()

        logger.info(files)

        for item in files:
            if item["name"].endswith(".json"):
                # jsonファイルをcsvに変換する
                logger.info(f"Processing {item['name']}")
                df = pd.read_json(item["download_url"])

                rows = []
                for num in range(len(df)):
                    title = df.iloc[num]["data"]["title"]

                    for paragraph in df.iloc[num]["data"]["paragraphs"]:
                        context = paragraph["context"]

                        for qa in paragraph["qas"]:
                            question = qa["question"]
                            qid = qa["id"]
                            question_type = qa.get("question_type", "")
                            answers = qa["answers"][0]["text"]

                        rows.append(
                            {
                                "question_id": qid,
                                "title": title,
                                "context": context,
                                "question": question,
                                "answer": answers,
                                "question_type": question_type,
                            }
                        )

                qa_df = pd.DataFrame(rows)

                # jsonファイル毎にcsvファイルを出力する
                output_path = self.output_folder + item["name"].replace(".json", ".csv")
                qa_df.to_csv(output_path, index=False)
                logger.info(f"CSVファイルを {output_path} に保存しました。")

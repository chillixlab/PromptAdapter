import os

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

    def load_jaquad_all(self, output_folder: str) -> None:
        logger.info("JaQuADデータセットのCSV構築を開始します。")

        # 出力フォルダが存在しない場合は作成
        os.makedirs(output_folder, exist_ok=True)

        for path in [self.path_train, self.path_dev]:
            response = requests.get(path)
            files = response.json()

            for item in files:
                if not item["name"].endswith(".json"):
                    continue

                logger.info(f"Processing {item['name']}")
                df = pd.read_json(item["download_url"])

                rows = []
                context_id = 1
                question_id = 1
                for num in range(len(df)):
                    title = df.iloc[num]["data"]["title"]

                    for paragraph in df.iloc[num]["data"]["paragraphs"]:
                        context = paragraph["context"]

                        for qa in paragraph["qas"]:
                            question = qa["question"]
                            question_type = qa.get("question_type", "")
                            answer = qa["answers"][0]["text"]

                            rows.append(
                                {
                                    "question_id": question_id,
                                    "context_id": context_id,
                                    "title": title,
                                    "context": context,
                                    "question": question,
                                    "answer": answer,
                                    "question_type": question_type,
                                }
                            )

                            question_id += 1  # 質問IDを増加
                        context_id += 1  # コンテキストが変わったら増加

                qa_df = pd.DataFrame(rows)

                output_path = os.path.join(
                    output_folder, item["name"].replace(".json", ".csv")
                )

                qa_df.to_csv(output_path, index=False)
                logger.info(f"CSV保存: {output_path}")

        logger.info("JaQuADデータセットのCSV構築が完了しました。")

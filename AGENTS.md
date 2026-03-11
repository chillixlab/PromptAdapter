# AGENTS.md

このファイルはこのリポジトリで作業する AI コーディングエージェント向けのガイドです。

---

## プロジェクト概要

**PromptAdapter** は Python 3.12 製の RAG 評価・実験フレームワークです。
LiteLLM / LangChain / RAGAS を組み合わせて、複数の LLM プロバイダに対応した
プロンプト評価パイプラインを提供します。

- パッケージマネージャ: `uv`
- タスクランナー: `mise`
- リンター・フォーマッタ: `ruff`
- 型チェッカー: `ty` (Astral)
- テストフレームワーク: `pytest`

---

## 絶対ルール（Copilot / GitHub instructions より）

> **自然言語はすべて日本語を使うこと。コードコメントも必ず日本語で書くこと。**
> 指示がない限り英語を使ってはいけない。

これは `.github/copilot-instructions.md` および `.github/instructions/code.instructions.md`
で定義されたルールです。エージェントも同様に従うこと。

---

## ビルド・開発コマンド

```bash
# 依存パッケージのインストール
mise init           # uv sync --all-extras

# リント（静的解析 + 型チェック）
mise lint           # uv run ruff check . && uv run ty check .

# フォーマット（自動修正）
mise format         # uv run ruff check --fix . && uv run ruff format .

# テスト実行（全件）
mise test           # uv run pytest tests
```

mise を使わずに直接実行する場合:

```bash
uv sync --all-extras     # 依存インストール
uv run ruff check .      # リント
uv run ruff format .     # フォーマット
uv run ty check .        # 型チェック
uv run pytest tests      # テスト
```

コミット前に必ず `mise format && mise lint` を実行すること。

---

## テスト実行

```bash
# 全テスト実行
uv run pytest tests

# 特定のファイルのみ実行
uv run pytest tests/test_sample.py

# テスト名（-k フィルタ）で絞り込み（日本語名も使用可）
uv run pytest -k "テスト内容のキーワード"

# 詳細出力付き
uv run pytest tests/test_sample.py -v
```

テストファイルは `tests/test_*.py` パターンで配置する。

---

## コードスタイル

### 基本方針（`.github/instructions/code.instructions.md` より）

- docstring は **numpy 形式** で書く
- 変数名・関数名は **意味のある名前** を使う
- コメントは **簡潔かつ具体的** に書く（日本語で）
- **マジックナンバーは避け、定数として定義**する
- コードの **可読性を重視**する

### ruff 設定（`pyproject.toml`）

```toml
[tool.ruff.lint]
extend-select = [
    "B",   # flake8-bugbear: 潜在的なバグを検出
    "I",   # isort: import 順の整形チェック
    "SIM", # flake8-simplify: 冗長な書き方の簡略化
    "UP",  # pyupgrade: 新しい Python 構文への改善提案
]
```

---

## 命名規則

| 要素 | 規則 | 例 |
|---|---|---|
| ファイル名・モジュール名 | `snake_case` | `ai_models.py`, `eval_models.py` |
| パッケージ（ディレクトリ）名 | `snake_case` | `dataset_preparer/`, `evaluator/` |
| クラス名 | `PascalCase` | `AIModelConfig`, `JaQuADPreparer`, `EvalResult` |
| 関数名・メソッド名 | `snake_case` | `load_jaquad_all()`, `from_ragas_result()` |
| 変数名 | `snake_case` | `model_name`, `output_folder`, `qa_df` |
| モジュールレベル定数 | `UPPER_SNAKE_CASE` | `ENV_PATH` |
| プライベート関数・変数 | 先頭アンダースコア `_` | `_get_log_level()` |
| Enum の値 | `UPPER_SNAKE_CASE` | `AZURE_OPENAI`, `MICROSOFT_FOUNDRY` |

---

## インポートスタイル

- **絶対インポートのみ**使用する（`from . import ...` などの相対インポートは使わない）
- インポート順: 標準ライブラリ → サードパーティ → 内部パッケージ（ruff の `I` ルールで自動整形）

```python
# 良い例
import logging
from pathlib import Path

import pandas as pd
from pydantic_settings import BaseSettings

from prompt_adapter.config import Settings
from prompt_adapter.logger import logger
```

---

## 型定義

- Python 3.12 以上の構文を使用（`ty` で型チェック）
- `Annotated[type, Field(...)]` を Pydantic フィールド定義に使用
- `StrEnum`（Python 3.11+）を文字列 Enum に使用（`.value` 不要で API 呼び出しに直接使用可能）
- `match/case`（Python 3.10+）をプロバイダ分岐などに使用

```python
# Pydantic フィールドの例
from typing import Annotated
from pydantic import BaseModel, Field

class EvalResult(BaseModel):
    context_recall: Annotated[float, Field(title="コンテキスト再現率", description="...")]
```

---

## エラー処理

- 型検証には `isinstance` チェック + `ValueError` を使用
- エラーメッセージは **日本語**で記述する
- Pydantic の `ValidationError` を活用してデータ境界での検証を行う
- 外部 API 呼び出しには `response.raise_for_status()` を使用して HTTP エラーを明示的に処理する

```python
# 良い例
if not isinstance(results, EvaluationResult):
    raise ValueError("RAGASの評価結果がEvaluationResultのインスタンスではありません")
```

---

## テストの書き方（`.github/instructions/test.instructions.md` より）

- テスト関数名: `test_<日本語でテスト内容>` 形式
- テスト内部は **GIVEN / WHEN / THEN** コメントで構造化する

```python
def test_評価結果が正常に変換されること():
    # GIVEN
    ragas_result = ...

    # WHEN
    result = EvalResult.from_ragas_result(ragas_result)

    # THEN
    assert result.context_recall == 0.9
```

---

## プロジェクト構造

```
prompt_adapter/              # メインパッケージ
├── config.py                # 設定（pydantic-settings、.env 読み込み）
├── logger.py                # 共有ロガー（全モジュールからインポート）
├── ai_model_runnner/        # LLM 実行レイヤー（※ディレクトリ名にタイポあり）
│   └── ai_models.py         # モデル設定の enum + Pydantic モデル
├── dataset_preparer/        # データセット取得・前処理レイヤー
│   └── jaquad_preparer.py   # JaQuAD データセットの CSV 構築
└── evaluator/               # 評価パイプラインレイヤー
    ├── eval_models.py        # 評価結果の Pydantic モデル
    └── run.py                # RAGAS 評価クラス
tests/                       # テストディレクトリ
experiments/                 # Jupyter ノートブック（研究・実験用）
instance/                    # 実行時生成ファイル（datasets/, experiments_result/）
```

---

## 設定管理

- `.env` ファイルをプロジェクトルートに配置（`config.py` が `pathlib.Path(__file__)` で解決）
- 必須設定は `Field(...)` で定義（デフォルト値なし）
- オプション設定はデフォルト値あり
- `Settings()` はクラスの `__init__` 内でインスタンス化する

---

## ロギング

```python
# 共有ロガーの使い方
from prompt_adapter.logger import logger

logger.info("処理を開始します")
```

- ログレベルは `LOG_LEVEL` 環境変数で制御
- ログメッセージは日本語で記述
- `@lru_cache(maxsize=1)` により `basicConfig` は一度だけ呼ばれる（重複ハンドラ登録を防止）

---

## CI パイプライン（`.github/workflows/qulity-test.yml`）

PR 時に以下が自動実行されます:

1. `uv run ruff check .` — リント
2. `uv run ty check .` — 型チェック
3. `uv run pytest tests` — テスト

ローカルでも `mise lint && mise test` で同等のチェックが可能です。

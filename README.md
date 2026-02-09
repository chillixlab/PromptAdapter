# PromptAdapter

LLMの検索パフォーマンスを改善するためのPrompt Adaperを検証・開発するリポジトリ

## 概要

このプロジェクトは、Azure OpenAIやMicrosoft Foundryなどの様々なAIモデルプロバイダーと連携し、LLMの検索パフォーマンスを向上させるためのPrompt Adapterを開発・検証するためのものです。

## 必要要件

- [mise](https://mise.jdx.dev/) - 開発ツールのバージョン管理ツール

## 開発環境のセットアップ

### 1. miseのインストール

miseをインストールしていない場合は、以下のコマンドでインストールします：

```bash
# macOS / Linux
curl https://mise.run | sh

# または Homebrew を使用する場合 (macOS)
brew install mise

# シェル設定を追加（例: bashの場合）
echo 'eval "$(mise activate bash)"' >> ~/.bashrc
source ~/.bashrc

# zshの場合
echo 'eval "$(mise activate zsh)"' >> ~/.zshrc
source ~/.zshrc
```

詳しいインストール方法は [mise公式ドキュメント](https://mise.jdx.dev/getting-started.html) を参照してください。

### 2. プロジェクトのセットアップ

```bash
# リポジトリをクローン
git clone https://github.com/chillixlab/PromptAdapter.git
cd PromptAdapter

# Python と uv を自動インストール（mise.tomlに基づく）
mise install

# 依存関係をインストール
mise run init
```

`mise install` コマンドにより、以下がインストールされます：
- Python 3.14
- uv (最新版)

### 3. 環境変数の設定

必要に応じて `.env` ファイルをプロジェクトルートに作成し、APIキーなどの環境変数を設定します：

```bash
# .env の例
AZURE_API_KEY=your_api_key_here
AZURE_API_BASE=https://your-resource.openai.azure.com/
```

## プロジェクト構造

```
PromptAdapter/
├── prompt-adapter/          # メインのパッケージディレクトリ
│   └── ai_model_runnner/    # AIモデル実行関連のモジュール
│       ├── ai_models.py     # AIモデルの設定とプロバイダー定義
│       ├── loader.py        # モデルローダー
│       └── run.py           # モデル実行ロジック
├── tests/                   # テストディレクトリ
├── mise.toml               # mise設定ファイル（ツールバージョン管理）
├── pyproject.toml          # Pythonプロジェクト設定
└── uv.lock                 # uvの依存関係ロックファイル
```

## 利用可能なコマンド

miseタスクを使用して以下のコマンドを実行できます：

```bash
# 依存関係のインストール
mise run init

# テストの実行
mise run test

# リントチェック
mise run lint

# コードフォーマット
mise run format
```

または、uvを直接使用することもできます：

```bash
# テスト実行
uv run pytest tests

# リント実行
uv run ruff check .
uv run ty check .

# フォーマット実行
uv run ruff format .
```

## 開発フロー

1. 機能の開発またはバグ修正を行う
2. コードをフォーマットする: `mise run format`
3. リントチェックを実行する: `mise run lint`
4. テストを実行する: `mise run test`
5. コミットしてプルリクエストを作成する

## ライセンス

このプロジェクトのライセンスについては、リポジトリの管理者にお問い合わせください。

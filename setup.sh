#!/bin/bash
# OpenFOAM AI Agent セットアップスクリプト
# 実行方法: bash setup.sh

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "=== OpenFOAM AI Agent セットアップ ==="
echo "プロジェクトディレクトリ: $PROJECT_DIR"

# 1. pip / venv のインストール
echo ""
echo "[1/4] Python 依存パッケージのシステムインストール..."
sudo apt-get update -qq
sudo apt-get install -y python3-pip python3.12-venv

# 2. 仮想環境の作成
echo ""
echo "[2/4] Python 仮想環境 (venv) を作成..."
cd "$PROJECT_DIR"
python3 -m venv venv
echo "  → venv/ を作成しました"

# 3. ライブラリのインストール
echo ""
echo "[3/4] 必要なライブラリをインストール..."
./venv/bin/pip install --upgrade pip -q
./venv/bin/pip install -r requirements.txt
./venv/bin/pip install -e .

# 4. .env ファイルの作成 (初回のみ)
echo ""
echo "[4/4] API キー設定ファイルを作成..."
if [ ! -f "$PROJECT_DIR/.env" ]; then
    cat > "$PROJECT_DIR/.env" << 'EOF'
# OpenAI を使う場合はこちらを設定
OPENAI_API_KEY=your-openai-api-key-here

# Anthropic (Claude) を使う場合はこちらを設定
ANTHROPIC_API_KEY=your-anthropic-api-key-here

# 使用するLLMプロバイダー: "openai" または "anthropic"
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o

# OpenFOAM の設定
OPENFOAM_VERSION=2512
OPENFOAM_ROOT=/usr/lib/openfoam/openfoam2512
EOF
    echo "  → .env を作成しました (APIキーを設定してください)"
else
    echo "  → .env は既に存在します (スキップ)"
fi

echo ""
echo "=== セットアップ完了！ ==="
echo ""
echo "次のステップ:"
echo "  1. .env ファイルを開いてAPIキーを設定してください"
echo "     例: code .env  または  nano .env"
echo ""
echo "  2. エージェントを起動するには:"
echo "     source venv/bin/activate"
echo "     cd $PROJECT_DIR"
echo "     python -m src.main run \"解析の説明\" -o ./output"
echo "     # または: openfoam-agent run \"解析の説明\" -o ./output"

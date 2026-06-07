# OpenFOAM AI Agent

自然言語の指示から OpenFOAM 解析ケースを自動生成・実行・可視化まで一括で行う AI エージェントです。

## デモ: 2D 円柱周りカルマン渦

```bash
python -m src.main run \
  "円柱周りの2D非定常カルマン渦流れ、流入速度0.15m/s、層流、Re=1000" \
  --stl ./cylinder_2d.stl \
  --output ./output/my_case
```

→ blockMesh → snappyHexMesh → potentialFoam → pimpleFoam → VTK 変換まで自動実行。

---

## アーキテクチャ

4 つのエージェントがパイプラインを構成します。

```
自然言語
    ↓
[Agent①] Preprocessing     自然言語 → SimulationSpec (solver, case_type, BC, Re数 など)
    ↓
[Agent②] RAG + Prompt Gen  ChromaDB からチュートリアル事例を検索・付加 → EnrichedContext
    ↓
[Agent③] OpenFOAM GPT      ケースファイル生成 → ソルバー実行 → 自己修正ループ (最大3回)
    ↓
[Agent④] Post-processing   残差チェック・物理妥当性検証 → レポート & ParaView パス案内
```

### 自己修正ループ

ソルバーが FATAL エラーで失敗した場合、エラーログを LLM に渡して `system/` ファイルを自動修正し、最大 3 回リトライします。

---

## 対応ケース

| `case_type`       | 説明                                  | ソルバー候補                |
|-------------------|---------------------------------------|-----------------------------|
| `channel_2d`      | 2D チャンネル・バックステップ等        | simpleFoam, pimpleFoam      |
| `channel_3d`      | 3D ダクト・室内流れ等                  | simpleFoam, pimpleFoam      |
| `snappy_2d`       | 2D 外部流れ（STL 必須、z 方向 1 セル） | pimpleFoam                  |
| `external_snappy` | 3D 外部流れ（STL 必須）               | simpleFoam, pimpleFoam      |

ソルバーは説明文から自動選択されます。

- 「定常」「steady」→ `simpleFoam`
- 「非定常」「カルマン渦」「unsteady」→ `pimpleFoam`

---

## プロジェクト構成

```
openfoam-ai-agent/
├── src/
│   ├── main.py                 ← CLI エントリポイント
│   ├── models.py               ← エージェント間データ契約 (SimulationSpec 等)
│   ├── runner.py               ← OpenFOAM コマンド実行ラッパー
│   ├── monitor.py              ← 残差モニタリング
│   ├── llm_client.py           ← OpenAI / Anthropic 統一クライアント
│   ├── config.py               ← 設定管理 (.env 読み込み)
│   ├── rag/
│   │   ├── indexer.py          ← ChromaDB へのドキュメント登録
│   │   └── retriever.py        ← ベクトル検索・事例取得
│   └── agents/
│       ├── preprocessing.py    ← Agent①: 自然言語 → SimulationSpec
│       ├── prompt_generation.py← Agent②: RAG 検索 + コンテキスト構築
│       ├── openfoam_gpt.py     ← Agent③: ケース生成・実行・自己修正
│       └── postprocessing.py   ← Agent④: 物理チェック・レポート生成
├── templates/                  ← Jinja2 テンプレート
│   ├── 0/                      (U, p, k, omega, nut)
│   ├── system/
│   │   ├── blockMeshDict/      (box_channel_2d, box_channel_3d,
│   │   │                        box_snappy, box_snappy_2d, ...)
│   │   ├── snappyHexMeshDict/
│   │   ├── fvSchemes.j2
│   │   ├── fvSolution.j2
│   │   └── controlDict.j2
│   └── constant/               (turbulenceProperties, transportProperties)
├── knowledge_base/             ← RAG 用ドキュメント・ChromaDB インデックス
├── generate_cylinder_stl.py    ← 3D テスト用円柱 STL 生成スクリプト
├── generate_cylinder_2d_stl.py ← 2D テスト用円柱 STL 生成スクリプト
├── output/                     ← 生成ケース出力先 (.gitignore 対象)
├── requirements.txt
└── pyproject.toml
```

---

## 環境

| 項目       | バージョン                                     |
|------------|------------------------------------------------|
| OS         | Ubuntu 24.04 on WSL2                           |
| Python     | 3.12+                                          |
| OpenFOAM   | v2512 (`/usr/lib/openfoam/openfoam2512`)       |
| LLM        | OpenAI gpt-4o または Anthropic Claude          |

---

## セットアップ

### 1. セットアップスクリプト

```bash
cd ~/openfoam-ai-agent
bash setup.sh
```

OpenFOAM・Python 依存パッケージを一括インストールします。

### 2. API キー設定

```bash
cp .env.example .env   # なければ手動で作成
nano .env
```

```dotenv
# OpenAI を使う場合
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o

# Anthropic Claude を使う場合
# ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxx
# LLM_PROVIDER=anthropic
# LLM_MODEL=claude-opus-4-5
```

### 3. 仮想環境の有効化

```bash
source venv/bin/activate
```

### 4. RAG インデックス構築（初回のみ）

```bash
python -m src.rag.indexer
```

---

## 使い方

### 基本実行

```bash
python -m src.main run "解析の説明"
```

### STL ファイルを使った外部流れ（snappyHexMesh）

```bash
# 3D 円柱 STL を生成してから実行
python generate_cylinder_stl.py
python -m src.main run \
  "円柱周りの定常外部流れ、流入速度5m/s" \
  --stl ./cylinder.stl

# 2D 円柱（カルマン渦）
python generate_cylinder_2d_stl.py
python -m src.main run \
  "円柱周りの2D非定常カルマン渦流れ、流入速度0.15m/s、層流" \
  --stl ./cylinder_2d.stl
```

### 出力先の指定

```bash
python -m src.main run "..." --output ./output/my_case
```

---

## テンプレートの境界条件

`case_type` と `is_snappy_2d` フラグに応じて以下のパッチが自動設定されます。

| パッチ     | channel_2d    | channel_3d    | snappy_2d       | external_snappy  |
|------------|---------------|---------------|-----------------|------------------|
| top/bottom | noSlip (U)    | noSlip (U)    | symmetryPlane   | symmetryPlane    |
| front/back | empty (2D)    | noSlip (U)    | empty (2D)      | symmetryPlane    |
| STL 物体   | —             | —             | noSlip + 壁関数 | noSlip + 壁関数  |

---

## 収束の鉄則（エージェントに組み込み済み）

- `simpleFoam` (SIMPLEC): noSlip 壁が 1 面以上必要。壁なし純外部流れは収束しない
- `pimpleFoam`: 外部流れには `potentialFoam` で速度場を初期化
- 乱流モデル: 3D → kOmegaSST、2D → laminar（2D 乱流は非物理的）
- 定常で Re > 10,000 → pimpleFoam 推奨の警告を表示

---

## ParaView での可視化（Windows + WSLg）

生成後、ターミナルに表示される UNC パスをエクスプローラーに貼り付けて `.foam` ファイルを開きます。

```
\\wsl.localhost\Ubuntu-24.04\home\akari\openfoam-ai-agent\output\...\case_name.foam
```

カルマン渦を確認する場合：
1. `Apply` → `Slice` (z 方向) → `Surface`
2. 速度ベクトル: `Filters → Glyph → Arrow`
3. 流線: `Filters → Stream Tracer`
4. 時間を進める: ▶ ボタン

---

## ライセンス

MIT

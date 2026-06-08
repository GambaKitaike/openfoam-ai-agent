# OpenFOAM AI Agent

自然言語の指示から OpenFOAM 解析ケースを自動生成・実行・可視化まで一括で行う AI エージェントです。

**目標:** OpenFOAM 版 Cursor — ユーザーと対話しながらケースを育て、実行・エラー修正を繰り返すエージェント（現在は検証段階のプロトタイプ）。

---

## 現状の機能（2025-06）

| レイヤ | 内容 |
|--------|------|
| **Agent①** | 自然言語 → `SimulationSpec`（phenomenon タグ、Re 等）。未指定パラメータは**対話で確認** |
| **Agent②** | ケース単位 RAG（477 チュートリアル + **intent メタデータ**）。参照ケース選定後 **Phase A: 典型条件を提案** |
| **Agent③** | 参照ケース適用 or Jinja テンプレート → blockMesh / ソルバー実行 → **自己修正**（最大 3 回） |
| **Agent④** | 残差・物理妥当性チェック → レポート & ParaView パス |

### ケース単位 RAG

- 1 チュートリアル = 1 ドキュメント（ChromaDB `openfoam_cases`）
- LLM 生成 **intent**（`title_ja`, `summary_ja`, `phenomenon`, `suitable_for_ja` …）を `build-index` 時に付与
- ハードフィルタ（solver / 定常 / 2D / phenomenon / mesh_prebuilt 等）+ ベクトル検索
- **mesh_prebuilt** ケース（例: `airFoil2D`）は polyMesh をコピーし blockMesh をスキップ

### 対話型パラメータ確認

1. **Agent① 後** — 流速・Re・乱流モデル等が未指定なら質問
2. **Agent② 後（Phase A）** — 参照ケースから典型条件（U, nu, Re, 乱流）を抽出し「参照ケースに合わせますか？」

```bash
# 対話 ON（TTY ではデフォルト）
python -m src.main run "2D翼周りの定常流れ simpleFoam" -o ./output/airfoil

# 非対話（CI / バッチ）
python -m src.main run "..." --no-interactive -o ./output/test
```

---

## アーキテクチャ

```
自然言語
    ↓
[Agent①] Preprocessing     → SimulationSpec + 条件確認
    ↓
[Agent②] Case Selection   → ChromaDB + intent RAG → EnrichedContext
    ↓                        └ Phase A: 参照ケース典型条件の確認
[Agent③] OpenFOAM GPT      → ケース生成・実行・自己修正
    ↓
[Agent④] Post-processing   → レポート & ParaView 案内
```

---

## クイックスタート

```bash
cd ~/openfoam-ai-agent
bash setup.sh                 # 初回
source venv/bin/activate      # 必須（python3 単体では typer 等が無い）

cp .env.example .env          # OPENAI_API_KEY を設定

# RAG インデックス（初回: intent LLM 生成あり / 2 回目以降はキャッシュ）
python -m src.main build-index --no-web

# 開発時: enrich スキップ（Chroma のみ更新）
python -m src.main build-index --no-web --skip-enrich

# 実行
python -m src.main run "2D円柱周りのカルマン渦 Re=100 層流" -o ./output/karman
python -m src.main run "2D翼周りの定常流れ simpleFoam" -o ./output/airfoil
```

---

## プロジェクト構成

```
openfoam-ai-agent/
├── src/
│   ├── main.py                      CLI
│   ├── models.py                    SimulationSpec, EnrichedContext
│   ├── orchestrator.py              4-Agent パイプライン
│   ├── case_applier.py              参照ケース適用 + mesh_prebuilt コピー
│   ├── rag/
│   │   ├── case_catalog.py          チュートリアル発見・メタデータ
│   │   ├── case_intent.py           intent スキーマ
│   │   ├── case_intent_enricher.py  LLM intent 生成 + キャッシュ
│   │   ├── reference_case_params.py Phase A: 典型条件抽出
│   │   ├── case_selector.py         ハードフィルタ + ベクトル検索
│   │   ├── indexer.py               build-index
│   │   └── retriever.py
│   └── agents/
│       ├── preprocessing.py         Agent①
│       ├── spec_clarification.py    対話確認（Agent① / Phase A）
│       ├── prompt_generation.py     Agent②
│       ├── openfoam_gpt.py          Agent③
│       └── postprocessing.py        Agent④
├── templates/                       Jinja2 フォールバック
├── knowledge_base/
│   ├── case_intents/                intent JSON キャッシュ（build-index で生成）
│   └── chroma_db/                   ベクトル DB（.gitignore）
├── tests/
└── requirements.txt
```

---

## 環境

| 項目 | バージョン |
|------|------------|
| OS | Ubuntu 24.04 on WSL2 推奨 |
| Python | 3.12+ |
| OpenFOAM | v2512 (`/usr/lib/openfoam/openfoam2512`) |
| LLM | OpenAI gpt-4o（`.env` で設定） |

---

## build-index オプション

| フラグ | 説明 |
|--------|------|
| `--no-web` | Web スクレイピングをスキップ（現行はケース RAG のみ使用） |
| `--skip-enrich` | LLM intent 生成をスキップ（既存キャッシュがあれば読込） |
| `--enrich-only` | Chroma 化せず intent 生成のみ |
| `--force-enrich` | キャッシュ無視で intent 再生成 |

---

## 対応 case_type（フォールバック用）

| `case_type` | 説明 |
|-------------|------|
| `channel_2d` / `channel_3d` | 壁あり内部流れ |
| `cylinder_2d_ogrid` | O-グリッド円柱（カルマン渦、STL 不要） |
| `snappy_2d` / `external_snappy` | STL + snappyHexMesh |

RAG で参照ケースが選ばれた場合は、チュートリアルファイルを優先適用します。

---

## ParaView（WSL → Windows）

実行後に表示される UNC パスで `.foam` を開きます。

```
\\wsl.localhost\Ubuntu-24.04\home\<user>\openfoam-ai-agent\output\<case>\<case>.foam
```

---

## ロードマップ

- **Phase A（実装済）** — RAG 選定後に参照ケースの典型 Re/U/乱流を対話反映
- **Phase B** — 1 ケースを会話で incremental に編集（Cursor 的 diff）
- **Phase C** — ログ駆動の自律修正ループ強化

---

## テスト

```bash
source venv/bin/activate
pytest tests/ -q
```

---

## ライセンス

MIT

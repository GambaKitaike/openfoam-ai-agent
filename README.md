# OpenFOAM AI Agent

自然言語の指示から OpenFOAM 解析ケースを自動生成・実行・可視化まで一括で行う AI エージェントです。

**目標:** OpenFOAM 版 Cursor — ユーザーと対話しながらケースを育て、実行・エラー修正を繰り返すエージェント（現在は検証段階のプロトタイプ）。

---

## 現状の機能（2026-06）

| レイヤ | 内容 |
|--------|------|
| **Agent①** | 自然言語 → draft `SimulationSpec` |
| **Agent① ↔ Agent②** | 内部ループ（充足 → `review_spec` → 修正、最大 3 ラウンド）。`--no-interactive` でも Agent 間レビューは動作 |
| **Agent②** | ケース単位 RAG + **RequirementProfile** + **match_score**（fast path / staged ルート判定） |
| **Agent③** | **fast path**（高一致参照ケースコピー）または **段階的生成**（Policy + Python builders + mesh）→ 実行・自己修正 |
| **Agent④** | 残差・物理妥当性チェック → レポート & ParaView パス |

### 主な変更点（staged case builder 移行）

- **Jinja テンプレート (`templates/*.j2`) を廃止** — OpenFOAM 辞書は `src/case_builder/builders.py` と Python メッシュ生成器で決定的に生成
- **Agent② レビューループ** — Re と乱流モデルの矛盾などを Agent 間で検出・修正（層流指定時は U を下げる選択肢も提示）
- **カルマン渦 (Re≈100)** — O-グリッド + `pimpleFoam` + 後流摂動で非定常渦列を確認済み
- **`test-agents` CLI** — ソルバー実行なしで Agent 間通信を可視化

### ケース単位 RAG

- 1 チュートリアル = 1 ドキュメント（ChromaDB `openfoam_cases`）
- LLM 生成 **intent**（`title_ja`, `summary_ja`, `phenomenon`, `suitable_for_ja` …）を `build-index` 時に付与
- ハードフィルタ（solver / 定常 / 2D / phenomenon / mesh_prebuilt 等）+ ベクトル検索
- **match_score ≥ 0.8** → fast path（`CaseApplier` で参照ケースをコピー＋スカラー置換）
- 一致度が低い / 参照なし → **staged path**（段階的生成パイプライン）
- **mesh_prebuilt** ケース（例: `airFoil2D`）は polyMesh をコピーし blockMesh をスキップ

### 対話・非対話

| モード | 挙動 |
|--------|------|
| **対話（TTY デフォルト）** | Agent② の指摘をユーザーにも確認。参照ケース典型値（Phase A）も確認可 |
| **`--no-interactive`** | Agent② が自動修正（説明文で明示した値は `user_locked` で維持） |

```bash
# 対話 ON
python -m src.main run "2D翼周りの定常流れ simpleFoam" -o ./output/airfoil

# 非対話（CI / バッチ）
python -m src.main run "2D円柱 Re=100 カルマン渦" -o ./output/karman --no-interactive

# Agent 間通信テスト（ソルバー実行なし）
python -m src.main test-agents --all --offline --no-interactive
python -m src.main test-agents -s channel_conflict --offline
```

---

## アーキテクチャ

```
自然言語
    ↓
[Agent①] extract                 → draft SimulationSpec
    ↓
[Agent① ↔ Agent②] 内部ループ
    Agent② RequirementProfile     → 未充足項目の補完
    Agent② review_spec            → 物理整合性レビュー（Re/乱流/ソルバー等）
    ↓
[Agent②] Reference Match         → match_score → fast path or staged
    ↓                            └ Phase A: 参照ケース典型条件の確認（対話時）
[Agent③] OpenFOAM GPT
    経路 A: CaseApplier            → 参照ケース丸ごとコピー
    経路 B: CaseBuildPipeline      → transport → turbulence → controlDict
                                     → blockMesh → 0/ → fv* → setFields
    ↓
[Agent④] Post-processing         → レポート & ParaView 案内
```

### 段階的生成（経路 B）の構成

| モジュール | 役割 |
|-----------|------|
| `case_builder/policy.py` | ソルバー選択、Re 整合、時間刻み（カルマンは Strouhal ベース） |
| `case_builder/builders.py` | controlDict, fvSchemes, fvSolution, 0/ 等の決定的生成 |
| `case_builder/mesh_generators.py` | 汎用 box channel メッシュ |
| `mesh/cylinder_2d_ogrid.py` | カルマン渦用 O-グリッド |
| `case_builder/pipeline.py` | 上記を順番に実行・検証 |

---

## クイックスタート

```bash
cd ~/openfoam-ai-agent
bash setup.sh                 # 初回
source venv/bin/activate      # 必須

cp .env.example .env          # OPENAI_API_KEY を設定

# RAG インデックス（初回: intent LLM 生成あり / 2 回目以降はキャッシュ）
python -m src.main build-index --no-web

# 実行例
python -m src.main run "2D円柱周りのカルマン渦 Re=100 層流" -o ./output/karman --no-interactive
python -m src.main run "2D翼周りの定常流れ simpleFoam" -o ./output/airfoil

# Agent 通信テスト
python -m src.main test-agents --all --offline
```

---

## プロジェクト構成

```
openfoam-ai-agent/
├── src/
│   ├── main.py                      CLI（run / build-index / test-agents / check）
│   ├── models.py                    SimulationSpec, RequirementProfile, ReferenceMatch
│   ├── orchestrator.py              4-Agent パイプライン
│   ├── agent_dialogue.py            Agent 間通信トレース（test-agents 用）
│   ├── case_builder/                段階的生成（pipeline, builders, policy, mesh）
│   ├── case_applier.py              fast path: 参照ケース適用
│   ├── rag/
│   │   ├── requirement_profile.py   現象別要件 + review_spec
│   │   ├── match_score.py           fast path 閾値判定
│   │   ├── reference_case_params.py 参照ケース典型条件抽出
│   │   ├── case_selector.py         ハードフィルタ + ベクトル検索
│   │   └── retriever.py             Agent② RAG 入口
│   └── agents/
│       ├── preprocessing.py         Agent①
│       ├── spec_clarification.py    ヒアリング + Agent② レビューループ
│       ├── prompt_generation.py     Agent②
│       ├── openfoam_gpt.py          Agent③
│       └── postprocessing.py        Agent④
├── knowledge_base/
│   ├── case_intents/                intent JSON キャッシュ
│   └── chroma_db/                   ベクトル DB（.gitignore）
└── tests/                           pytest（builders, policy, pipeline, agent_dialogue 等）
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

## CLI コマンド

| コマンド | 説明 |
|---------|------|
| `run` | フルパイプライン（生成 → メッシュ → ソルバー → レポート） |
| `build-index` | チュートリアル RAG インデックス構築 |
| `test-agents` | Agent 間通信テスト（`--offline` で LLM なし、`--all` で全シナリオ） |
| `check` | 既存ケースの AI レビュー |

### build-index オプション

| フラグ | 説明 |
|--------|------|
| `--no-web` | Web スクレイピングをスキップ |
| `--skip-enrich` | LLM intent 生成をスキップ |
| `--enrich-only` | Chroma 化せず intent 生成のみ |
| `--force-enrich` | キャッシュ無視で intent 再生成 |

---

## 対応 case_type（staged フォールバック）

| `case_type` | 説明 |
|-------------|------|
| `channel_2d` / `channel_3d` | 壁あり内部流れ |
| `cylinder_2d_ogrid` | O-グリッド円柱（カルマン渦、STL 不要） |
| `snappy_2d` / `external_snappy` | STL + snappyHexMesh |

RAG で参照ケースが選ばれた場合は fast path または参照ファイル adapt を優先します。

---

## ParaView（WSL → Windows）

ケースディレクトリに `.foam` を置いて開きます。

```bash
touch output/karman/pimpleFoam_cylinder_2d_ogrid/pimpleFoam_cylinder_2d_ogrid.foam
paraview output/karman/pimpleFoam_cylinder_2d_ogrid/pimpleFoam_cylinder_2d_ogrid.foam
```

Windows 側からは UNC パスでも可:

```
\\wsl.localhost\Ubuntu-24.04\home\<user>\openfoam-ai-agent\output\<case>\<case>.foam
```

---

## ロードマップ

- **実装済** — staged case builder（Jinja 廃止）、Agent①↔Agent② レビューループ、カルマン O-grid、`test-agents`
- **次** — Agent② がチュートリアル典型値をプロファイル／レビューの主入力に（`similar_case_ids` 活用）
- **次** — Agent③ → Agent② `get_file_guidance()`（ファイル単位 syntax 問い合わせ）
- **将来** — 1 ケースを会話で incremental に編集（Cursor 的 diff）

---

## テスト

```bash
source venv/bin/activate
pytest tests/ -q
```

---

## ライセンス

MIT

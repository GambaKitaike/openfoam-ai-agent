# openfoam-ai-agent v2 設計書 — 対話型エージェント化（OpenFOAM版Cursor）

- **版**: Draft v0.1（2026-06-12）
- **対象リポジトリ**: github.com/GambaKitaike/openfoam-ai-agent
- **実装担当**: Cursor Composer（本設計書をコンテキストとして実装を依頼する）
- **設計担当**: Gamba + Claude

---

## 1. 背景と目的

### 1.1 現状（v1）

v1 は「自然言語 → ケース生成 → 実行 → レポート」を1コマンドで行う**単発パイプライン**である。

```
python -m src.main run "..." --stl ... --output ...
```

```
[Agent①] Preprocessing      自然言語 → SimulationSpec
[Agent②] RAG + Prompt Gen   ChromaDB 検索 → EnrichedContext
[Agent③] OpenFOAM GPT       ケース生成 → 実行 → 自己修正ループ(最大3回)
[Agent④] Post-processing    残差・物理妥当性チェック → レポート
```

### 1.2 v1 の限界

1. **対話できない**。実行後に「メッシュをもう少し細かく」「Re数を変えて再実行」と言えない。やり直しは毎回ゼロから。
2. **既存ケースを扱えない**。生成専用であり、手元の既存ケースの読解・修正ができない。
3. **修正の主導権がユーザーにない**。自己修正ループはブラックボックスで、何をどう直したか追えない。
4. **結果の要約が弱い**。残差チェック止まりで、Cd/Cl・プローブ値・画像つきレポートまで出ない。

### 1.3 v2 の目的

**「ケースディレクトリをワークスペースとして開き、チャットで指示すると、エージェントがファイルの読み書き・ソルバー実行・結果要約をツールとして行う」** 対話型エージェントにする。Cursor のメンタルモデルをそのまま借りる：

| Cursor | openfoam-ai-agent v2 |
|---|---|
| プロジェクトフォルダ | ケースディレクトリ（or 親ワークスペース） |
| ソースコード | 0/, system/, constant/ の各 dict |
| ビルド・テスト実行 | blockMesh / snappyHexMesh / ソルバー実行 |
| エラー出力を見て修正 | log.* を読んで dict を修正 |
| Composer（エージェント） | AgentCore（ツール使用ループ） |

### 1.4 v2 で **やらないこと**（スコープ外）

- GUI エディタ（VSCode 拡張、独自 IDE）→ 当面 CLI チャット、後に Streamlit
- メッシュの GUI 編集、CAD 機能
- OpenFOAM 以外のソルバー対応
- マルチユーザー・サーバー化

---

## 2. 全体アーキテクチャ

### 2.1 基本方針：パイプライン → エージェントループ

v1 の固定パイプライン（①→②→③→④）を解体し、**LLM がツールを選んで呼ぶ ReAct 型ループ**に変える。v1 の各エージェントはツールまたはツール内部実装として再利用する。

```
ユーザー ──チャット──▶ AgentCore（ループ）
                         │ ① LLM にコンテキスト + ツール定義を渡す
                         │ ② LLM がツール呼び出し or 最終応答を返す
                         │ ③ ツール実行 → 結果を履歴に追加 → ①へ
                         ▼
   ┌─────────────────────────────────────────────┐
   │ Tools                                       │
   │  case_scaffold   : v1 パイプラインのケース生成部 │
   │  read_file / list_files                     │
   │  edit_file       : str_replace 方式          │
   │  run_openfoam    : blockMesh/solver 等の実行  │
   │  read_log        : ログのtail/エラー抽出       │
   │  rag_search      : ChromaDB 検索（v1 Agent②） │
   │  summarize_results : 残差/力係数/画像レポート   │
   └─────────────────────────────────────────────┘
```

### 2.2 レイヤ構成

```
┌────────────────────────────────────┐
│ UI 層         cli_chat.py (Phase1) / Streamlit (Phase3) │
├────────────────────────────────────┤
│ Agent 層      AgentCore（ループ・履歴・コンテキスト管理）   │
├────────────────────────────────────┤
│ Tool 層       tools/*.py（純粋関数に近い形で実装）        │
├────────────────────────────────────┤
│ Domain 層     v1 資産: runner, monitor, rag, templates,  │
│               models(SimulationSpec), 収束の鉄則          │
└────────────────────────────────────┘
```

**重要**: Tool 層は LLM なしでも単体テストできる純粋な Python 関数として実装する。LLM 依存は Agent 層に閉じ込める。

### 2.3 ディレクトリ構成（v2 目標）

```
openfoam-ai-agent/
├── src/
│   ├── main.py              ← CLI: `run`(v1互換) と `chat`(v2) の2コマンド
│   ├── agent/
│   │   ├── core.py          ← AgentCore: ツール使用ループ
│   │   ├── session.py       ← SessionState の保存/復元
│   │   └── prompts.py       ← システムプロンプト（収束の鉄則を含む）
│   ├── tools/
│   │   ├── registry.py      ← ツール定義(JSON Schema)と dispatch
│   │   ├── fs_tools.py      ← read_file, list_files, edit_file
│   │   ├── foam_tools.py    ← run_openfoam, read_log, foam_dict_check
│   │   ├── case_tools.py    ← case_scaffold（v1パイプライン流用）
│   │   ├── rag_tools.py     ← rag_search
│   │   └── post_tools.py    ← summarize_results, plot_residuals
│   ├── runner.py / monitor.py / llm_client.py / config.py  (v1から流用)
│   ├── rag/                 (v1から流用)
│   └── agents/              (v1から流用。case_tools/post_tools の内部実装として呼ぶ)
├── templates/               (v1のまま)
├── knowledge_base/          (v1のまま)
└── tests/                   ← Tool 層の単体テスト（pytest）
```

### 2.4 v1 資産の扱い（削除・リファクタ方針）

**本設計書に登場しない v1 ファイルも含め、v1 のコードは削除しない。** 設計書は「追加と再配置」を記述しており、削除指示ではない。

- `runner.py` / `monitor.py` / `llm_client.py` / `config.py` / `models.py` / `rag/` / `templates/` / `knowledge_base/` / STL 生成スクリプト: そのまま流用
- `agents/`（①〜④）: 削除しない。case_tools / rag_tools / post_tools の内部実装として呼ばれる側に回る
- `main.py` の `run` コマンド: v1 互換として温存。Phase 1〜3 の間、v1 は常に動く状態を維持する（唯一の動くデモを壊さない）

**実装者への注意**: 既存モジュールの「ついでのリファクタ」「未使用に見えるコードの削除」を行わないこと。v1 のクリーンアップ（`run` 廃止・Agent③ 実行部の削除等）は Phase 1 完了後に独立タスクとして別途判断する。

---

## 3. データモデル

### 3.1 SessionState

```python
@dataclass
class SessionState:
    workspace: Path              # ケースディレクトリ（必ずこの配下のみ操作可）
    spec: SimulationSpec | None  # 既知の解析仕様（生成時 or 推定時に埋まる）
    history: list[Message]       # チャット履歴（LLM API 形式）
    run_records: list[RunRecord] # 実行履歴
    created_at / updated_at: datetime
```

`workspace/.ofagent/session.json` に保存し、`chat --resume` で復元できる。

### 3.2 RunRecord

```python
@dataclass
class RunRecord:
    command: str           # 例: "pimpleFoam"
    log_path: Path         # 例: log.pimpleFoam
    exit_code: int
    started_at / finished_at: datetime
    summary: str           # 終了時刻・最終残差など1〜3行
```

### 3.3 ToolResult（全ツール共通の戻り値）

```python
@dataclass
class ToolResult:
    ok: bool
    content: str           # LLM に返す本文（長大ログはここで必ず要約/truncate）
    data: dict | None      # UI 表示用の構造化データ（画像パス等）
```

**設計判断**: ツールが LLM に返す `content` は**最大でも数千トークンに制限**する。OpenFOAM のログは平気で数万行になるため、`read_log` は「エラー抽出」「末尾N行」「残差の時系列サンプリング」のいずれかのモードでしか返さない。生ログ全文を LLM に流すことを構造的に禁止する。

---

## 4. ツール仕様

すべてのツールは `workspace` 外のパスを拒否する（パストラバーサル対策、`Path.resolve()` で検証）。

### 4.1 ファイル系（fs_tools.py）

| ツール | 引数 | 動作 |
|---|---|---|
| `list_files` | path?, depth? | workspace 配下のツリー表示。`processor*`, `postProcessing` の中身、時刻ディレクトリは件数のみに省略 |
| `read_file` | path, line_range? | テキストファイル読み取り。バイナリ(.stl等)はサイズとヘッダ情報のみ |
| `edit_file` | path, old_str, new_str | **str_replace 方式**。old_str が一意に1回出現しない場合はエラーを返し、LLM に再考させる |
| `write_file` | path, content | 新規作成。既存ファイルは拒否（上書きは edit_file を強制） |

**str_replace を採用する理由**: 全文書き換え方式は (a) トークン消費が大きい (b) 変更箇所が差分で追えない (c) 意図しない欠落が起きる。Cursor/Claude Code が実証済みの方式に合わせる。

**edit_file の追加動作**: OpenFOAM dict ファイル（0/, system/, constant/ 配下）を編集した場合、編集後に自動で構文チェック（4.2 `foam_dict_check`）を走らせ、結果を ToolResult に含める。構文エラーの混入をその場で検知させる。

### 4.2 OpenFOAM 実行系（foam_tools.py）

| ツール | 引数 | 動作 |
|---|---|---|
| `run_openfoam` | command, args?, timeout? | **許可リスト方式**で OpenFOAM コマンドを実行。stdout/stderr は `log.<command>` に保存し、ToolResult には要約のみ返す |
| `read_log` | log_path, mode | mode = `errors`（FATAL/エラー周辺の抽出）/ `tail`（末尾N行）/ `residuals`（残差の時系列を間引いて返す） |
| `foam_dict_check` | path | `foamDictionary <path>` を実行し dict の構文妥当性を確認 |

**コマンド許可リスト（初期値）**:
```python
ALLOWED_COMMANDS = {
    "blockMesh", "snappyHexMesh", "surfaceFeatureExtract",
    "checkMesh", "potentialFoam", "simpleFoam", "pimpleFoam",
    "foamToVTK", "foamDictionary", "postProcess", "decomposePar",
    "reconstructPar",
}
```
任意シェル実行ツール（`bash` 相当）は**意図的に提供しない**。LLM 製エージェントに生シェルを渡すのは事故のもとで、OpenFOAM ワークフローは上記で完結する。不足が出たらリストに追加する運用。

**実行の同期/非同期**: Phase 1 では同期実行（timeout デフォルト 30 分）。長時間ジョブの非同期化（バックグラウンド実行 + `check_run` ツール）は Phase 2 の検討項目とし、最初から作らない。

### 4.3 ケース生成系（case_tools.py）

| ツール | 引数 | 動作 |
|---|---|---|
| `case_scaffold` | description, stl_path?, case_dir? | v1 の Agent①＋②＋③のケース生成部を呼び、テンプレートからケース一式を生成する。**実行はしない**（実行は LLM が `run_openfoam` で明示的に行う） |

**v1 からの最重要変更**: v1 の Agent③ は「生成→実行→自己修正」を内部で完結させていた。v2 ではこれを分解し、**実行と修正の判断をエージェントループ側（=会話に見える場所）に引き上げる**。これにより：

- ユーザーは「メッシュ生成まででいったん止めて」と言える
- 自己修正の各ステップ（エラー → 修正内容）がチャット上で可視化される
- 修正回数の上限はハードコードの3回ではなく、ループのターン上限＋ユーザー判断になる

v1 の自己修正ループのコードは捨てるのではなく、システムプロンプトに「ソルバーが FATAL で落ちたら read_log(errors) → 原因特定 → edit_file → 再実行」という**手順知識として移植**する。

### 4.4 RAG 系（rag_tools.py）

| ツール | 引数 | 動作 |
|---|---|---|
| `rag_search` | query, scope="case"\|"file", filters?, top_k=3〜5 | 2コレクション構成の ChromaDB を検索（詳細は §11） |

v1 では RAG は Agent② として必ず1回実行されたが、v2 では**LLM が必要と判断したときに呼ぶツール**になる。「fvSchemes の div スキームの書き方が怪しい」と思ったときに引く辞書、という位置づけ。検索条件・コレクション設計・知識ベース拡張方針は §11 を正とする。

### 4.5 ポスト処理系（post_tools.py）— Phase 2

| ツール | 引数 | 動作 |
|---|---|---|
| `plot_residuals` | log_path | 残差を matplotlib で PNG 化し、パスを返す |
| `compute_forces` | patch_name | controlDict に forceCoeffs functionObject を挿入（edit_file 経由）→ postProcess 実行 → Cd/Cl 時系列の要約 |
| `summarize_results` | — | 実行履歴・残差・力係数・checkMesh 結果を統合した `report.md` を生成（v1 Agent④ の発展形） |
| `render_snapshot` | field, time? | PyVista で断面コンター画像を PNG 出力（オフスクリーンレンダリング） |

`render_snapshot` は WSL2 でのヘッドレスレンダリング依存（OSMesa等）があり不確実性が高いため、**Phase 2 の中でも最後**に回す。失敗しても report.md は画像なしで成立する設計にする。

---

## 5. AgentCore（エージェントループ）

### 5.1 ループ仕様

```python
def run_turn(user_input: str, state: SessionState) -> str:
    state.history.append(user_message(user_input))
    for _ in range(MAX_STEPS):          # 1ターンあたりのツール呼び出し上限（初期値 15）
        resp = llm.chat(
            system=build_system_prompt(state),
            messages=state.history,
            tools=registry.schemas(),
        )
        if resp.has_tool_calls:
            for call in resp.tool_calls:
                result = registry.dispatch(call, state)   # 確認ゲートを含む
                state.history.append(tool_result_message(call, result))
        else:
            state.history.append(assistant_message(resp.text))
            return resp.text
    return "(ステップ上限に達しました。状況を整理します…)" + 強制要約
```

### 5.2 確認ゲート（human-in-the-loop）

破壊的・高コストな操作の前にユーザー確認を挟む。

| 操作 | 確認 |
|---|---|
| read系（list/read/log/rag） | 不要（自動実行） |
| edit_file / write_file | **diff を表示して y/n**（`--yolo` フラグで省略可） |
| run_openfoam（メッシュ・ソルバー） | 実行コマンドを表示して y/n |
| case_scaffold | 生成される spec の要約を表示して y/n |

Cursor の「Accept / Reject」に相当する。Phase 1 ではターミナル上の y/n プロンプトで十分。

### 5.3 システムプロンプト

`src/agent/prompts.py` に集約。含めるもの：

1. 役割定義（OpenFOAM ケース構築・実行・デバッグを行うエージェント）
2. **収束の鉄則**（v1 README 記載のもの＋今後の知見追加）
3. 標準ワークフロー知識：
   - 新規ケース: case_scaffold → checkMesh → (potentialFoam) → ソルバー → summarize
   - エラー時: read_log(errors) → 原因の仮説 → rag_search（必要時）→ edit_file → 再実行
4. 行動規範：「編集は最小差分」「実行前に何をするか1行で宣言」「不確かな物理設定はユーザーに確認」
5. ワークスペース情報（list_files の初期スナップショット、spec があればその内容）

**知識の置き場所の判断基準**: システムプロンプトには「ほぼ毎ターン真である知識」のみを置く（収束の鉄則・手順の骨格）。状況依存の知識（個別エラーパターン等）は RAG 側（§11.4）に置き、必要時のみ取得する。システムプロンプトはエージェントループ内で LLM 呼び出しごとに送信されるため、長さは MAX_STEPS 倍で課金・劣化に効く。**失敗事例等をシステムプロンプトへ自動追記する機構は実装しないこと**。DB 内で頻出するパターンを鉄則へ昇格させる判断は人間が手動で行う。

### 5.4 コンテキスト管理

- 履歴が長くなったら（目安 80K トークン）古いツール結果から要約圧縮する。`run_records` と spec は常に system prompt 側に保持されるため、ログ詳細は捨ててよい。
- Phase 1 では単純な「古いツール結果の truncate」で開始し、要約圧縮は問題が出てから実装する。

---

## 6. UI

### Phase 1: CLI チャット（REPL）

```
$ python -m src.main chat --workspace ./output/my_case
[ofagent] ワークスペース: ./output/my_case (既存ケースを検出: pimpleFoam, 最終時刻 2.5s)
> メッシュをもう一段細かくして再実行して
[tool] read_file system/blockMeshDict
[tool] edit_file system/blockMeshDict   ← diff 表示 → 承認 y/n
[tool] run_openfoam blockMesh           ← 承認 y/n
...
```

- `rich` ライブラリで diff・ツール呼び出し・スピナーを表示
- スラッシュコマンド: `/status`（spec・実行履歴）, `/yolo`（確認省略切替）, `/quit`
- `chat` 開始時に workspace を走査し、既存ケースなら自動でソルバー種別・最終時刻を認識（`controlDict` と時刻ディレクトリから判定）

### Phase 3: Streamlit チャット UI

- 左: ファイルツリー＋ファイルビューア、右: チャット
- diff の Accept/Reject ボタン、残差グラフ・スナップショット画像のインライン表示
- rag_project で Streamlit 実績があるため技術リスクは低い。ただし**CLI が完成してから**着手する（Agent/Tool 層が UI 非依存なら載せ替えは薄い作業）

---

## 7. フェーズ計画と受け入れ基準

### Phase 1: 対話型エージェント MVP（最優先）

実装: AgentCore / fs_tools / foam_tools / case_tools / rag_tools / cli_chat / SessionState

**受け入れ基準（このシナリオが通ること）**:

1. `chat` で空ディレクトリから「円柱周り2Dカルマン渦、流入0.15m/s、層流」→ ケース生成 → 実行 → 完了報告まで対話で到達できる
2. 続けて「流入速度を0.3m/sに変えて再実行」→ `0/U` と必要なら `controlDict` の差分編集 → 再実行ができる（ゼロから再生成しない）
3. わざと壊した fvSolution を持つ既存ケースを開き、「実行して、落ちたら直して」でエラー特定→修正→完走できる
4. すべての編集・実行が確認ゲートを通り、diff が表示される
5. `tests/` に fs_tools / foam_tools の単体テストがあり pytest が通る（run_openfoam はモック）

### Phase 2: 結果要約の強化

実装: post_tools（plot_residuals → compute_forces → summarize_results → render_snapshot の順）

**受け入れ基準**: カルマン渦ケース完走後に「結果をまとめて」→ 残差グラフPNG＋Cd/Cl統計＋checkMesh要約入りの `report.md` が生成される。

### Phase 3: Streamlit UI

**受け入れ基準**: Phase 1 のシナリオ1〜3がブラウザ上で実行でき、diff 承認がボタンで行える。

### マイルストーン外（アイデア駐車場）

非同期実行＋ジョブ監視 / パラメトリックスタディ（Re数スイープ）/ メッシュ収束性スタディ自動化 / 過去ケースの RAG 自動取り込み

---

## 8. リスクと設計上の判断（記録）

| リスク | 対応 |
|---|---|
| ログが巨大で LLM コンテキストを食いつぶす | read_log のモード制限・ToolResult の content 上限で構造的に防ぐ |
| LLM が dict を壊す編集をする | edit_file 後の foamDictionary 自動チェック＋確認ゲート |
| ソルバー長時間実行で対話が固まる | Phase 1 は timeout で割り切り。非同期化は需要が確認できてから |
| 「Cursor並み」への期待過多 | 受け入れ基準を3シナリオに限定。汎用性よりシナリオ完遂を優先 |
| WSL2 のヘッドレス描画 | render_snapshot を Phase 2 末尾に隔離。失敗しても他が成立する依存方向にする |
| v1 互換を壊す | `run` コマンドは v1 のまま温存。`chat` を別エントリで追加 |

## 9. 決定事項（2026-06-12 確定）

1. **LLM 既定モデル**: OpenAI `gpt-4o` を既定とする。コスト削減用に `gpt-4o-mini` も `.env` で切替可能にする（AgentCore のツール使用ループは 4o、case_scaffold 内の spec 変換など単発の構造化出力は mini、という使い分けを想定。llm_client は Anthropic 両対応のまま維持）
2. **ワークスペースの粒度**: 1ケース = 1ワークスペース。複数ケース横断はアイデア駐車場行き
3. **確認ゲートのデフォルト**: 安全側（確認あり）。`--yolo` はオプトイン

## 10. 用語の整理：エージェントとツール

v2 において「エージェント」は **AgentCore の1体のみ**。エージェントとは「次に何をするかを自分で判断しループを回す主体」を指し、v1 の Agent①②④は判断主体ではなくなるため、以後「ツール」と呼ぶ。

| v1 呼称 | v2 での位置づけ | 内部での LLM 使用 |
|---|---|---|
| Agent① Preprocessing | `case_scaffold` の内部処理（自然言語→spec） | あり（単発の構造化出力。ループなし） |
| Agent② RAG | `rag_search` ツール | なし（ベクトル検索のみ） |
| Agent③ OpenFOAM GPT | 解体（生成→case_scaffold、実行・自己修正→AgentCore） | — |
| Agent④ Post-processing | `summarize_results` ツール | あり（レポート文生成） |

**実装者への注意**: 「4エージェント構成を維持したまま対話機能を追加する」実装は本設計の意図に反する。判断ループは AgentCore に一元化すること。

---

## 11. RAG 設計（v2）

### 11.1 検索条件の方針転換

v1 は検索結果のチュートリアルをほぼそのまま流用していたため、誤マッチが生成物全体を汚染する＝precision 全振りの厳格な条件が必要だった。v2 では検索結果は「AgentCore が読んで取捨選択する参考資料」であり、採用内容は foamDictionary チェック・実行・ログ確認の検証ループを通る。誤マッチのコストが構造的に低いため、**recall 寄りに緩める**：

- **ハードフィルタ（where 句）は間違えると致命的な軸のみ**: `solver`, `steady_or_transient`,（該当時）`physics` の一部
- その他はベクトル類似度に任せ top_k=3〜5 で返し、取捨選択は AgentCore が行う

### 11.2 2コレクション構成

| コレクション | 粒度 | 用途 | 主なメタデータ |
|---|---|---|---|
| `cases` | 1チュートリアル = 1ドキュメント | case_scaffold 時の類似事例参照 | §11.3 のスキーマ全体 |
| `files` | 1 dict ファイル = 1ドキュメント | ループ中のピンポイント参照（「fvSchemes の div スキーム実例」等） | case_id, rel_path, solver |

`files` の埋め込みテキストはファイル内容そのもの（＋ケース要約1行をヘッダ付与）。`rag_search` の `scope` 引数で切り替える。

### 11.3 ケースメタデータ スキーマ v2

設計原則: **「埋め込み対象の自由文」と「where 句で絞る構造化フィールド」を分離する**。構造化フィールドは Allrun / dict ファイルの**機械的パースで抽出**し（LLM 不要＝誤抽出しない）、LLM 抽出は自由文要約のみに使う。

```json
{
  "case_id": "basic/chtMultiRegionFoam/2DImplicitCyclic",

  "solver": "chtMultiRegionFoam",
  "physics": ["heat_transfer", "multi_region"],
  "steady_or_transient": "transient",
  "dimensionality": "2D",
  "turbulence_model": "laminar",
  "mesh_type": "script",
  "bc_types": ["fixedValue", "zeroGradient"],
  "files": ["system/fvSchemes", "system/fvSolution", "0/T"],
  "run_commands": ["./Allmesh", "chtMultiRegionFoam"],
  "openfoam_version": "v2512",

  "title_ja": "…",
  "summary_ja": "…",
  "suitable_for_ja": ["…"],
  "not_suitable_for_ja": ["…"],
  "source_hash": "…"
}
```

v1 スキーマからの変更点:

- `solver` / `steady_or_transient` / `dimensionality` / `turbulence_model` / `bc_types` を構造化フィールドとして新設（v1 では散文中に埋没していた）
- `phenomenon` / `geometry` は**列挙型語彙を固定**し、該当なしは `"general"` ではなく `null` とする（"general" はフィルタとして無意味なため）
- `run_commands` は Allrun / Allmesh をパースして必ず埋める（v1 で空配列になる抽出漏れがあった）
- `bc_summary_ja`（散文）は廃止し `bc_types`（実際に使用されている BC 型名リスト）へ置換
- 埋め込み対象テキスト: `title_ja + summary_ja + suitable_for_ja`。構造化フィールドは埋め込まずメタデータフィルタ専用

### 11.4 知識ベース拡張の優先順位

前提: 一般流体力学・CFD の概念知識は LLM の重みに既に十分含まれる。RAG が補うべきは**モデルが持たない・間違えやすい脆い知識**（v2512 固有の記法、エラーパターン）である。一般教科書の取り込みは行わない（効果薄・著作権リスク）。

| 優先度 | 内容 | ソース | 備考 |
|---|---|---|---|
| 1 | エラー→原因→修正のマッピング | **自前生成**: RunRecord + 修正 diff + 成否を蓄積 | エージェントが使うほど賢くなる自己改善構造。差別化ポイント |
| 2 | BC 型・キーワードのリファレンス | OpenFOAM ソースの各 BC 型ヘッダ（.H）のドキュメントコメント | v2512 固有記法はモデルが最も間違える部分 |
| 3 | 数値スキーム・収束の解説 | OpenFOAM User Guide 該当章 | チュートリアルにない「なぜこの設定か」の知識 |
| — | 一般流体力学の教科書 | — | やらない |

**運用方針**: 先回りで全部入れない。Phase 1 を回し、エージェントが実際に失敗・誤答した箇所のログを観測してから拡張する（不足は推測でなく観測で決める）。優先度1の蓄積機構（RunRecord の永続化）だけは Phase 1 から仕込んでおく。

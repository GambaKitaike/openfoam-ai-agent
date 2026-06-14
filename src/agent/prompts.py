"""システムプロンプト構築（DESIGN.md §5.3）。"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from src.agent.session import SessionState
from src.tools.fs_tools import list_files

_MESH_ENTITY_FILES = ("points", "faces", "owner")

_CONVERGENCE_RULES = """\
【収束の鉄則】
- blockMesh 単独の場合、必ず noSlip 壁パッチが 1 つ以上必要（symmetryPlane/empty のみでは定常収束しない）
- 壁あり内部流れ → SIMPLEC (consistent yes) + linearUpwind → 50〜300 step で定常収束が目安
- 壁なし純外部流れ → STL を使った snappyHexMesh で物体面を wall 化する
- 2D 円柱・カルマン渦: 上下境界は slip / zeroGradient（symmetryPlane は使わない）
- Re のみ指定時は U = 1 m/s 固定、ν = U·L/Re で整合させる
- 非定常ケース: blockMesh → checkMesh 後に Δt = maxCo × Δx / U（maxCo=0.5）で初期 deltaT を設定
- 非定常実行中は adjustTimeStep yes / maxCo 0.5 / maxDeltaT ≈ 100 × deltaT
- 定常 simpleFoam は残差閾値（既定 1e-4）を下回れば収束。非定常 pimpleFoam は Time/endTime 進捗で完了判定
"""

_STANDARD_WORKFLOW = """\
【標準ワークフロー】
- 新規ケース（下記）: case_scaffold → run_openfoam blockMesh → run_openfoam checkMesh → (potentialFoam) → ソルバー → summarize
- 既存ケース: スナップショットのメッシュ・計算状態を確認し、既存資産（メッシュ・時刻ディレクトリ）を再利用して続きを実行する
- エラー時: read_log(errors) → 原因の仮説 → rag_search（必要時）→ edit_file → 再実行
"""

_BEHAVIOR_RULES = """\
【行動規範】
- 編集は最小差分（edit_file の str_replace）。全文書き換えは避ける
- 実行・編集の前に、次に何をするか 1 行で宣言する
- ユーザーの指示にファイル編集と実行の両方が含まれる場合、編集をすべて完了・適用してから実行ツールを呼ぶ（実行を先に済ませて編集を後回しにしない）
- 「半分」「倍」などの相対的な数値変更や、特定の値への変更を指示された場合は、必ず対象ファイルを read_file で現在値を確認してから edit_file する
- workspace に既にメッシュが存在する場合（スナップショットで「メッシュ: 生成済み」）、blockMesh / snappyHexMesh を再実行しない。ユーザーが明示的にメッシュ再生成を指示した場合のみ実行する（既存メッシュの意図しない上書きを避ける）
- ユーザーが特定のソルバー/コマンド（例: pimpleFoam）の実行を明示した場合は、不要な前段ステップ（メッシュ再生成・不要な checkMesh）を挟まず、指示されたコマンドの実行を優先する。前段が本当に必要な場合（メッシュ未生成など）のみ最小限で補う
- blockMesh を実行した場合は、その直後に run_openfoam checkMesh を1回実行してからソルバーへ進む
- foam_dict_check は foamDictionary による dict 構文チェックであり、checkMesh（メッシュ品質確認）とは別物
- 不確かな物理設定（Re 数域、乱流モデル、境界条件の解釈）はユーザーに確認する
- OpenFOAM dict（0/, system/, constant/）編集後は構文チェック結果を確認する
- dict の構文エラーが不正な行・余分なトークンの混入による場合は、その不正な部分を削除して直すことを最優先とする。閉じ括弧やトークンを新たに補って辻褄を合わせようとしない（安易な括弧の追加は別の構文エラーを誘発しやすい）
- edit_file が "not unique"（old_str が複数箇所に一致）で失敗した場合は、同じ old_str で再試行しない。対象行の前後の行を含めた、ファイル内で一意になる文字列に old_str を拡張してからやり直す
- edit_file が "old_str と new_str が同一" で失敗した場合は、変更が実際に反映される old_str / new_str の組に作り直す（同一内容で再試行しない）
- 同じファイルに対する編集が 2 回続けて失敗した場合は、いったん read_file でファイル全体を読み直して現在の正確な内容を確認してから次の編集を組み立てる
"""


def _format_spec_summary(state: SessionState) -> str:
    if state.spec is None:
        return "(未設定)"
    spec = state.spec
    lines = [
        f"solver: {spec.solver}",
        f"case_type: {spec.case_type}",
        f"dimensions: {spec.dimensions}D",
        f"turbulence_model: {spec.turbulence_model}",
        f"steady_state: {spec.steady_state}",
    ]
    if spec.description:
        lines.append(f"description: {spec.description}")
    if spec.re_number is not None:
        lines.append(f"re_number: {spec.re_number:.0f}")
    if spec.inlet_velocity:
        lines.append(f"inlet_velocity: {spec.inlet_velocity} m/s")
    return "\n".join(lines)


def _format_run_records_summary(state: SessionState) -> str:
    if not state.run_records:
        return "(実行履歴なし)"
    lines: list[str] = []
    for record in state.run_records[-5:]:
        status = "OK" if record.exit_code == 0 else f"exit {record.exit_code}"
        lines.append(f"- {record.command} [{status}]: {record.summary}")
    if len(state.run_records) > 5:
        lines.insert(0, f"(直近 5 件 / 全 {len(state.run_records)} 件)")
    return "\n".join(lines)


def _is_mesh_generated(workspace: Path) -> bool:
    """constant/polyMesh にメッシュ実体（points / faces / owner）が存在するか。"""
    poly_mesh = workspace / "constant" / "polyMesh"
    if not poly_mesh.is_dir():
        return False
    return all((poly_mesh / name).exists() for name in _MESH_ENTITY_FILES)


def _mesh_status_line(workspace: Path) -> str:
    if _is_mesh_generated(workspace):
        return "メッシュ: 生成済み（constant/polyMesh あり）"
    return "メッシュ: 未生成"


def _latest_solver_time(workspace: Path) -> float | None:
    """0 以外の数値タイムディレクトリの最大時刻を返す。"""
    times: list[float] = []
    for entry in workspace.iterdir():
        if not entry.is_dir() or entry.name.startswith("processor"):
            continue
        try:
            time_value = float(entry.name)
        except ValueError:
            continue
        if time_value != 0.0:
            times.append(time_value)
    return max(times) if times else None


def _computation_status_line(workspace: Path) -> str:
    latest = _latest_solver_time(workspace)
    if latest is not None:
        return f"計算: 実行済み（最終時刻 {latest:g}）"
    return "計算: 未実行"


def _workspace_snapshot(state: SessionState) -> str:
    workspace = state.workspace.resolve()
    summary = "\n".join(
        [
            _mesh_status_line(workspace),
            _computation_status_line(workspace),
        ]
    )
    result = list_files(state.workspace, path=".", depth=2)
    if result.ok:
        return f"{summary}\n\n{result.content}"
    return f"{summary}\n\n(ワークスペース走査失敗: {result.content})"


def build_system_prompt(state: SessionState) -> str:
    """§5.3 の 5 要素を含むシステムプロンプトを構築する。"""
    workspace_path = state.workspace.resolve().as_posix()
    spec_json = "(なし)" if state.spec is None else str(asdict(state.spec))

    return f"""\
あなたは OpenFOAM ケースの構築・実行・デバッグを行う対話型エージェントです。
ユーザーのワークスペース内のケースファイルを読み書きし、許可された OpenFOAM コマンドを実行し、
結果を要約して次の手を提案します。SI 単位系を使用してください。

{_CONVERGENCE_RULES}
{_STANDARD_WORKFLOW}
{_BEHAVIOR_RULES}
【ワークスペース情報】
path: {workspace_path}

ファイルツリー（list_files スナップショット）:
{_workspace_snapshot(state)}

SimulationSpec 要約:
{_format_spec_summary(state)}

実行履歴（run_records 要約）:
{_format_run_records_summary(state)}

spec (raw): {spec_json}
"""

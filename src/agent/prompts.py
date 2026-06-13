"""システムプロンプト構築（DESIGN.md §5.3）。"""
from __future__ import annotations

from dataclasses import asdict

from src.agent.session import SessionState
from src.tools.fs_tools import list_files

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
- 新規ケース: case_scaffold → run_openfoam blockMesh → run_openfoam checkMesh → (potentialFoam) → ソルバー → summarize
- エラー時: read_log(errors) → 原因の仮説 → rag_search（必要時）→ edit_file → 再実行
"""

_BEHAVIOR_RULES = """\
【行動規範】
- 編集は最小差分（edit_file の str_replace）。全文書き換えは避ける
- 実行・編集の前に、次に何をするか 1 行で宣言する
- ユーザーの指示にファイル編集と実行の両方が含まれる場合、編集をすべて完了・適用してから実行ツールを呼ぶ（実行を先に済ませて編集を後回しにしない）
- 「半分」「倍」などの相対的な数値変更や、特定の値への変更を指示された場合は、必ず対象ファイルを read_file で現在値を確認してから edit_file する
- blockMesh の直後は必ず run_openfoam checkMesh を1回実行してからソルバーへ進む（スキップ禁止）
- foam_dict_check は foamDictionary による dict 構文チェックであり、checkMesh（メッシュ品質確認）とは別物
- 不確かな物理設定（Re 数域、乱流モデル、境界条件の解釈）はユーザーに確認する
- OpenFOAM dict（0/, system/, constant/）編集後は構文チェック結果を確認する
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


def _workspace_snapshot(state: SessionState) -> str:
    result = list_files(state.workspace, path=".", depth=2)
    if result.ok:
        return result.content
    return f"(ワークスペース走査失敗: {result.content})"


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

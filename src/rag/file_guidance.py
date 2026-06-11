"""Agent②: ファイル単位 OpenFOAM syntax ガイダンス。"""
from __future__ import annotations

from ..models import EnrichedContext, SimulationSpec

FILE_GUIDANCE_TIPS: dict[str, str] = {
    "constant/transportProperties": "nu は transportModel Newtonian; の下に nu [m²/s]; を置く。",
    "constant/turbulenceProperties": "simulationType laminar または RAS; RASModel は kOmegaSST 等。",
    "system/controlDict": "application, endTime, deltaT, writeInterval, writeControl を spec に合わせる。",
    "system/fvSchemes": "定常は steadyState、非定常は backward/Euler 等。divSchemes は bounded 系を優先。",
    "system/fvSolution": "ソルバー名は application と整合。SIMPLE/PIMPLE/PISO ブロックを分ける。",
    "0/U": "boundaryField は polyMesh/boundary のパッチ名と完全一致させる。2D は frontAndBack を empty。",
    "0/p": "inlet は usually zeroGradient、outlet は fixedValue 0 が一般的（外部流れ）。",
    "system/setFieldsDict": "defaultFieldValues と regions で internalField を上書き。",
}


def build_file_guidance(
    rel_path: str,
    spec: SimulationSpec,
    *,
    reference_content: str = "",
    case_label: str = "",
    patch_names: list[str] | None = None,
    question: str = "",
) -> str:
    """参照ファイル断片 + spec から Agent③ 向けガイダンス文字列を組み立てる。"""
    lines = [f"対象ファイル: {rel_path}"]
    if case_label:
        lines.append(f"参照ケース: {case_label}")
    lines.append(
        f"解析条件: solver={spec.solver}, "
        f"{'定常' if spec.steady_state else '非定常'}, "
        f"乱流={spec.turbulence_model}, "
        f"U={spec.inlet_velocity:g} m/s, nu={spec.nu:g}, L={spec.characteristic_length:g} m"
    )
    if patch_names:
        lines.append(f"メッシュパッチ: {', '.join(patch_names)}")

    tip = FILE_GUIDANCE_TIPS.get(rel_path, "")
    if tip:
        lines.append(f"構文メモ: {tip}")

    if reference_content:
        snippet = reference_content.strip()
        if len(snippet) > 3500:
            snippet = snippet[:3500] + "\n... (truncated)"
        lines.append("\n参照ファイル例（構文の手本）:\n" + snippet)
    elif not case_label:
        lines.append(
            "\n参照ケースなし — OpenFOAM v2512 形式・FoamFile ヘッダ・"
            "上記解析条件に合わせた boundaryField を生成すること。"
        )

    if question:
        lines.append(f"\nAgent③ からの質問: {question}")

    return "\n".join(lines)


def resolve_reference_snippet(
    rel_path: str,
    spec: SimulationSpec,
    context: EnrichedContext | None,
    selector,
) -> tuple[str, str]:
    """EnrichedContext または RAG 再検索から参照ファイル内容を取得。"""
    if context and context.reference_files.get(rel_path):
        label = context.reference_title_ja or context.reference_case_id
        return context.reference_files[rel_path], label

    if selector and selector.is_available:
        selected = selector.select(spec)
        if selected:
            content = selected.get("reference_files", {}).get(rel_path, "")
            if content:
                label = selected.get("title_ja") or selected.get("case_id", "")
                return content, label

    return "", ""

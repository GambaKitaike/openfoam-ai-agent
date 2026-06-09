"""参照ケース適応 + 決定的ビルダー + LLM フォールバック。"""
from __future__ import annotations

import re
from pathlib import Path

from ..llm_client import LLMClient
from ..models import EnrichedContext, SimulationSpec
from . import builders

FILE_BUILDERS = {
    "constant/transportProperties": builders.build_transport_properties,
    "constant/turbulenceProperties": builders.build_turbulence_properties,
    "system/fvSchemes": builders.build_fv_schemes,
    "system/fvSolution": builders.build_fv_solution,
    "system/controlDict": builders.build_control_dict,
}

ZERO_BUILDERS = {
    "0/U": builders.build_u_field,
    "0/p": builders.build_p_field,
}

LLM_SYSTEM = """OpenFOAM v2512 のケースファイルを1つだけ生成してください。
- コードブロック(```)は使わない
- FoamFile ヘッダを含める
- 参照例の構文を踏襲しつつ、与えられた解析条件に合わせる
"""


class FileGenerator:
    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm

    def generate(
        self,
        rel_path: str,
        context: EnrichedContext,
        patch_names: list[str] | None = None,
    ) -> str:
        spec = context.spec
        ref = context.reference_files.get(rel_path, "")

        if ref and self._can_adapt_reference(rel_path, ref, spec, patch_names):
            return self._adapt_reference(rel_path, ref, spec, patch_names or [])

        if rel_path in FILE_BUILDERS:
            return FILE_BUILDERS[rel_path](spec)

        if rel_path in ZERO_BUILDERS and patch_names:
            return ZERO_BUILDERS[rel_path](spec, patch_names)

        if rel_path == "system/setFieldsDict":
            return builders.build_set_fields_dict(spec)

        if self.llm and ref:
            return self._llm_generate(rel_path, spec, ref, patch_names)

        raise ValueError(f"No builder for {rel_path}")

    def _can_adapt_reference(
        self, rel_path: str, ref: str, spec: SimulationSpec, patch_names: list[str] | None
    ) -> bool:
        if rel_path.startswith("0/") and patch_names:
            ref_patches = set(re.findall(r"^\s{4}(\w+)\s*\{", ref, re.MULTILINE))
            return ref_patches <= set(patch_names) or ref_patches == set(patch_names)
        if rel_path == "constant/transportProperties":
            return "nu" in ref
        return rel_path in FILE_BUILDERS

    def _adapt_reference(
        self, rel_path: str, ref: str, spec: SimulationSpec, patch_names: list[str]
    ) -> str:
        if rel_path == "constant/transportProperties":
            return builders.build_transport_properties(spec)
        if rel_path == "constant/turbulenceProperties":
            return builders.build_turbulence_properties(spec)
        if rel_path == "system/controlDict":
            text = ref
            text = re.sub(r"application\s+\w+\s*;", f"application     {spec.solver};", text, count=1)
            return builders.build_control_dict(spec)
        if rel_path in ZERO_BUILDERS:
            return ZERO_BUILDERS[rel_path](spec, patch_names)
        if rel_path in FILE_BUILDERS:
            return FILE_BUILDERS[rel_path](spec)
        return ref

    def _llm_generate(
        self, rel_path: str, spec: SimulationSpec, ref: str, patch_names: list[str] | None
    ) -> str:
        prompt = f"""ファイル: {rel_path}
解析: {spec.description}
ソルバー: {spec.solver}, 定常: {spec.steady_state}, 乱流: {spec.turbulence_model}
U={spec.inlet_velocity}, nu={spec.nu}, L={spec.characteristic_length}
パッチ: {patch_names or []}

参照例:
{ref[:4000]}

上記を参考に完全なファイル内容のみ出力してください。"""
        return self.llm.chat(prompt, system=LLM_SYSTEM)

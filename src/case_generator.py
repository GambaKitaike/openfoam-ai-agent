"""
OpenFOAMケースファイル生成器 - 決定的ビルダーでファイルを生成する
"""
from __future__ import annotations

import re
from pathlib import Path

from .case_builder import builders
from .config import Settings
from .llm_client import AnalysisSpec
from .models import GenerationResult, SimulationSpec


class CaseGenerator:
    """AnalysisSpecからOpenFOAMケースファイルを生成するクラス。"""

    def __init__(self, settings: Settings):
        self.settings = settings

    def generate(
        self,
        spec: AnalysisSpec,
        output_dir: str,
        block_mesh_dict_content: str | None = None,
    ) -> GenerationResult:
        sim = self._to_simulation_spec(spec)
        out_path = Path(output_dir)
        case_name = self._make_case_name(spec)
        case_path = out_path / case_name
        case_path.mkdir(parents=True, exist_ok=True)
        (case_path / "0").mkdir(exist_ok=True)
        (case_path / "constant").mkdir(exist_ok=True)
        (case_path / "system").mkdir(exist_ok=True)

        files_created: list[str] = []
        if block_mesh_dict_content:
            (case_path / "system" / "blockMeshDict").write_text(block_mesh_dict_content)
            files_created.append("system/blockMeshDict")

        patch_names = ["inlet", "outlet", "top", "bottom", "front", "back"]
        builder_map = {
            "system/controlDict": lambda: builders.build_control_dict(sim),
            "system/fvSchemes": lambda: builders.build_fv_schemes(sim),
            "system/fvSolution": lambda: builders.build_fv_solution(sim),
            "constant/turbulenceProperties": lambda: builders.build_turbulence_properties(sim),
            "constant/transportProperties": lambda: builders.build_transport_properties(sim),
            "0/U": lambda: builders.build_u_field(sim, patch_names),
            "0/p": lambda: builders.build_p_field(sim, patch_names),
        }

        for rel, fn in builder_map.items():
            try:
                content = fn()
                (case_path / rel).write_text(content)
                files_created.append(rel)
            except Exception:
                pass

        return GenerationResult(
            output_path=str(case_path),
            case_type=spec.case_type,
            files_created=files_created,
        )

    def _make_case_name(self, spec: AnalysisSpec) -> str:
        name = f"{spec.solver}_{spec.case_type}"
        return re.sub(r"[^\w-]", "_", name)

    @staticmethod
    def _to_simulation_spec(spec: AnalysisSpec) -> SimulationSpec:
        bc = spec.boundary_conditions
        raw_velocity = bc.get("inlet", {}).get("velocity", "10")
        inlet_velocity = CaseGenerator._parse_velocity(raw_velocity)
        return SimulationSpec(
            solver=spec.solver,
            case_type=spec.case_type,
            mesh_template="box_channel_2d",
            turbulence_model=spec.turbulence_model,
            steady_state=spec.steady_state,
            inlet_velocity=inlet_velocity,
            dimensions=spec.dimensions,
            description=spec.description,
        )

    @staticmethod
    def _parse_velocity(value) -> float:
        if isinstance(value, (int, float)):
            return float(value)
        match = re.search(r"[\d.]+", str(value))
        if match:
            return float(match.group())
        return 10.0

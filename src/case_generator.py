"""
OpenFOAMケースファイル生成器 - Jinja2テンプレートを使ってファイルを生成する
"""
from __future__ import annotations

import os
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from .config import Settings
from .llm_client import AnalysisSpec
from .models import GenerationResult


TEMPLATES_DIR = Path(__file__).parent.parent / "templates"


class CaseGenerator:
    """AnalysisSpecからOpenFOAMケースファイルを生成するクラス。"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(TEMPLATES_DIR)),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def generate(self, spec: AnalysisSpec, output_dir: str, block_mesh_dict_content: str | None = None) -> GenerationResult:
        """
        解析仕様に基づいてOpenFOAMケースファイル一式を生成する。

        Args:
            spec: 解析仕様
            output_dir: 出力先ディレクトリ
            block_mesh_dict_content: LLMが生成したblockMeshDictの内容（Noneの場合はテンプレートを使用）

        Returns:
            GenerationResult
        """
        # 出力ディレクトリを作成
        out_path = Path(output_dir)
        case_name = self._make_case_name(spec)
        case_path = out_path / case_name
        case_path.mkdir(parents=True, exist_ok=True)

        # OpenFOAMのディレクトリ構造を作成
        (case_path / "0").mkdir(exist_ok=True)
        (case_path / "constant").mkdir(exist_ok=True)
        (case_path / "system").mkdir(exist_ok=True)

        files_created = []

        # blockMeshDict はLLM生成コンテンツを直接書き出す
        if block_mesh_dict_content:
            bmd_path = case_path / "system" / "blockMeshDict"
            bmd_path.write_text(block_mesh_dict_content)
            files_created.append("system/blockMeshDict")

        # テンプレートから生成するファイル（blockMeshDictは除く）
        template_map = {
            "system/controlDict": "system/controlDict.j2",
            "system/fvSchemes": "system/fvSchemes.j2",
            "system/fvSolution": "system/fvSolution.j2",
            "constant/turbulenceProperties": "constant/turbulenceProperties.j2",
            "constant/transportProperties": "constant/transportProperties.j2",
            "0/U": "0/U.j2",
            "0/p": "0/p.j2",
            "0/k": "0/k.j2",
            "0/omega": "0/omega.j2",
            "0/nut": "0/nut.j2",
        }

        context = self._build_context(spec)

        for output_rel, template_name in template_map.items():
            try:
                template = self.jinja_env.get_template(template_name)
                content = template.render(**context)
                output_file = case_path / output_rel
                output_file.write_text(content)
                files_created.append(str(output_rel))
            except Exception:
                pass

        return GenerationResult(
            output_path=str(case_path),
            case_type=spec.case_type,
            files_created=files_created,
        )

    def _make_case_name(self, spec: AnalysisSpec) -> str:
        """ケース名を生成する。"""
        import re
        name = f"{spec.solver}_{spec.case_type}"
        return re.sub(r'[^\w-]', '_', name)

    def _build_context(self, spec: AnalysisSpec) -> dict:
        """テンプレートに渡すコンテキスト変数を構築する。"""
        bc = spec.boundary_conditions
        raw_velocity = bc.get("inlet", {}).get("velocity", "10")
        inlet_velocity = self._parse_velocity(raw_velocity)

        return {
            "solver": spec.solver,
            "case_type": spec.case_type,
            "turbulence_model": spec.turbulence_model,
            "steady_state": spec.steady_state,
            "dimensions": spec.dimensions,
            "description": spec.description,
            "inlet_velocity": inlet_velocity,
            "end_time": 1000 if spec.steady_state else 1.0,
            "delta_t": 1 if spec.steady_state else 0.001,
            "write_interval": 100 if spec.steady_state else 0.1,
        }

    @staticmethod
    def _parse_velocity(value) -> float:
        """
        速度の値から数値だけを取り出す。
        "10m/s" "10 m/s" "10.5" 10 などに対応。
        """
        import re
        if isinstance(value, (int, float)):
            return float(value)
        # 先頭の数値部分だけ抽出
        match = re.search(r"[\d.]+", str(value))
        if match:
            return float(match.group())
        return 10.0  # フォールバック

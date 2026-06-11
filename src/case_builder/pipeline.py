"""段階的ケース生成パイプライン。"""
from __future__ import annotations

import re
import shutil
from collections.abc import Callable
from pathlib import Path

from rich.console import Console

from ..case_validator import CaseValidator
from ..config import Settings
from ..llm_client import LLMClient
from ..models import CaseBuildState, EnrichedContext, GenerationResult, SimulationSpec
from ..runner import OpenFOAMRunner
from .file_generators import FileGenerator
from .mesh_generators import render_block_mesh_dict
from .policy import read_patch_names

console = Console()

BUILD_STEPS = [
    "transportProperties",
    "turbulenceProperties",
    "controlDict",       # blockMesh / checkMesh が case 読み込みに必要
    "blockMeshDict",
    "zero_fields",
    "fv_system",
    "karman_seed",
]


class CaseBuildPipeline:
    """依存順に OpenFOAM ケースファイルを生成する。"""

    def __init__(
        self,
        settings: Settings,
        guidance_fn: Callable[[str, SimulationSpec, list[str] | None], str] | None = None,
    ):
        self.settings = settings
        self.runner = OpenFOAMRunner(settings)
        self.validator = CaseValidator()
        self.file_gen = FileGenerator(LLMClient(settings), guidance_fn=guidance_fn)

    def run(
        self,
        context: EnrichedContext,
        output_dir: str,
        *,
        run_mesh: bool = True,
    ) -> tuple[CaseBuildState, GenerationResult]:
        spec = context.spec
        case_name = re.sub(r"[^\w-]", "_", f"{spec.solver}_{spec.case_type}")
        case_path = Path(output_dir) / case_name
        self._reset_case_dir(case_path)

        state = CaseBuildState(case_dir=str(case_path), spec=spec, build_path="staged")
        files: list[str] = []

        # 1. transportProperties
        content = self.file_gen.generate("constant/transportProperties", context)
        self._write(case_path, "constant/transportProperties", content)
        files.append("constant/transportProperties")
        state.completed_steps.append("transportProperties")

        # 2. turbulenceProperties
        content = self.file_gen.generate("constant/turbulenceProperties", context)
        self._write(case_path, "constant/turbulenceProperties", content)
        files.append("constant/turbulenceProperties")
        state.completed_steps.append("turbulenceProperties")

        # 3. controlDict（blockMesh 実行前に必須）
        content = self.file_gen.generate("system/controlDict", context)
        self._write(case_path, "system/controlDict", content)
        files.append("system/controlDict")
        state.completed_steps.append("controlDict")

        # 4. blockMeshDict + mesh
        bmd = render_block_mesh_dict(context)
        self._write(case_path, "system/blockMeshDict", bmd)
        files.append("system/blockMeshDict")
        state.completed_steps.append("blockMeshDict")

        mesh_built = False
        if run_mesh:
            bm = self.runner.run_block_mesh(str(case_path))
            if bm.returncode == 0:
                cm = self.runner.run_check_mesh(str(case_path))
                if not cm.success:
                    console.print("[yellow]  ⚠ checkMesh warnings[/yellow]")
                state.patch_names = read_patch_names(str(case_path))
                mesh_built = True
            else:
                log = bm.log_file or "log.blockMesh"
                console.print(
                    f"[yellow]  ⚠ pipeline 内 blockMesh 失敗 ({log})"
                    f" — Agent③ の自己修正ループで再試行します[/yellow]"
                )
                state.patch_names = self._default_patches(spec)
        else:
            state.patch_names = self._default_patches(spec)

        # 5. 0/ boundary fields
        for rel in ("0/U", "0/p"):
            content = self.file_gen.generate(rel, context, state.patch_names)
            self._write(case_path, rel, content)
            files.append(rel)
        if spec.turbulence_model != "laminar":
            for rel in ("0/k", "0/omega", "0/nut"):
                ref = context.reference_files.get(rel, "")
                if ref:
                    self._write(case_path, rel, ref)
                    files.append(rel)
        state.completed_steps.append("zero_fields")

        # 6. fvSchemes / fvSolution
        for rel in ("system/fvSchemes", "system/fvSolution"):
            content = self.file_gen.generate(rel, context)
            self._write(case_path, rel, content)
            files.append(rel)
        state.completed_steps.append("fv_system")

        issues = self.validator.validate(case_path, spec, after_blockmesh=bool(state.patch_names))
        for issue in issues:
            if issue.severity == "error":
                console.print(f"[yellow]  ⚠ {issue.check}: {issue.message}[/yellow]")

        # 7. karman seed (optional)
        if spec.phenomenon == "karman_vortex_shedding" and spec.case_type == "cylinder_2d_ogrid":
            sf = self.file_gen.generate("system/setFieldsDict", context)
            self._write(case_path, "system/setFieldsDict", sf)
            files.append("system/setFieldsDict")
            state.completed_steps.append("karman_seed")

        state.files_created = files
        gen = GenerationResult(
            output_path=str(case_path),
            case_type=spec.case_type,
            files_created=files,
            mesh_built=mesh_built,
            build_path="staged",
        )
        console.print(f"  [green]段階的生成完了[/green] ({len(files)} ファイル)")
        return state, gen

    def apply_karman_seed(self, case_dir: str) -> bool:
        sf_path = Path(case_dir) / "system" / "setFieldsDict"
        if not sf_path.exists():
            return False
        result = self.runner.run_set_fields(case_dir)
        return result.returncode == 0

    @staticmethod
    def _reset_case_dir(case_path: Path) -> None:
        if case_path.exists():
            for name in case_path.iterdir():
                if name.is_dir():
                    shutil.rmtree(name)
                else:
                    name.unlink()
        else:
            case_path.mkdir(parents=True, exist_ok=True)
        for sub in ("0", "constant", "system"):
            (case_path / sub).mkdir(exist_ok=True)

    @staticmethod
    def _write(case_path: Path, rel: str, content: str) -> None:
        dest = case_path / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content)

    @staticmethod
    def _default_patches(spec) -> list[str]:
        if spec.case_type == "cylinder_2d_ogrid":
            return ["inlet", "outlet", "top", "bottom", "cylinder", "frontAndBack"]
        if spec.dimensions == 2:
            return ["inlet", "outlet", "top", "bottom", "front", "back"]
        return ["inlet", "outlet", "top", "bottom", "front", "back"]

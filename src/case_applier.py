"""
参照ケースの適用と安全なパラメータ差し替え
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

from rich.console import Console

from .models import EnrichedContext, SimulationSpec

console = Console()

LAMINAR_TURBULENCE_PROPERTIES = """\
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      turbulenceProperties;
}
simulationType  laminar;
"""

class CaseApplier:
    """参照チュートリアルケースを出力ディレクトリにコピーし、パラメータを差し替える。"""

    def apply(
        self,
        context: EnrichedContext,
        case_path: Path,
    ) -> list[str]:
        """
        reference_files を case_path に書き込み、SimulationSpec のパラメータを差し替える。

        Returns:
            作成したファイルの相対パスリスト
        """
        spec = context.spec
        files_created: list[str] = []

        if not context.reference_files:
            return files_created

        for rel, content in context.reference_files.items():
            dest = case_path / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content)
            files_created.append(rel)

        self._substitute_parameters(case_path, spec, files_created, context)
        self._sync_control_dict_solver(case_path, spec)

        if context.reference_mesh_prebuilt and context.reference_case_path:
            if copy_prebuilt_mesh(Path(context.reference_case_path), case_path):
                console.print("  [dim]事前メッシュ (polyMesh) をコピーしました[/dim]")

        console.print(
            f"  [green]参照ケース適用: {context.reference_case_id}[/green] "
            f"({len(files_created)} ファイル)"
        )
        return files_created

    def _substitute_parameters(
        self,
        case_path: Path,
        spec: SimulationSpec,
        files_created: list[str],
        context: EnrichedContext | None = None,
    ) -> None:
        """安全なスカラー値のみ差し替え。"""
        preserve_turbulence = bool(context and context.reference_mesh_prebuilt)
        # turbulenceProperties
        if spec.turbulence_model == "laminar" and not preserve_turbulence:
            tp_path = case_path / "constant" / "turbulenceProperties"
            tp_path.write_text(LAMINAR_TURBULENCE_PROPERTIES)
            for fname in ("k", "omega", "epsilon", "nut", "nuTilda", "alphat"):
                f = case_path / "0" / fname
                if f.exists():
                    f.unlink()
        else:
            tp_path = case_path / "constant" / "turbulenceProperties"
            if tp_path.exists() and not preserve_turbulence:
                text = tp_path.read_text()
                if spec.turbulence_model == "kOmegaSST" and "kOmegaSST" not in text:
                    if "kEpsilon" in text:
                        text = text.replace("kEpsilon", "kOmegaSST")
                tp_path.write_text(text)

        # transportProperties: nu
        tr = case_path / "constant" / "transportProperties"
        if tr.exists():
            text = tr.read_text()
            text = re.sub(
                r"(nu\s+)([\d.eE+-]+)(\s*;)",
                rf"\g<1>{spec.nu:g}\g<3>",
                text,
                count=1,
            )
            tr.write_text(text)

        # controlDict: endTime, deltaT, application
        cd = case_path / "system" / "controlDict"
        if cd.exists():
            text = cd.read_text()
            text = re.sub(r"application\s+\w+\s*;", f"application     {spec.solver};", text, count=1)
            end_time = self._compute_end_time(spec)
            text = re.sub(r"endTime\s+[\d.eE+-]+\s*;", f"endTime         {end_time:g};", text, count=1)
            if not spec.steady_state:
                delta_t = self._compute_delta_t(spec)
                text = re.sub(r"deltaT\s+[\d.eE+-]+\s*;", f"deltaT          {delta_t:g};", text, count=1)
                if spec.solver == "icoFoam":
                    text = re.sub(r"adjustTimeStep\s+\w+\s*;", "adjustTimeStep  no;", text)
            cd.write_text(text)

        # 0/U: inlet velocity
        u_file = case_path / "0" / "U"
        if u_file.exists():
            text = u_file.read_text()
            text = self._replace_inlet_velocity(text, spec.inlet_velocity)
            u_file.write_text(text)

    def _replace_inlet_velocity(self, text: str, velocity: float) -> str:
        """inlet パッチと internalField の x 成分を更新。"""
        vec = f"({velocity:g} 0 0)"
        text = re.sub(
            r"(internalField\s+uniform\s+\()[^)]+(\))",
            rf"\g<1>{velocity:g} 0 0\g<2>",
            text,
            count=1,
        )
        # inlet fixedValue
        text = re.sub(
            r"(inlet\s*\{[^}]*value\s+uniform\s+\()[^)]+(\))",
            rf"\g<1>{velocity:g} 0 0\g<2>",
            text,
            flags=re.DOTALL,
        )
        return text

    def _compute_end_time(self, spec: SimulationSpec) -> float:
        if not spec.steady_state:
            return spec.boundary_conditions.get("end_time", 30.0) if isinstance(
                spec.boundary_conditions.get("end_time"), (int, float)
            ) else 30.0
        return 1000.0

    def _compute_delta_t(self, spec: SimulationSpec) -> float:
        char_len = spec.characteristic_length or 0.1
        min_cell = char_len * 0.023
        return round(min_cell * 0.3 / max(spec.inlet_velocity * 3, 0.01), 6)

    def _sync_control_dict_solver(self, case_path: Path, spec: SimulationSpec) -> None:
        cd = case_path / "system" / "controlDict"
        if cd.exists():
            text = cd.read_text()
            if f"application     {spec.solver}" not in text:
                text = re.sub(
                    r"application\s+\w+\s*;",
                    f"application     {spec.solver};",
                    text,
                    count=1,
                )
                cd.write_text(text)


def copy_prebuilt_mesh(source_case: Path, dest_case: Path) -> bool:
    """constant/polyMesh.orig または polyMesh を dest の constant/polyMesh にコピー。"""
    dest_pm = dest_case / "constant" / "polyMesh"
    if dest_pm.is_dir() and any(dest_pm.iterdir()):
        return True
    dest_case.mkdir(parents=True, exist_ok=True)
    (dest_case / "constant").mkdir(parents=True, exist_ok=True)
    for src_name in ("polyMesh", "polyMesh.orig"):
        src = source_case / "constant" / src_name
        if src.is_dir():
            if dest_pm.exists():
                shutil.rmtree(dest_pm)
            shutil.copytree(src, dest_pm)
            return True
    return False


def copy_zero_orig_if_needed(case_path: Path) -> None:
    """0/ が空の場合 0.orig を 0/ にコピー。"""
    zero = case_path / "0"
    orig = case_path / "0.orig"
    if orig.is_dir() and not zero.exists():
        shutil.copytree(orig, zero)
    elif orig.is_dir() and zero.is_dir() and not any(zero.iterdir()):
        for f in orig.iterdir():
            if f.is_file():
                shutil.copy2(f, zero / f.name)

"""
OpenFOAM ケースの実行前整合性チェック
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .models import SimulationSpec


@dataclass
class ValidationIssue:
    check: str
    message: str
    severity: str = "error"  # error | warning


class CaseValidator:
    """生成ケースのファイル間整合性を検証する。"""

    def validate(
        self,
        case_dir: Path,
        spec: SimulationSpec,
        *,
        after_blockmesh: bool = False,
    ) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        case_dir = Path(case_dir)

        issues.extend(self._check_solver_consistency(case_dir, spec))
        issues.extend(self._check_ddt_consistency(case_dir, spec))
        issues.extend(self._check_turbulence(case_dir, spec))
        issues.extend(self._check_transport(case_dir, spec))
        issues.extend(self._check_zero_fields(case_dir, spec))

        if after_blockmesh:
            issues.extend(self._check_patch_names(case_dir))

        return issues

    def validate_or_raise(self, case_dir: Path, spec: SimulationSpec, **kwargs) -> bool:
        issues = self.validate(case_dir, spec, **kwargs)
        errors = [i for i in issues if i.severity == "error"]
        return len(errors) == 0

    def _read(self, path: Path) -> str:
        try:
            return path.read_text(errors="ignore") if path.exists() else ""
        except OSError:
            return ""

    def _check_solver_consistency(self, case_dir: Path, spec: SimulationSpec) -> list[ValidationIssue]:
        issues = []
        cd = self._read(case_dir / "system" / "controlDict")
        fv = self._read(case_dir / "system" / "fvSolution")

        app_match = re.search(r"application\s+(\w+)\s*;", cd)
        app = app_match.group(1) if app_match else ""

        if app and app != spec.solver:
            issues.append(ValidationIssue(
                "solver",
                f"controlDict.application={app} != spec.solver={spec.solver}",
            ))

        if spec.steady_state and "SIMPLE" not in fv and app == "simpleFoam":
            issues.append(ValidationIssue("solver", "定常 simpleFoam なのに fvSolution に SIMPLE ブロックなし"))

        if not spec.steady_state:
            algo = {"icoFoam": "PISO", "pimpleFoam": "PIMPLE", "pisoFoam": "PISO"}.get(spec.solver, "")
            if algo and algo not in fv:
                issues.append(ValidationIssue(
                    "solver",
                    f"非定常 {spec.solver} なのに fvSolution に {algo} ブロックなし",
                ))

        return issues

    def _check_ddt_consistency(self, case_dir: Path, spec: SimulationSpec) -> list[ValidationIssue]:
        issues = []
        fs = self._read(case_dir / "system" / "fvSchemes")
        if not spec.steady_state and "steadyState" in fs and "ddtSchemes" in fs:
            if re.search(r"default\s+steadyState", fs):
                issues.append(ValidationIssue(
                    "ddt",
                    "非定常解析なのに fvSchemes.ddtSchemes が steadyState",
                ))
        return issues

    def _check_turbulence(self, case_dir: Path, spec: SimulationSpec) -> list[ValidationIssue]:
        issues = []
        tp = self._read(case_dir / "constant" / "turbulenceProperties")
        if spec.turbulence_model == "laminar":
            if "simulationType      laminar" not in tp and "simulationType  laminar" not in tp:
                if "RAS" in tp or "LES" in tp:
                    issues.append(ValidationIssue(
                        "turbulence",
                        "laminar 指定なのに turbulenceProperties が RAS/LES",
                        severity="warning",
                    ))
            for field in ("k", "omega", "epsilon", "nut"):
                if (case_dir / "0" / field).exists():
                    issues.append(ValidationIssue(
                        "turbulence",
                        f"層流なのに 0/{field} が存在",
                        severity="warning",
                    ))
        return issues

    def _check_transport(self, case_dir: Path, spec: SimulationSpec) -> list[ValidationIssue]:
        issues = []
        tr = self._read(case_dir / "constant" / "transportProperties")
        m = re.search(r"nu\s+([\d.eE+-]+)\s*;", tr)
        if m:
            file_nu = float(m.group(1))
            rel_err = abs(file_nu - spec.nu) / max(spec.nu, 1e-12)
            if rel_err > 0.01:
                issues.append(ValidationIssue(
                    "transport",
                    f"transportProperties.nu={file_nu} != spec.nu={spec.nu}",
                    severity="warning",
                ))
        return issues

    def _check_zero_fields(self, case_dir: Path, spec: SimulationSpec) -> list[ValidationIssue]:
        issues = []
        zero = case_dir / "0"
        if not zero.exists():
            issues.append(ValidationIssue("zero", "0/ ディレクトリが存在しない"))
            return issues
        for required in ("U", "p"):
            if not (zero / required).exists():
                issues.append(ValidationIssue("zero", f"0/{required} が存在しない"))
        return issues

    def _check_patch_names(self, case_dir: Path) -> list[ValidationIssue]:
        issues = []
        boundary = self._read(case_dir / "constant" / "polyMesh" / "boundary")
        if not boundary:
            return issues

        mesh_patches = set(re.findall(r"^\s{4}(\w+)\s*\{", boundary, re.MULTILINE))
        for field_file in (case_dir / "0").glob("*"):
            if not field_file.is_file() or field_file.name.startswith("."):
                continue
            content = self._read(field_file)
            if "boundaryField" not in content:
                continue
            bc_patches = set(re.findall(r"^\s{4}(\w+)\s*\{", content.split("boundaryField")[-1], re.MULTILINE))
            bc_patches -= {"boundaryField"}
            missing = bc_patches - mesh_patches
            extra = mesh_patches - bc_patches - {"defaultFaces", "frontAndBack"}
            if missing:
                issues.append(ValidationIssue(
                    "patch",
                    f"{field_file.name}: BC にあってメッシュにないパッチ: {missing}",
                ))
            if extra and field_file.name in ("U", "p"):
                issues.append(ValidationIssue(
                    "patch",
                    f"{field_file.name}: メッシュにあって BC にないパッチ: {extra}",
                    severity="warning",
                ))
        return issues

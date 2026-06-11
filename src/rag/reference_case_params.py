"""
参照チュートリアルケースから典型物理パラメータを抽出する（Phase A）
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ..models import ReferenceHint, SimulationSpec
from ..rag.case_catalog import _find_zero_dir, _parse_turbulence_model, _read_text


@dataclass
class ReferenceCaseParams:
    """参照ケースから読み取った典型条件。"""
    case_id: str = ""
    title_ja: str = ""
    summary_ja: str = ""
    inlet_velocity: float | None = None
    velocity_note: str = ""
    nu: float | None = None
    turbulence_model: str = ""
    re_from_summary: float | None = None
    characteristic_length: float | None = None
    end_time: float | None = None
    delta_t: float | None = None
    steady_state: bool | None = None
    solver: str = ""

    @property
    def re_number(self) -> float | None:
        if self.re_from_summary is not None:
            return self.re_from_summary
        if (
            self.inlet_velocity is not None
            and self.nu
            and self.characteristic_length
        ):
            return (self.inlet_velocity * self.characteristic_length) / self.nu
        return None

    def to_display_lines(self) -> list[str]:
        lines = []
        if self.inlet_velocity is not None:
            note = f" ({self.velocity_note})" if self.velocity_note else ""
            lines.append(f"流速: {self.inlet_velocity:g} m/s{note}")
        if self.nu is not None:
            lines.append(f"動粘度 nu: {self.nu:g} m²/s")
        if self.turbulence_model:
            lines.append(f"乱流: {self.turbulence_model}")
        if self.re_number is not None:
            src = "intent要約" if self.re_from_summary else "推定"
            lines.append(f"Re: {self.re_number:,.0f} ({src})")
        if self.end_time is not None:
            lines.append(f"endTime: {self.end_time:g}")
        if self.solver:
            lines.append(f"solver: {self.solver}")
        return lines


def _parse_re_from_text(text: str) -> float | None:
    for pat in (
        r"Re\s*[=:：]?\s*([\d.eE+\-]+)",
        r"レイノルズ数\s*[=:：]?\s*([\d.eE+\-]+)",
        r"reynolds\s+number\s*[=:：]?\s*([\d.eE+\-]+)",
    ):
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
    return None


def _parse_nu(text: str) -> float | None:
    m = re.search(r"\bnu\s+([\d.eE+\-]+)\s*;", text)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


def _parse_velocity_from_u(text: str) -> tuple[float | None, str]:
    """internalField または inlet/fixedValue から速度の大きさを推定。"""
    m = re.search(
        r"internalField\s+uniform\s+\(([\d.eE+\-\s]+)\)",
        text,
    )
    if m:
        parts = m.group(1).split()
        if len(parts) >= 2:
            try:
                ux, uy = float(parts[0]), float(parts[1])
                mag = (ux ** 2 + uy ** 2) ** 0.5
                return mag, "internalField"
            except ValueError:
                pass
    m = re.search(
        r"inlet\s*\{[^}]*value\s+uniform\s+\(([\d.eE+\-\s]+)\)",
        text,
        re.DOTALL,
    )
    if m:
        parts = m.group(1).split()
        if parts:
            try:
                vals = [float(x) for x in parts[:3]]
                mag = sum(v * v for v in vals) ** 0.5
                return mag, "inlet fixedValue"
            except ValueError:
                pass
    for patch in ("freestream", "inlet", "inletVelocity"):
        m = re.search(
            rf"{patch}\s*\{{[^}}]*freestreamValue\s+uniform\s+\(([\d.eE+\-\s]+)\)",
            text,
            re.DOTALL,
        )
        if m:
            parts = m.group(1).split()
            if len(parts) >= 2:
                try:
                    ux, uy = float(parts[0]), float(parts[1])
                    return (ux ** 2 + uy ** 2) ** 0.5, f"{patch} freestream"
                except ValueError:
                    pass
    return None, ""


def _parse_control_times(text: str) -> tuple[float | None, float | None]:
    end = None
    dt = None
    m = re.search(r"endTime\s+([\d.eE+\-]+)\s*;", text)
    if m:
        try:
            end = float(m.group(1))
        except ValueError:
            pass
    m = re.search(r"deltaT\s+([\d.eE+\-]+)\s*;", text)
    if m:
        try:
            dt = float(m.group(1))
        except ValueError:
            pass
    return end, dt


def extract_reference_params(
    case_id: str,
    case_path: str | Path,
    reference_files: dict[str, str] | None = None,
    title_ja: str = "",
    summary_ja: str = "",
    metadata: dict | None = None,
) -> ReferenceCaseParams:
    """参照ケースのファイルと intent から典型パラメータを抽出。"""
    case_path = Path(case_path)
    files = reference_files or {}
    meta = metadata or {}

    params = ReferenceCaseParams(
        case_id=case_id,
        title_ja=title_ja,
        summary_ja=summary_ja,
        solver=str(meta.get("solver", "")),
        steady_state=meta.get("steady_state") in (True, "True", "true", "1"),
    )

    params.re_from_summary = _parse_re_from_text(summary_ja) or _parse_re_from_text(title_ja)

    tr = files.get("constant/transportProperties") or _read_text(
        case_path / "constant" / "transportProperties"
    )
    params.nu = _parse_nu(tr)

    u_text = files.get("0/U", "")
    if not u_text:
        zero = _find_zero_dir(case_path)
        if zero:
            u_text = _read_text(zero / "U")
    if u_text:
        vel, note = _parse_velocity_from_u(u_text)
        params.inlet_velocity = vel
        params.velocity_note = note

    tp = files.get("constant/turbulenceProperties") or _read_text(
        case_path / "constant" / "turbulenceProperties"
    )
    if tp:
        params.turbulence_model = _parse_turbulence_model(tp)
    elif meta.get("turbulence_model"):
        params.turbulence_model = str(meta["turbulence_model"])

    cd = files.get("system/controlDict") or _read_text(case_path / "system" / "controlDict")
    if cd:
        params.end_time, params.delta_t = _parse_control_times(cd)
        if not params.solver:
            m = re.search(r"application\s+(\w+)\s*;", cd)
            if m:
                params.solver = m.group(1)

    if params.re_from_summary and params.nu and params.inlet_velocity:
        params.characteristic_length = (
            params.re_from_summary * params.nu / params.inlet_velocity
        )

    return params


def reference_hint_from_params(params: ReferenceCaseParams) -> ReferenceHint:
    """ReferenceCaseParams → RequirementProfile 用の軽量ヒント。"""
    return ReferenceHint(
        case_id=params.case_id,
        title_ja=params.title_ja,
        inlet_velocity=params.inlet_velocity,
        nu=params.nu,
        turbulence_model=params.turbulence_model,
        solver=params.solver,
        steady_state=params.steady_state,
        characteristic_length=params.characteristic_length,
        re_number=params.re_number,
    )


def params_differ_significantly(
    spec: SimulationSpec,
    ref: ReferenceCaseParams,
    rel_tol: float = 0.15,
) -> bool:
    """spec と参照典型値に有意な差があるか。"""
    if ref.inlet_velocity is not None and spec.inlet_velocity:
        if abs(ref.inlet_velocity - spec.inlet_velocity) / max(spec.inlet_velocity, 1e-9) > rel_tol:
            return True
    if ref.nu is not None and spec.nu:
        if abs(ref.nu - spec.nu) / spec.nu > rel_tol:
            return True
    if ref.turbulence_model and ref.turbulence_model not in ("unknown", "RAS"):
        if ref.turbulence_model != spec.turbulence_model:
            return True
    if ref.re_number and spec.re_number:
        if abs(ref.re_number - spec.re_number) / max(spec.re_number, 1) > rel_tol:
            return True
    return False

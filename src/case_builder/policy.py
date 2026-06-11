"""決定的ポリシー: ソルバー選択、Re 整合、時間設定。"""
from __future__ import annotations

import re

from ..models import SimulationSpec
from .mesh_metrics import (
    DEFAULT_MAX_CO,
    compute_delta_t,
    compute_max_delta_t,
    estimate_min_cell_length,
)

DEFAULT_RE_VELOCITY = 1.0  # m/s — Re のみ指定時の実務デフォルト

_VELOCITY_PATTERNS = (
    r"流入速度",
    r"来流速度",
    r"来流",
    r"流速",
    r"velocity",
    r"inlet\s*velocity",
    r"\bU\s*[=:：]",
    r"[\d.]+\s*m\s*/\s*s",
    r"[\d.]+\s*m/s",
)


def parse_re_from_description(description: str) -> float | None:
    """説明文からレイノルズ数を抽出。"""
    m = re.search(r"Re\s*[=:：]?\s*([\d.eE+\-]+)", description, re.IGNORECASE)
    if not m:
        m = re.search(r"レイノルズ数\s*[=:：]?\s*([\d.eE+\-]+)", description)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


def velocity_mentioned_in(description: str) -> bool:
    """説明文に流速の明示指定があるか。"""
    return any(re.search(pat, description, re.IGNORECASE) for pat in _VELOCITY_PATTERNS)


def nu_from_re(velocity: float, length: float, re_number: float) -> float:
    """Re = U·L/ν より ν を算出。"""
    return velocity * length / re_number


def apply_solver_policy(spec: SimulationSpec) -> SimulationSpec:
    """現象・ケースタイプに応じたソルバー上書き。"""
    if spec.phenomenon == "karman_vortex_shedding" or spec.case_type == "cylinder_2d_ogrid":
        spec.solver = "pimpleFoam"
        spec.steady_state = False
    return spec


def reconcile_re(
    spec: SimulationSpec,
    target_re: float | None,
    description: str = "",
) -> SimulationSpec:
    """
    Re 入力を最優先して U / ν を整合させる。

    - Re のみ（流速未指定）: U=1 m/s 固定 → ν = U·L/Re
    - Re + 流速明示: U を維持 → ν = U·L/Re
    """
    if target_re is None or not spec.characteristic_length:
        return spec

    length = spec.characteristic_length
    applied = set(spec.defaults_applied or [])
    u_explicit = (
        velocity_mentioned_in(description)
        or "confirmed_inlet_velocity" in applied
    )

    if u_explicit:
        velocity = spec.inlet_velocity
        spec.nu = nu_from_re(velocity, length, target_re)
    else:
        spec.inlet_velocity = DEFAULT_RE_VELOCITY
        spec.nu = nu_from_re(DEFAULT_RE_VELOCITY, length, target_re)
        for tag in ("re_policy_inlet_velocity", "re_policy_nu"):
            if tag not in spec.defaults_applied:
                spec.defaults_applied.append(tag)

    spec.re_number = target_re
    return spec


def compute_time_settings(
    spec: SimulationSpec,
    min_cell_length: float | None = None,
    mesh_params: dict | None = None,
) -> dict:
    """controlDict 用の endTime, deltaT, writeInterval, purgeWrite。"""
    char_len = spec.characteristic_length or 1.0
    u = max(spec.inlet_velocity, 1e-6)

    if spec.steady_state:
        return {"end_time": 1000.0, "delta_t": 1.0, "write_interval": 100.0, "purge_write": 0}

    if min_cell_length is None:
        min_cell_length = estimate_min_cell_length(
            spec, mesh_params or spec.mesh_params
        )
    delta_t = compute_delta_t(min_cell_length, u)
    max_delta_t = compute_max_delta_t(delta_t)

    if spec.phenomenon == "karman_vortex_shedding" and spec.case_type == "cylinder_2d_ogrid":
        st = 0.2
        shed_period = char_len / (st * u)
        raw_periods = spec.mesh_params.get("karman_periods")
        if raw_periods is not None:
            n_periods = int(raw_periods)
        elif bool(spec.mesh_params.get("demo_mode")):
            n_periods = 5
        else:
            n_periods = 25
        return {
            "end_time": round(shed_period * n_periods, 2),
            "delta_t": delta_t,
            "max_delta_t": max_delta_t,
            "min_cell_length": min_cell_length,
            "write_interval": round(shed_period / 20, 4),
            "purge_write": 0,
            "write_control": "runTime",
        }

    flow_through = char_len / u
    return {
        "end_time": round(flow_through * 10, 4),
        "delta_t": delta_t,
        "max_delta_t": max_delta_t,
        "min_cell_length": min_cell_length,
        "write_interval": round(flow_through / 10, 5),
        "purge_write": 5,
        "write_control": "timeStep",
    }


def read_patch_names(case_dir: str) -> list[str]:
    """polyMesh/boundary からパッチ名一覧を取得。"""
    from pathlib import Path
    text = (Path(case_dir) / "constant" / "polyMesh" / "boundary").read_text(errors="ignore")
    return re.findall(r"^\s{4}(\w+)\s*$", text, re.MULTILINE)

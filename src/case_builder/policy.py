"""決定的ポリシー: ソルバー選択、Re 整合、時間設定。"""
from __future__ import annotations

import re

from ..models import SimulationSpec


def apply_solver_policy(spec: SimulationSpec) -> SimulationSpec:
    """現象・ケースタイプに応じたソルバー上書き。"""
    if spec.phenomenon == "karman_vortex_shedding" or spec.case_type == "cylinder_2d_ogrid":
        spec.solver = "pimpleFoam"
        spec.steady_state = False
    return spec


def reconcile_re(spec: SimulationSpec, target_re: float | None) -> SimulationSpec:
    """Re 入力を最優先し、inlet_velocity を調整。"""
    if target_re is None or not spec.nu or not spec.characteristic_length:
        return spec
    spec.inlet_velocity = target_re * spec.nu / spec.characteristic_length
    spec.re_number = target_re
    return spec


def compute_time_settings(spec: SimulationSpec) -> dict:
    """controlDict 用の endTime, deltaT, writeInterval, purgeWrite。"""
    char_len = spec.characteristic_length or 1.0
    u = max(spec.inlet_velocity, 1e-6)

    if spec.steady_state:
        return {"end_time": 1000.0, "delta_t": 1.0, "write_interval": 100.0, "purge_write": 0}

    if spec.phenomenon == "karman_vortex_shedding" and spec.case_type == "cylinder_2d_ogrid":
        st = 0.2
        shed_period = char_len / (st * u)
        return {
            "end_time": round(shed_period * 25, 2),
            "delta_t": round(char_len * 0.007 * 0.1 / max(u * 2, 0.01), 6),
            "write_interval": round(shed_period / 20, 4),
            "purge_write": 0,
            "write_control": "runTime",
        }

    flow_through = char_len / u
    min_cell = char_len * 0.007 if spec.case_type == "cylinder_2d_ogrid" else char_len / 500
    return {
        "end_time": round(flow_through * 10, 4),
        "delta_t": round(min_cell * 0.1 / max(u * 2, 0.01), 6),
        "write_interval": round(flow_through / 10, 5),
        "purge_write": 5,
        "write_control": "timeStep",
    }


def read_patch_names(case_dir: str) -> list[str]:
    """polyMesh/boundary からパッチ名一覧を取得。"""
    from pathlib import Path
    text = (Path(case_dir) / "constant" / "polyMesh" / "boundary").read_text(errors="ignore")
    return re.findall(r"^\s{4}(\w+)\s*$", text, re.MULTILINE)

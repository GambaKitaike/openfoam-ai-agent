"""メッシュから代表セルサイズ・Δt を算出する。"""
from __future__ import annotations

import re
from pathlib import Path

from ..models import SimulationSpec

DEFAULT_MAX_CO = 0.5
MAX_DELTA_T_FACTOR = 100.0

# デフォルト O-grid (n_r=15, gr_r=0.05) で checkMesh min face area ≈ (0.007×L)²
OGRID_MIN_CELL_RATIO = 0.007


def parse_min_cell_length_from_checkmesh(log_path: str | Path) -> float | None:
    """
    checkMesh ログの Minimum face area から代表最小セル長を推定。

    2D/薄い 3D メッシュでは sqrt(minFaceArea) を Δx の目安とする。
    """
    path = Path(log_path)
    if not path.exists():
        return None
    text = path.read_text(errors="ignore")
    m = re.search(r"Minimum face area = ([0-9.eE+\-]+)\.", text)
    if not m:
        return None
    try:
        area = float(m.group(1))
    except ValueError:
        return None
    if area <= 0:
        return None
    return area ** 0.5


def estimate_min_cell_length(
    spec: SimulationSpec,
    mesh_params: dict | None = None,
) -> float:
    """blockMesh 前の Δt 見積もり用（メッシュ未生成時）。"""
    char_len = spec.characteristic_length or 1.0
    if spec.case_type == "cylinder_2d_ogrid":
        return char_len * OGRID_MIN_CELL_RATIO

    params = mesh_params or {}
    nx = max(int(params.get("nx", 40)), 1)
    ny = max(int(params.get("ny", 20)), 1)
    lx = float(params.get("lx", 20.0))
    ly = float(params.get("ly", 10.0))
    return min(lx / nx, ly / ny)


def compute_delta_t(
    min_cell_length: float,
    velocity: float,
    max_co: float = DEFAULT_MAX_CO,
) -> float:
    """Courant 数 max_co に基づく初期 deltaT。"""
    return round(max_co * min_cell_length / max(velocity, 1e-6), 6)


def compute_max_delta_t(
    delta_t: float,
    factor: float = MAX_DELTA_T_FACTOR,
) -> float:
    """adjustTimeStep 用 maxDeltaT 上限。"""
    return round(delta_t * factor, 6)


def patch_control_dict_timestep(
    case_dir: str | Path,
    delta_t: float,
    max_delta_t: float | None = None,
) -> None:
    """system/controlDict の deltaT / maxDeltaT を更新。"""
    path = Path(case_dir) / "system" / "controlDict"
    text = path.read_text()
    text = re.sub(
        r"deltaT\s+[\d.eE+-]+\s*;",
        f"deltaT          {delta_t:g};",
        text,
        count=1,
    )
    if max_delta_t is not None and re.search(r"maxDeltaT\s+[\d.eE+-]+\s*;", text):
        text = re.sub(
            r"maxDeltaT\s+[\d.eE+-]+\s*;",
            f"maxDeltaT       {max_delta_t:g};",
            text,
            count=1,
        )
    path.write_text(text)


def apply_mesh_linked_timestep(
    case_dir: str | Path,
    spec: SimulationSpec,
    *,
    checkmesh_log: str | Path | None = None,
    mesh_params: dict | None = None,
) -> tuple[float, float, str]:
    """
    メッシュ指標から deltaT を決め controlDict を更新する。

    Returns:
        (min_cell_length, delta_t, source)  source は "checkMesh" | "estimate"
    """
    log = Path(checkmesh_log) if checkmesh_log else Path(case_dir) / "log.checkMesh"
    min_cell = parse_min_cell_length_from_checkmesh(log)
    source = "checkMesh"
    if min_cell is None:
        min_cell = estimate_min_cell_length(spec, mesh_params)
        source = "estimate"

    delta_t = compute_delta_t(min_cell, spec.inlet_velocity)
    max_dt = compute_max_delta_t(delta_t)
    patch_control_dict_timestep(case_dir, delta_t, max_dt)

    spec.mesh_params["min_cell_length"] = min_cell
    spec.mesh_params["delta_t"] = delta_t
    return min_cell, delta_t, source

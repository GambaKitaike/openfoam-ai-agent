"""mesh_metrics のユニットテスト。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.case_builder.mesh_metrics import (
    apply_mesh_linked_timestep,
    compute_delta_t,
    parse_min_cell_length_from_checkmesh,
)
from src.models import SimulationSpec


def _karman_spec(**overrides) -> SimulationSpec:
    defaults = dict(
        solver="pimpleFoam",
        case_type="cylinder_2d_ogrid",
        mesh_template="ogrid_cylinder_2d",
        turbulence_model="laminar",
        steady_state=False,
        inlet_velocity=1.0,
        dimensions=2,
        characteristic_length=1.0,
        nu=0.001,
        phenomenon="karman_vortex_shedding",
    )
    defaults.update(overrides)
    return SimulationSpec(**defaults)


class TestMeshMetrics:
    def test_parse_checkmesh_log(self, tmp_path: Path):
        log = tmp_path / "log.checkMesh"
        log.write_text("    Minimum face area = 5.01849e-05. Maximum face area = 0.479708.\n")
        dx = parse_min_cell_length_from_checkmesh(log)
        assert dx == pytest.approx(0.007083, rel=1e-3)

    def test_cfl_delta_t(self):
        # maxCo=0.5, dx=0.007083, U=1 → deltaT≈0.00354
        dt = compute_delta_t(0.007083, 1.0)
        assert dt == pytest.approx(0.003542, rel=1e-3)

    def test_apply_mesh_linked_timestep(self, tmp_path: Path):
        case = tmp_path / "case"
        (case / "system").mkdir(parents=True)
        (case / "system" / "controlDict").write_text(
            "deltaT 0.0001;\nadjustTimeStep yes;\nmaxDeltaT 0.01;\n"
        )
        (case / "log.checkMesh").write_text(
            "Minimum face area = 5.01849e-05. Maximum face area = 0.1.\n"
        )
        spec = _karman_spec()
        min_cell, delta_t, source = apply_mesh_linked_timestep(case, spec)
        assert source == "checkMesh"
        assert min_cell == pytest.approx(0.007083, rel=1e-3)
        assert delta_t == pytest.approx(0.003542, rel=1e-3)
        text = (case / "system" / "controlDict").read_text()
        assert "deltaT          0.0035415" in text or "0.003542" in text

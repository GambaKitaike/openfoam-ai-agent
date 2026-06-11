"""policy 層のユニットテスト。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.case_builder.policy import apply_solver_policy, compute_time_settings, reconcile_re
from src.models import SimulationSpec


def _spec(**overrides) -> SimulationSpec:
    defaults = dict(
        solver="icoFoam",
        case_type="cylinder_2d_ogrid",
        mesh_template="ogrid_cylinder_2d",
        turbulence_model="laminar",
        steady_state=True,
        inlet_velocity=1.0,
        dimensions=2,
        characteristic_length=1.0,
        nu=0.01,
        phenomenon="karman_vortex_shedding",
    )
    defaults.update(overrides)
    return SimulationSpec(**defaults)


class TestPolicy:
    def test_karman_forces_pimplefoam(self):
        spec = _spec()
        apply_solver_policy(spec)
        assert spec.solver == "pimpleFoam"
        assert spec.steady_state is False

    def test_reconcile_re_adjusts_velocity(self):
        spec = _spec(inlet_velocity=0.1, nu=0.01, characteristic_length=1.0)
        reconcile_re(spec, 100.0)
        assert spec.inlet_velocity == pytest.approx(1.0)
        assert spec.re_number == pytest.approx(100.0)

    def test_karman_time_settings(self):
        ts = compute_time_settings(_spec(steady_state=False))
        assert ts["purge_write"] == 0
        assert ts["write_control"] == "runTime"
        assert ts["end_time"] == pytest.approx(125.0, rel=0.05)

    def test_karman_demo_time_settings(self):
        spec = _spec(steady_state=False, mesh_params={"demo_mode": True})
        ts = compute_time_settings(spec)
        assert ts["end_time"] == pytest.approx(25.0, rel=0.05)

    def test_karman_custom_periods(self):
        spec = _spec(steady_state=False, mesh_params={"karman_periods": 10})
        ts = compute_time_settings(spec)
        assert ts["end_time"] == pytest.approx(50.0, rel=0.05)

    def test_periods_overrides_demo(self):
        spec = _spec(
            steady_state=False,
            mesh_params={"demo_mode": True, "karman_periods": 3},
        )
        ts = compute_time_settings(spec)
        assert ts["end_time"] == pytest.approx(15.0, rel=0.05)

    def test_steady_time_settings(self):
        ts = compute_time_settings(_spec(steady_state=True, solver="simpleFoam"))
        assert ts["end_time"] == 1000.0

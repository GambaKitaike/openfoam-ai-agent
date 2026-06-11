"""決定的 OpenFOAM ビルダーのテスト。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.case_builder import builders
from src.models import SimulationSpec


def _spec(**overrides) -> SimulationSpec:
    defaults = dict(
        solver="simpleFoam",
        case_type="channel_2d",
        mesh_template="box_channel_2d",
        turbulence_model="kOmegaSST",
        steady_state=True,
        inlet_velocity=1.0,
        dimensions=2,
        characteristic_length=1.0,
        nu=1.5e-5,
    )
    defaults.update(overrides)
    return SimulationSpec(**defaults)


class TestBuilders:
    def test_transport_has_nu(self):
        out = builders.build_transport_properties(_spec(nu=0.001))
        assert "nu              0.001" in out

    def test_fv_solution_steady_has_simple(self):
        out = builders.build_fv_solution(_spec(steady_state=True))
        assert "SIMPLE" in out

    def test_fv_solution_ogrid_pimple(self):
        out = builders.build_fv_solution(_spec(
            steady_state=False,
            case_type="cylinder_2d_ogrid",
            solver="pimpleFoam",
        ))
        assert "PIMPLE" in out
        assert "PISO" not in out.split("PIMPLE")[0]

    def test_control_dict_karman_runtime_write(self):
        out = builders.build_control_dict(_spec(
            steady_state=False,
            case_type="cylinder_2d_ogrid",
            phenomenon="karman_vortex_shedding",
            solver="pimpleFoam",
            inlet_velocity=1.0,
            characteristic_length=1.0,
        ))
        assert "writeControl    runTime" in out
        assert "purgeWrite      0" in out

    def test_control_dict_karman_uses_adjust_time_step(self):
        out = builders.build_control_dict(_spec(
            steady_state=False,
            case_type="cylinder_2d_ogrid",
            phenomenon="karman_vortex_shedding",
            solver="pimpleFoam",
            inlet_velocity=1.0,
            characteristic_length=1.0,
        ))
        assert "adjustTimeStep  yes;" in out
        assert "maxCo           0.5;" in out
        assert "maxDeltaT" in out
        # maxCo * dx / U, dx≈0.007*L
        assert "deltaT          0.0035" in out

    def test_u_field_ogrid_patches(self):
        patches = ["inlet", "outlet", "top", "bottom", "cylinder", "frontAndBack"]
        out = builders.build_u_field(_spec(case_type="cylinder_2d_ogrid"), patches)
        assert "cylinder { type noSlip; }" in out
        assert "frontAndBack { type empty; }" in out
        assert "top { type slip; }" in out
        assert "bottom { type slip; }" in out
        assert "symmetryPlane" not in out
        assert "freestream" not in out

    def test_p_field_ogrid_slip_walls(self):
        patches = ["inlet", "outlet", "top", "bottom", "cylinder", "frontAndBack"]
        out = builders.build_p_field(_spec(case_type="cylinder_2d_ogrid"), patches)
        assert "top { type zeroGradient; }" in out
        assert "bottom { type zeroGradient; }" in out

    def test_set_fields_dict_karman(self):
        out = builders.build_set_fields_dict(_spec(
            inlet_velocity=1.0,
            characteristic_length=1.0,
        ))
        assert "setFieldsDict" in out
        assert "boxToCell" in out

    def test_decompose_par_dict(self):
        out = builders.build_decompose_par_dict(4)
        assert "numberOfSubdomains 4" in out
        assert "(2 2 1)" in out

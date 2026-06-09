"""preprocessing の phenomenon 正規化テスト"""
from __future__ import annotations

from src.agents.preprocessing import PreprocessingAgent


def test_normalize_phenomenon_karman():
    assert PreprocessingAgent._normalize_phenomenon("", "2D円柱周りのカルマン渦 Re=100") == "karman_vortex_shedding"


def test_normalize_phenomenon_airfoil():
    assert PreprocessingAgent._normalize_phenomenon("", "2D翼の定常流れと揚力") == "airfoil_steady"


def test_normalize_phenomenon_passthrough():
    assert PreprocessingAgent._normalize_phenomenon("backward_facing_step", "") == "backward_facing_step"


def test_karman_ogrid_forces_pimplefoam_over_icofoam():
    agent = PreprocessingAgent.__new__(PreprocessingAgent)
    data = {
        "solver": "icoFoam",
        "case_type": "cylinder_2d_ogrid",
        "dimensions": 2,
        "turbulence_model": "laminar",
        "steady_state": False,
        "inlet_velocity": 1.0,
        "characteristic_length": 0.1,
        "nu": 0.01,
        "phenomenon": "karman_vortex_shedding",
    }
    spec = agent._build_spec(data, "2D円柱 Re=100 層流")
    assert spec.solver == "pimpleFoam"
    assert spec.steady_state is False

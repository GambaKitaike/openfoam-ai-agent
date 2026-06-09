"""clarify_with_profile のユニットテスト。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.spec_clarification import clarify_with_profile
from src.models import RequirementProfile, SimulationSpec
from src.rag.requirement_profile import build_requirement_profile


def _spec(**overrides) -> SimulationSpec:
    defaults = dict(
        solver="pimpleFoam",
        case_type="cylinder_2d_ogrid",
        mesh_template="ogrid_cylinder_2d",
        turbulence_model="laminar",
        steady_state=False,
        inlet_velocity=1.0,
        dimensions=2,
        characteristic_length=1.0,
        nu=0.01,
        phenomenon="karman_vortex_shedding",
        defaults_applied=["inlet_velocity", "nu"],
        description="2D円柱周りのカルマン渦 Re=100 層流 流入速度1m/s",
    )
    defaults.update(overrides)
    return SimulationSpec(**defaults)


class TestClarifyWithProfile:
    def test_no_interactive_karman_re100(self):
        spec = _spec()
        description = "2D円柱周りのカルマン渦 Re=100 層流 流入速度1m/s"
        profile = build_requirement_profile(spec, description)
        result = clarify_with_profile(spec, profile, description, interactive=False)
        assert result.inlet_velocity == 1.0
        assert result.nu == 0.01
        assert result.re_number == 100

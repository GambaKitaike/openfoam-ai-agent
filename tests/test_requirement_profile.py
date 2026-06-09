"""requirement_profile のユニットテスト。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import SimulationSpec
from src.rag.requirement_profile import apply_profile_defaults, build_requirement_profile


def _spec(**overrides) -> SimulationSpec:
    defaults = dict(
        solver="icoFoam",
        case_type="channel_2d",
        mesh_template="box_channel_2d",
        turbulence_model="laminar",
        steady_state=True,
        inlet_velocity=0.1,
        dimensions=2,
        characteristic_length=1.0,
        nu=0.01,
        phenomenon="karman_vortex_shedding",
        defaults_applied=[],
    )
    defaults.update(overrides)
    return SimulationSpec(**defaults)


class TestRequirementProfile:
    def test_karman_lists_missing_fields(self):
        profile = build_requirement_profile(_spec(), "Re=100 カルマン渦")
        keys = {f.key for f in profile.fields}
        assert "inlet_velocity" in keys or "steady_state" in keys

    def test_re_priority_suggests_velocity(self):
        profile = build_requirement_profile(
            _spec(inlet_velocity=0.1, defaults_applied=["inlet_velocity"]),
            "Re=100",
        )
        u_field = next((f for f in profile.fields if f.key == "inlet_velocity"), None)
        if u_field:
            assert u_field.suggested == pytest.approx(1.0)

    def test_apply_defaults_sets_ogrid(self):
        spec = _spec()
        profile = build_requirement_profile(spec, "Re=100")
        apply_profile_defaults(spec, profile)
        assert spec.case_type == "cylinder_2d_ogrid"
        assert spec.solver == "pimpleFoam"

    def test_karman_re100_complete_spec_has_no_missing_fields(self):
        """Re=100 + U=1 + nu=0.01 が揃っているときは未充足項目なし。"""
        spec = _spec(
            solver="pimpleFoam",
            steady_state=False,
            inlet_velocity=1.0,
            nu=0.01,
            characteristic_length=1.0,
            defaults_applied=["inlet_velocity", "nu"],
        )
        profile = build_requirement_profile(
            spec,
            "2D円柱周りのカルマン渦 Re=100 層流 流入速度1m/s",
        )
        assert profile.fields == []

"""requirement_profile のユニットテスト。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import ReferenceHint, SimulationSpec
from src.rag.requirement_profile import apply_profile_defaults, build_requirement_profile, review_spec


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


def _hints(**kwargs) -> list[ReferenceHint]:
    return [ReferenceHint(
        case_id="incompressible/pimpleFoam/vortexShed",
        title_ja="2D渦放出",
        inlet_velocity=kwargs.get("inlet_velocity", 0.15),
        nu=kwargs.get("nu", 1e-3),
        turbulence_model=kwargs.get("turbulence_model", "laminar"),
        solver="pimpleFoam",
        steady_state=False,
        characteristic_length=0.1,
        re_number=kwargs.get("re_number", 150.0),
    )]


class TestRequirementProfile:
    def test_karman_lists_missing_fields(self):
        profile = build_requirement_profile(_spec(), "Re=100 カルマン渦")
        keys = {f.key for f in profile.fields}
        assert "inlet_velocity" in keys or "steady_state" in keys

    def test_re_only_suggests_nu_from_re(self):
        profile = build_requirement_profile(
            _spec(inlet_velocity=0.1, nu=1.5e-5),
            "Re=1000",
        )
        nu_field = next((f for f in profile.fields if f.key == "nu"), None)
        assert nu_field is not None
        assert nu_field.suggested == pytest.approx(0.001)

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

    def test_reference_hints_preferred_for_steady_state(self):
        profile = build_requirement_profile(
            _spec(steady_state=True, solver="icoFoam"),
            "カルマン渦",
            reference_hints=_hints(),
        )
        ss_field = next((f for f in profile.fields if f.key == "steady_state"), None)
        assert ss_field is not None
        assert ss_field.suggested is False
        assert "類似チュートリアル" in ss_field.reason

    def test_reference_hints_in_constraints(self):
        profile = build_requirement_profile(
            _spec(),
            "カルマン渦",
            similar_case_ids=["incompressible/pimpleFoam/vortexShed"],
            reference_hints=_hints(),
        )
        assert profile.similar_case_ids == ["incompressible/pimpleFoam/vortexShed"]
        assert any("参照典型値" in c for c in profile.constraints)

    def test_review_warns_when_differs_from_reference(self):
        spec = _spec(
            solver="pimpleFoam",
            steady_state=False,
            inlet_velocity=2.0,
            nu=0.001,
            characteristic_length=1.0,
        )
        profile = build_requirement_profile(
            spec,
            "カルマン渦",
            reference_hints=_hints(inlet_velocity=0.15, nu=1e-3),
        )
        issues = review_spec(spec, profile, "カルマン渦")
        assert any("類似チュートリアル" in i.message for i in issues)

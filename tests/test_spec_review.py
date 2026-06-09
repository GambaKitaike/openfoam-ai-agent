"""Agent② spec レビューのユニットテスト。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import SimulationSpec
from src.rag.requirement_profile import (
    LAMINAR_RE_LIMIT,
    apply_review_fixes,
    build_requirement_profile,
    review_spec,
)


def _spec(**overrides) -> SimulationSpec:
    defaults = dict(
        solver="simpleFoam",
        case_type="channel_2d",
        mesh_template="box_channel_2d",
        turbulence_model="laminar",
        steady_state=True,
        inlet_velocity=1.0,
        dimensions=2,
        characteristic_length=1.0,
        nu=1.5e-5,
        phenomenon="channel_internal",
    )
    defaults.update(overrides)
    return SimulationSpec(**defaults)


class TestReviewSpec:
    def test_high_re_laminar_flags_issue(self):
        spec = _spec()
        profile = build_requirement_profile(spec, "")
        issues = review_spec(spec, profile, "")
        assert any(i.key == "turbulence_model" for i in issues)

    def test_high_re_laminar_flags_issue(self):
        spec = _spec()
        profile = build_requirement_profile(spec, "")
        issues = review_spec(spec, profile, "")
        assert any(i.key == "turbulence_model" for i in issues)
        turb = next(i for i in issues if i.key == "turbulence_model")
        assert turb.alternatives
        assert turb.alternatives[0].key == "inlet_velocity"

    def test_high_re_laminar_user_prefers_velocity_reduction(self):
        spec = _spec()
        profile = build_requirement_profile(spec, "層流")
        issues = review_spec(spec, profile, "2D 層流 channel")
        vel = next(i for i in issues if i.key == "inlet_velocity")
        assert vel.suggested == spec.nu * LAMINAR_RE_LIMIT / spec.characteristic_length
        apply_review_fixes(spec, issues, respect_user_lock=True)
        assert spec.turbulence_model == "laminar"
        assert spec.re_number == pytest.approx(LAMINAR_RE_LIMIT)

    def test_high_re_laminar_locked_when_explicit_re_in_description(self):
        spec = _spec()
        profile = build_requirement_profile(spec, "Re=50000 層流")
        issues = review_spec(spec, profile, "Re=50000 層流 channel")
        turb = next(i for i in issues if i.key == "turbulence_model")
        assert turb.user_locked is True

    def test_auto_fix_turbulence_when_not_user_locked(self):
        spec = _spec()
        profile = build_requirement_profile(spec, "")
        issues = review_spec(spec, profile, "")
        apply_review_fixes(spec, issues, respect_user_lock=True)
        assert spec.turbulence_model == "kOmegaSST"

    def test_karman_re100_laminar_ok(self):
        spec = _spec(
            solver="pimpleFoam",
            steady_state=False,
            case_type="cylinder_2d_ogrid",
            phenomenon="karman_vortex_shedding",
            nu=0.01,
            inlet_velocity=1.0,
            re_number=100.0,
        )
        profile = build_requirement_profile(spec, "Re=100 層流")
        issues = review_spec(spec, profile, "Re=100 層流")
        assert not any(i.key == "turbulence_model" for i in issues)

"""spec_clarification のユニットテスト"""
from __future__ import annotations

from src.agents.spec_clarification import (
    apply_auto_fixes,
    collect_clarifications,
    _parse_re_from_description,
)
from src.models import SimulationSpec


def _airfoil_spec(**kwargs) -> SimulationSpec:
    defaults = {
        "solver": "simpleFoam",
        "case_type": "snappy_2d",
        "mesh_template": "box_snappy_2d",
        "turbulence_model": "kOmegaSST",
        "steady_state": True,
        "inlet_velocity": 20.0,
        "dimensions": 2,
        "characteristic_length": 1.0,
        "nu": 1.5e-5,
        "phenomenon": "airfoil_steady",
        "defaults_applied": ["inlet_velocity", "turbulence_model", "characteristic_length", "nu"],
    }
    defaults.update(kwargs)
    return SimulationSpec(**defaults)


def test_parse_re_from_description():
    assert _parse_re_from_description("Re=100 層流") == 100.0
    assert _parse_re_from_description("レイノルズ数：5000") == 5000.0


def test_collect_clarifications_high_re_airfoil():
    spec = _airfoil_spec()
    fields = collect_clarifications(spec, "2D翼周りの定常流れ simpleFoam")
    keys = {f.key for f in fields}
    assert "inlet_velocity" in keys
    assert spec.re_number and spec.re_number > 100_000


def test_apply_auto_fixes_reduces_re():
    spec = _airfoil_spec()
    fields = collect_clarifications(spec, "2D翼周りの定常流れ simpleFoam")
    apply_auto_fixes(spec, fields)
    assert spec.re_number is not None
    assert spec.re_number < 2_000_000
    assert spec.inlet_velocity < 20.0


def test_re_only_suggests_unit_velocity():
    spec = _airfoil_spec(inlet_velocity=10.0, characteristic_length=0.1)
    fields = collect_clarifications(spec, "2D翼 Re=100000 定常")
    vel_field = next(f for f in fields if f.key == "inlet_velocity")
    assert vel_field.suggested == 1.0

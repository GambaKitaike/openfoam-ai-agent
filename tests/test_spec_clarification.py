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
    fields, _warnings = collect_clarifications(spec, "2D翼周りの定常流れ simpleFoam")
    keys = {f.key for f in fields}
    assert "inlet_velocity" in keys
    assert spec.re_number and spec.re_number > 100_000


def test_apply_auto_fixes_reduces_re():
    spec = _airfoil_spec()
    fields, _warnings = collect_clarifications(spec, "2D翼周りの定常流れ simpleFoam")
    apply_auto_fixes(spec, fields)
    assert spec.re_number is not None
    assert spec.re_number < 2_000_000
    assert spec.inlet_velocity < 20.0


def test_re_only_suggests_unit_velocity():
    spec = _airfoil_spec(inlet_velocity=10.0, characteristic_length=0.1)
    fields, _warnings = collect_clarifications(spec, "2D翼 Re=100000 定常")
    vel_field = next(f for f in fields if f.key == "inlet_velocity")
    assert vel_field.suggested == 1.0


def test_laminar_high_re_keeps_user_velocity_and_warns():
    spec = SimulationSpec(
        solver="pimpleFoam",
        case_type="cylinder_2d_ogrid",
        mesh_template="ogrid_cylinder_2d",
        turbulence_model="laminar",
        steady_state=False,
        inlet_velocity=0.15,
        dimensions=2,
        characteristic_length=1.0,
        nu=1.5e-5,
        phenomenon="karman_vortex_shedding",
    )
    description = "流入0.15m/s、層流"
    fields, warnings = collect_clarifications(spec, description)
    apply_auto_fixes(spec, fields)
    assert spec.inlet_velocity == 0.15
    assert not any(f.key == "inlet_velocity" for f in fields)
    assert any("発散" in w or "Re" in w for w in warnings)


def test_clarify_from_reference_non_interactive_does_not_overwrite():
    from src.agents.spec_clarification import clarify_from_reference
    from src.models import EnrichedContext
    from src.rag.reference_case_params import ReferenceCaseParams

    spec = SimulationSpec(
        solver="pimpleFoam",
        case_type="cylinder_2d_ogrid",
        mesh_template="ogrid_cylinder_2d",
        turbulence_model="laminar",
        steady_state=False,
        inlet_velocity=0.15,
        dimensions=2,
        characteristic_length=0.1,
        nu=1.5e-5,
        phenomenon="karman_vortex_shedding",
    )
    ref = ReferenceCaseParams(
        case_id="tutorials/incompressible/pimpleFoam/cylinder2D",
        title_ja="2D円柱",
        inlet_velocity=1.0,
        nu=1.5e-5,
        turbulence_model="laminar",
    )
    context = EnrichedContext(
        spec=spec,
        reference_case_id=ref.case_id,
        reference_typical_params=ref,
        reference_phenomenon="karman_vortex_shedding",
    )
    updated, warnings = clarify_from_reference(spec, context, interactive=False)
    assert updated.inlet_velocity == 0.15
    assert warnings


def test_clarify_from_reference_interactive_applies_on_yes(monkeypatch):
    from src.agents.spec_clarification import clarify_from_reference
    from src.models import EnrichedContext
    from src.rag.reference_case_params import ReferenceCaseParams

    spec = SimulationSpec(
        solver="pimpleFoam",
        case_type="cylinder_2d_ogrid",
        mesh_template="ogrid_cylinder_2d",
        turbulence_model="laminar",
        steady_state=False,
        inlet_velocity=0.15,
        dimensions=2,
        characteristic_length=0.1,
        nu=1.5e-5,
        phenomenon="karman_vortex_shedding",
    )
    ref = ReferenceCaseParams(
        case_id="tutorials/incompressible/pimpleFoam/cylinder2D",
        title_ja="2D円柱",
        inlet_velocity=1.0,
        nu=1.5e-5,
        turbulence_model="laminar",
    )
    context = EnrichedContext(
        spec=spec,
        reference_case_id=ref.case_id,
        reference_typical_params=ref,
        reference_phenomenon="karman_vortex_shedding",
    )
    monkeypatch.setattr(
        "src.agents.spec_clarification.Prompt.ask",
        lambda *args, **kwargs: "はい",
    )
    updated, warnings = clarify_from_reference(spec, context, interactive=True)
    assert updated.inlet_velocity == 1.0
    assert warnings == []

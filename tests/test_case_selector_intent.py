"""case_selector の phenomenon フィルタテスト"""
from __future__ import annotations

from src.models import SimulationSpec
from src.rag.case_selector import (
    CaseSelector,
    _has_usable_mesh,
    _turbulence_compatible,
)


def _meta(**kwargs):
    base = {
        "case_id": "incompressible/simpleFoam/pitzDaily",
        "case_path": "/tmp/pitzDaily",
        "solver": "simpleFoam",
        "steady_state": True,
        "dimensions": 2,
        "turbulence_model": "kEpsilon",
        "category": "incompressible",
        "requires_preprocessing": False,
        "has_blockmesh": True,
        "has_snappy": False,
        "phenomenon": "backward_facing_step",
    }
    base.update(kwargs)
    return base


def test_mesh_prebuilt_passes_hard_filter():
    selector = CaseSelector.__new__(CaseSelector)
    spec = SimulationSpec(
        solver="simpleFoam",
        case_type="channel_2d",
        mesh_template="box_channel_2d",
        turbulence_model="kOmegaSST",
        steady_state=True,
        inlet_velocity=20.0,
        dimensions=2,
        phenomenon="airfoil_steady",
    )
    meta = _meta(
        case_id="incompressible/simpleFoam/airFoil2D",
        has_blockmesh=False,
        has_snappy=False,
        mesh_prebuilt=True,
        turbulence_model="SpalartAllmaras",
        phenomenon="airfoil_steady",
    )
    assert _has_usable_mesh(meta)
    assert _turbulence_compatible("kOmegaSST", "SpalartAllmaras")
    assert selector._passes_hard_filter(spec, meta)


def test_ras_models_mutually_compatible():
    assert _turbulence_compatible("kOmegaSST", "SpalartAllmaras")
    assert _turbulence_compatible("SpalartAllmaras", "kOmegaSST")
    assert not _turbulence_compatible("LES", "kOmegaSST")


def test_phenomenon_filter_rejects_mismatch():
    selector = CaseSelector.__new__(CaseSelector)
    spec = SimulationSpec(
        solver="simpleFoam",
        case_type="channel_2d",
        mesh_template="box_channel_2d",
        turbulence_model="kEpsilon",
        steady_state=True,
        inlet_velocity=10.0,
        dimensions=2,
        phenomenon="karman_vortex_shedding",
    )
    assert not selector._passes_hard_filter(spec, _meta(phenomenon="airfoil_steady"))


def test_phenomenon_filter_accepts_match():
    selector = CaseSelector.__new__(CaseSelector)
    spec = SimulationSpec(
        solver="simpleFoam",
        case_type="channel_2d",
        mesh_template="box_channel_2d",
        turbulence_model="kEpsilon",
        steady_state=True,
        inlet_velocity=10.0,
        dimensions=2,
        phenomenon="backward_facing_step",
    )
    assert selector._passes_hard_filter(spec, _meta())


def test_meshing_demo_excluded():
    selector = CaseSelector.__new__(CaseSelector)
    spec = SimulationSpec(
        solver="simpleFoam",
        case_type="channel_2d",
        mesh_template="box_channel_2d",
        turbulence_model="laminar",
        steady_state=True,
        inlet_velocity=1.0,
        dimensions=2,
        phenomenon="general",
    )
    assert not selector._passes_hard_filter(spec, _meta(phenomenon="meshing_demo"))


def test_build_query_includes_phenomenon():
    selector = CaseSelector.__new__(CaseSelector)
    spec = SimulationSpec(
        solver="simpleFoam",
        case_type="channel_2d",
        mesh_template="box_channel_2d",
        turbulence_model="laminar",
        steady_state=True,
        inlet_velocity=1.0,
        dimensions=2,
        phenomenon="airfoil_steady",
        description="2D翼の定常流れ",
    )
    q = selector._build_query(spec)
    assert "airfoil_steady" in q
    assert "翼" in q or "airfoil" in q

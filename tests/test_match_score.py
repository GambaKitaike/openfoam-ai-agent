"""match_score のユニットテスト。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import SimulationSpec
from src.rag.match_score import compute_match_score, should_use_fast_path


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
    )
    defaults.update(overrides)
    return SimulationSpec(**defaults)


class TestMatchScore:
    def test_high_score_for_matching_meta(self):
        spec = _spec()
        meta = {
            "solver": "pimpleFoam",
            "phenomenon": "karman_vortex_shedding",
            "dimensions": 2,
            "steady_state": False,
            "turbulence_model": "laminar",
            "has_blockmesh": True,
        }
        score = compute_match_score(spec, meta)
        assert score >= 0.8

    def test_fast_path_requires_threshold_and_mesh(self):
        spec = _spec()
        meta = {"has_blockmesh": True}
        assert should_use_fast_path(spec, meta, 0.85) is True
        assert should_use_fast_path(spec, meta, 0.5) is False

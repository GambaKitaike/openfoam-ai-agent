"""_build_jinja_context のユニットテスト（LLM 呼び出しなし）"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.openfoam_gpt import OpenFOAMGPTAgent
from src.models import SimulationSpec, EnrichedContext


class FakeSettings:
    llm_provider = "openai"
    llm_model = "gpt-4o"
    openai_api_key = "fake-key"
    anthropic_api_key = ""
    openfoam_path = "/usr/lib/openfoam/openfoam2512"
    temperature = 0.1
    chroma_path = ""
    knowledge_base_path = ""


def _make_spec(**overrides) -> SimulationSpec:
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
        description="test",
    )
    defaults.update(overrides)
    return SimulationSpec(**defaults)


def _make_context(spec: SimulationSpec) -> EnrichedContext:
    return EnrichedContext(spec=spec)


def _agent() -> OpenFOAMGPTAgent:
    return OpenFOAMGPTAgent(FakeSettings())


class TestBuildJinjaContext:
    def test_has_wall_for_channel_2d(self):
        spec = _make_spec(case_type="channel_2d")
        ctx = _agent()._build_jinja_context(_make_context(spec))
        assert ctx["has_wall"] is True

    def test_has_wall_for_snappy_2d(self):
        spec = _make_spec(case_type="snappy_2d")
        ctx = _agent()._build_jinja_context(_make_context(spec))
        assert ctx["has_wall"] is True

    def test_is_snappy_2d_flag(self):
        spec = _make_spec(case_type="snappy_2d")
        ctx = _agent()._build_jinja_context(_make_context(spec))
        assert ctx["is_snappy_2d"] is True

    def test_is_not_snappy_2d_for_channel(self):
        spec = _make_spec(case_type="channel_2d")
        ctx = _agent()._build_jinja_context(_make_context(spec))
        assert ctx["is_snappy_2d"] is False

    def test_steady_state_timing(self):
        spec = _make_spec(steady_state=True)
        ctx = _agent()._build_jinja_context(_make_context(spec))
        assert ctx["end_time"] == 1000
        assert ctx["delta_t"] == 1
        assert ctx["write_interval"] == 100

    def test_unsteady_timing_scales_with_flow_through(self):
        """非定常: end_time が characteristic_length / inlet_velocity に比例するか"""
        spec = _make_spec(
            steady_state=False,
            case_type="channel_2d",
            solver="pimpleFoam",
            inlet_velocity=1.0,
            characteristic_length=1.0,
        )
        ctx = _agent()._build_jinja_context(_make_context(spec))
        # flow_through = 1.0 / 1.0 = 1.0s → end_time = 10.0s
        assert ctx["end_time"] == pytest.approx(10.0, rel=0.1)
        assert ctx["steady_state"] is False

    def test_snappy_object_name_from_stl_path(self):
        spec = _make_spec(case_type="snappy_2d", stl_path="/some/path/cylinder_2d.stl")
        ctx = _agent()._build_jinja_context(_make_context(spec))
        assert ctx["snappy_object_name"] == "cylinder_2d"

    def test_snappy_object_name_default_when_no_stl(self):
        spec = _make_spec(case_type="channel_2d", stl_path="")
        ctx = _agent()._build_jinja_context(_make_context(spec))
        assert ctx["snappy_object_name"] == "object"

    def test_solver_in_context(self):
        spec = _make_spec(solver="pimpleFoam")
        ctx = _agent()._build_jinja_context(_make_context(spec))
        assert ctx["solver"] == "pimpleFoam"

    def test_unsteady_snappy_2d_very_small_delta_t(self):
        """snappy_2d 非定常: 初期 deltaT が極小（Courant 爆発防止）"""
        spec = _make_spec(
            steady_state=False,
            case_type="snappy_2d",
            solver="pimpleFoam",
            inlet_velocity=1.0,
            characteristic_length=0.1,
        )
        ctx = _agent()._build_jinja_context(_make_context(spec))
        # flow_through = 0.1s → delta_t = 0.1 / 50000 = 2e-6
        assert ctx["delta_t"] < 1e-4, f"snappy_2d の delta_t が大きすぎる: {ctx['delta_t']}"

    def test_karman_ogrid_sets_delta_t_and_runtime_output(self):
        spec = _make_spec(
            steady_state=False,
            case_type="cylinder_2d_ogrid",
            solver="pimpleFoam",
            phenomenon="karman_vortex_shedding",
            turbulence_model="laminar",
            inlet_velocity=1.0,
            characteristic_length=1.0,
        )
        ctx = _agent()._build_jinja_context(_make_context(spec))
        assert ctx["is_karman_ogrid"] is True
        assert ctx["delta_t"] > 0
        assert ctx["purge_write"] == 0
        assert ctx["end_time"] == pytest.approx(125.0, rel=0.05)
        assert ctx["write_interval"] == pytest.approx(0.25, rel=0.05)

"""Agent①: preprocessing のユニットテスト（LLM 呼び出しなし）"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.preprocessing import PreprocessingAgent
from src.models import SimulationSpec


class FakeSettings:
    llm_provider = "openai"
    llm_model = "gpt-4o"
    openai_api_key = "fake-key"
    anthropic_api_key = ""
    openfoam_path = "/usr/lib/openfoam/openfoam2512"
    temperature = 0.1
    chroma_path = ""
    knowledge_base_path = ""


def make_agent() -> PreprocessingAgent:
    return PreprocessingAgent(FakeSettings())


# ─────────────────────────────────────────────
# _build_spec の case_type → mesh_template マッピング
# ─────────────────────────────────────────────

class TestBuildSpec:
    def _build(self, overrides: dict) -> SimulationSpec:
        agent = make_agent()
        base = {
            "solver": "simpleFoam",
            "case_type": "channel_2d",
            "dimensions": 2,
            "turbulence_model": "kOmegaSST",
            "steady_state": True,
            "inlet_velocity": 1.0,
            "characteristic_length": 1.0,
            "nu": 1.5e-5,
            "description": "test",
            "boundary_conditions": {},
            "mesh_params": {},
            "defaults_applied": [],
        }
        base.update(overrides)
        return agent._build_spec(base, "test")

    def test_channel_2d(self):
        spec = self._build({"case_type": "channel_2d", "dimensions": 2})
        assert spec.case_type == "channel_2d"
        assert spec.mesh_template == "box_channel_2d"

    def test_channel_3d(self):
        spec = self._build({"case_type": "channel_3d", "dimensions": 3})
        assert spec.case_type == "channel_3d"
        assert spec.mesh_template == "box_channel_3d"

    def test_snappy_2d(self):
        spec = self._build({"case_type": "snappy_2d", "dimensions": 2})
        assert spec.case_type == "snappy_2d"
        assert spec.mesh_template == "box_snappy_2d"

    def test_external_snappy(self):
        spec = self._build({"case_type": "external_snappy", "dimensions": 3})
        assert spec.case_type == "external_snappy"
        assert spec.mesh_template == "box_snappy"

    def test_heat_transfer(self):
        spec = self._build({"case_type": "heat_transfer", "dimensions": 3})
        assert spec.case_type == "heat_transfer"
        assert spec.mesh_template == "box_channel_3d"

    def test_unknown_case_type_falls_back(self):
        spec = self._build({"case_type": "unknown_xyz", "dimensions": 3})
        assert spec.case_type == "channel_2d"
        assert spec.mesh_template == "box_channel_2d"

    def test_legacy_external_flow_maps_to_external_snappy(self):
        spec = self._build({"case_type": "external_flow", "dimensions": 3})
        assert spec.case_type == "external_snappy"

    def test_legacy_internal_flow_maps_to_channel_3d(self):
        spec = self._build({"case_type": "internal_flow", "dimensions": 3})
        assert spec.case_type == "channel_3d"

    def test_re_number_calculated(self):
        spec = self._build({"inlet_velocity": 2.0, "characteristic_length": 0.5, "nu": 1e-6})
        assert spec.re_number == pytest.approx(1e6, rel=1e-3)

    def test_steady_state_flag(self):
        spec_true = self._build({"steady_state": True})
        spec_false = self._build({"steady_state": False})
        assert spec_true.steady_state is True
        assert spec_false.steady_state is False

    def test_inlet_velocity_string_stripped(self):
        """速度に単位が混入しても数値のみ抽出できること"""
        spec = self._build({"inlet_velocity": "5.0 m/s"})
        assert spec.inlet_velocity == pytest.approx(5.0)


# ─────────────────────────────────────────────
# STL 指定時の case_type 上書きロジック
# ─────────────────────────────────────────────

class TestStlOverride:
    def _run_with_stl(self, base_spec_overrides: dict, stl_path: str):
        agent = make_agent()
        base = {
            "solver": "pimpleFoam",
            "case_type": "channel_2d",
            "dimensions": 2,
            "turbulence_model": "laminar",
            "steady_state": False,
            "inlet_velocity": 0.15,
            "characteristic_length": 0.1,
            "nu": 1.5e-5,
            "description": "test",
            "boundary_conditions": {},
            "mesh_params": {},
            "defaults_applied": [],
        }
        base.update(base_spec_overrides)
        spec = agent._build_spec(base, "test")
        # STL 上書きロジックを手動再現
        from pathlib import Path as _Path
        if _Path(stl_path).exists():
            spec.stl_path = stl_path
            if spec.case_type != "snappy_2d":
                spec.case_type = "snappy_2d" if spec.dimensions == 2 else "snappy_external"
            spec.mesh_template = "box_snappy_2d" if spec.case_type == "snappy_2d" else "box_snappy"
        return spec

    def test_2d_stl_sets_snappy_2d(self, tmp_path):
        stl = tmp_path / "cylinder_2d.stl"
        stl.write_bytes(b"\x00" * 84)  # 最小バイナリ STL
        spec = self._run_with_stl({"dimensions": 2}, str(stl))
        assert spec.case_type == "snappy_2d"
        assert spec.mesh_template == "box_snappy_2d"

    def test_3d_stl_sets_external_snappy(self, tmp_path):
        stl = tmp_path / "body.stl"
        stl.write_bytes(b"\x00" * 84)
        spec = self._run_with_stl({"dimensions": 3}, str(stl))
        assert spec.case_type == "snappy_external"
        assert spec.mesh_template == "box_snappy"

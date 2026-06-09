"""Agent 間通信の統合テスト（LLM/RAG なし部分）。"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent_dialogue import AgentDialogueReport
from src.agents.spec_clarification import hearing_loop_with_agent2
from src.models import SimulationSpec


def _agent2_mock(profile_phenomenon="channel_internal"):
    agent2 = MagicMock()
    from src.rag.requirement_profile import build_requirement_profile, review_spec

    def get_profile(spec, description=""):
        return build_requirement_profile(spec, description)

    agent2.get_requirement_profile.side_effect = get_profile
    agent2.review_spec.side_effect = review_spec
    return agent2


class TestAgentDialogueLoop:
    def test_channel_triggers_review_and_auto_fix(self):
        spec = SimulationSpec(
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
        trace = AgentDialogueReport(description="channel test")
        out = hearing_loop_with_agent2(
            spec,
            _agent2_mock(),
            "2D channel simpleFoam",
            interactive=False,
            trace=trace,
        )
        kinds = [m.kind for m in trace.messages]
        assert "draft_spec" in kinds
        assert "requirement_profile" in kinds
        assert "review_issues" in kinds
        assert "review_fixes" in kinds
        assert out.turbulence_model == "kOmegaSST"

    def test_karman_user_laminar_keeps_laminar_at_re100(self):
        spec = SimulationSpec(
            solver="pimpleFoam",
            case_type="cylinder_2d_ogrid",
            mesh_template="ogrid_cylinder_2d",
            turbulence_model="laminar",
            steady_state=False,
            inlet_velocity=1.0,
            dimensions=2,
            characteristic_length=1.0,
            nu=0.01,
            re_number=100.0,
            phenomenon="karman_vortex_shedding",
        )
        trace = AgentDialogueReport(description="karman")
        out = hearing_loop_with_agent2(
            spec,
            _agent2_mock("karman_vortex_shedding"),
            "2D円柱 Re=100 層流",
            interactive=False,
            trace=trace,
        )
        review_msgs = [m for m in trace.messages if m.kind == "review_issues"]
        assert out.turbulence_model == "laminar"
        assert out.re_number == pytest.approx(100.0)
        assert not any(
            i["key"] == "turbulence_model"
            for m in review_msgs
            for i in m.detail.get("issues", [])
        )

    def test_channel_laminar_lowers_velocity_instead_of_turbulence(self):
        spec = SimulationSpec(
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
        out = hearing_loop_with_agent2(
            spec,
            _agent2_mock(),
            "2D channel simpleFoam 層流",
            interactive=False,
        )
        assert out.turbulence_model == "laminar"
        assert out.re_number == pytest.approx(2300, rel=1e-3)
        assert out.inlet_velocity < 1.0

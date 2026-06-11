"""get_file_guidance / file_guidance モジュールのテスト。"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.prompt_generation import PromptGenerationAgent
from src.config import Settings
from src.models import EnrichedContext, SimulationSpec
from src.rag.file_guidance import build_file_guidance


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
        nu=0.001,
        phenomenon="karman_vortex_shedding",
    )
    defaults.update(overrides)
    return SimulationSpec(**defaults)


class TestFileGuidance:
    def test_build_file_guidance_includes_reference(self):
        text = build_file_guidance(
            "0/U",
            _spec(),
            reference_content="internalField uniform (1 0 0);",
            case_label="vortexShed",
            patch_names=["inlet", "outlet"],
        )
        assert "0/U" in text
        assert "vortexShed" in text
        assert "internalField" in text
        assert "inlet, outlet" in text

    def test_get_file_guidance_uses_context(self, monkeypatch):
        settings = Settings()
        agent = PromptGenerationAgent(settings)
        ctx = EnrichedContext(
            spec=_spec(),
            reference_case_id="incompressible/pimpleFoam/vortexShed",
            reference_title_ja="渦放出",
            reference_files={
                "system/controlDict": "application pimpleFoam;\nendTime 10;",
            },
        )
        guidance = agent.get_file_guidance("system/controlDict", ctx.spec, ctx)
        assert "pimpleFoam" in guidance
        assert "渦放出" in guidance

    def test_get_file_guidance_without_context(self, monkeypatch):
        settings = Settings()
        agent = PromptGenerationAgent(settings)
        agent.retriever.selector = MagicMock()
        agent.retriever.selector.is_available = False
        guidance = agent.get_file_guidance("0/p", _spec())
        assert "0/p" in guidance
        assert "pimpleFoam" in guidance

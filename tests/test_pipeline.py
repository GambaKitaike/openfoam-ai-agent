"""CaseBuildPipeline のユニットテスト。"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.case_builder.pipeline import CaseBuildPipeline
from src.models import EnrichedContext, SimulationSpec


class FakeSettings:
    openfoam_path = "/usr/lib/openfoam/openfoam2512"
    openai_api_key = "fake"
    llm_provider = "openai"
    llm_model = "gpt-4o"
    anthropic_api_key = ""
    temperature = 0.1


def _context() -> EnrichedContext:
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
        phenomenon="karman_vortex_shedding",
    )
    return EnrichedContext(spec=spec)


class TestCaseBuildPipeline:
    def test_control_dict_written_before_blockmesh(self, tmp_path):
        pipeline = CaseBuildPipeline(FakeSettings())
        written: list[str] = []
        call_order: list[str] = []

        original_write = pipeline._write

        def track_write(case_path, rel, content):
            written.append(rel)
            call_order.append(rel)
            original_write(case_path, rel, content)

        mock_bm = MagicMock(returncode=0, log_file="log.blockMesh")
        mock_cm = MagicMock(success=True)
        mock_runner = MagicMock()
        mock_runner.run_block_mesh.return_value = mock_bm
        mock_runner.run_check_mesh.return_value = mock_cm
        pipeline.runner = mock_runner

        with patch.object(pipeline, "_write", side_effect=track_write), patch(
            "src.case_builder.pipeline.read_patch_names",
            return_value=["inlet", "outlet", "top", "bottom", "cylinder", "frontAndBack"],
        ):
            _state, gen = pipeline.run(_context(), str(tmp_path), run_mesh=True)

        assert "system/controlDict" in written
        assert written.index("system/controlDict") < written.index("system/blockMeshDict")
        assert call_order.index("system/controlDict") < call_order.index("system/blockMeshDict")
        mock_runner.run_block_mesh.assert_called_once()
        assert gen.mesh_built is True

    def test_blockmesh_failure_does_not_raise(self, tmp_path):
        pipeline = CaseBuildPipeline(FakeSettings())
        mock_bm = MagicMock(returncode=1, log_file="log.blockMesh")
        mock_runner = MagicMock()
        mock_runner.run_block_mesh.return_value = mock_bm
        pipeline.runner = mock_runner

        with patch(
            "src.case_builder.pipeline.read_patch_names",
            return_value=["inlet", "outlet"],
        ):
            _state, gen = pipeline.run(_context(), str(tmp_path), run_mesh=True)

        assert gen.mesh_built is False
        assert (tmp_path / "pimpleFoam_cylinder_2d_ogrid" / "system" / "controlDict").exists()

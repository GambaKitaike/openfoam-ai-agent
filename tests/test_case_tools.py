"""case_tools の単体テスト — 生成のみ・実行系は呼ばないこと。"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import EnrichedContext, GenerationResult, SimulationSpec
from src.tools.case_tools import case_scaffold


class FakeSettings:
    llm_provider = "openai"
    llm_model = "gpt-4o"
    llm_model_mini = "gpt-4o-mini"
    openai_api_key = "fake-key"
    anthropic_api_key = ""
    openfoam_version = "2512"
    openfoam_root = "/usr/lib/openfoam/openfoam2512"
    default_output_dir = "./output"

    def model_copy(self, *, update=None):
        clone = FakeSettings()
        if update:
            for key, value in update.items():
                setattr(clone, key, value)
        return clone


def _sample_spec(**overrides) -> SimulationSpec:
    base = dict(
        solver="pimpleFoam",
        case_type="cylinder_2d_ogrid",
        mesh_template="ogrid_cylinder_2d",
        turbulence_model="laminar",
        steady_state=False,
        inlet_velocity=0.15,
        dimensions=2,
        characteristic_length=0.1,
        nu=1.5e-5,
        description="2D cylinder Karman vortex",
        phenomenon="karman_vortex_shedding",
        boundary_conditions={
            "inlet": {"velocity": 0.15},
            "outlet": {"type": "zeroGradient"},
            "wall": {"type": "noSlip"},
        },
    )
    base.update(overrides)
    return SimulationSpec(**base)


def _sample_context(spec: SimulationSpec | None = None) -> EnrichedContext:
    spec = spec or _sample_spec()
    return EnrichedContext(
        spec=spec,
        rag_available=False,
        mesh_template_name=spec.mesh_template,
        mesh_params_suggestion={"nx": 40, "ny": 20, "nz": 1},
    )


def _sample_generation(output_path: Path) -> GenerationResult:
    (output_path / "system").mkdir(parents=True, exist_ok=True)
    (output_path / "system" / "controlDict").write_text("application pimpleFoam;\n")
    return GenerationResult(
        output_path=str(output_path),
        case_type="cylinder_2d_ogrid",
        files_created=["system/controlDict", "system/blockMeshDict"],
        mesh_built=False,
        build_path="staged",
    )


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "case_ws"
    ws.mkdir()
    return ws


class TestCaseScaffold:
    @patch("src.tools.case_tools.OpenFOAMGPTAgent")
    @patch("src.tools.case_tools.PromptGenerationAgent")
    @patch("src.tools.case_tools.PreprocessingAgent")
    @patch("src.tools.case_tools.clarify_from_reference")
    def test_scaffold_generates_without_execution(
        self,
        mock_clarify,
        mock_pre_cls,
        mock_agent2_cls,
        mock_agent3_cls,
        workspace: Path,
    ):
        spec = _sample_spec()
        context = _sample_context(spec)
        mock_clarify.side_effect = lambda s, _ctx, **_: s

        mock_agent1 = MagicMock()
        mock_agent1.extract.return_value = spec
        mock_agent1.complete_hearing.return_value = spec
        mock_pre_cls.return_value = mock_agent1

        mock_match = MagicMock()
        mock_match.context = context
        mock_match.context.reference_case_id = ""
        mock_agent2 = MagicMock()
        mock_agent2.retrieve_match.return_value = mock_match
        mock_agent2_cls.return_value = mock_agent2

        gen_dir = workspace / "pimpleFoam_cylinder_2d_ogrid"
        mock_agent3 = MagicMock()
        mock_agent3._generate_case.return_value = _sample_generation(gen_dir)
        mock_agent3_cls.return_value = mock_agent3

        result = case_scaffold(
            workspace,
            "円柱周り2Dカルマン渦、流入0.15m/s、層流",
            settings=FakeSettings(),
        )

        assert result.ok is True
        assert result.data is not None
        assert result.data["spec"]["solver"] == "pimpleFoam"
        assert result.data["spec"]["case_type"] == "cylinder_2d_ogrid"
        assert "solver: pimpleFoam" in result.content
        assert "case_type: cylinder_2d_ogrid" in result.content
        assert "inlet" in result.content

        mock_agent1.extract.assert_called_once()
        mock_agent1.complete_hearing.assert_called_once()
        mock_agent2.retrieve_match.assert_called_once()
        mock_agent3._generate_case.assert_called_once()
        mock_agent3.run.assert_not_called()

        assert (workspace / "system" / "controlDict").exists()
        assert not (workspace / "pimpleFoam_cylinder_2d_ogrid").exists()

    @patch("src.tools.case_tools.OpenFOAMGPTAgent")
    @patch("src.tools.case_tools.PromptGenerationAgent")
    @patch("src.tools.case_tools.PreprocessingAgent")
    @patch("src.tools.case_tools.clarify_from_reference")
    def test_pipeline_run_mesh_disabled(
        self,
        mock_clarify,
        mock_pre_cls,
        mock_agent2_cls,
        mock_agent3_cls,
        workspace: Path,
    ):
        spec = _sample_spec()
        context = _sample_context(spec)
        mock_clarify.side_effect = lambda s, _ctx, **_: s

        mock_agent1 = MagicMock()
        mock_agent1.extract.return_value = spec
        mock_agent1.complete_hearing.return_value = spec
        mock_pre_cls.return_value = mock_agent1

        mock_match = MagicMock()
        mock_match.context = context
        mock_agent2 = MagicMock()
        mock_agent2.retrieve_match.return_value = mock_match
        mock_agent2_cls.return_value = mock_agent2

        mock_agent3 = MagicMock()
        real_pipeline = MagicMock()
        runner = MagicMock()
        real_pipeline.runner = runner
        run_mesh_flags: list[bool] = []

        def capture_run(_ctx, _out_dir, *, run_mesh=True):
            run_mesh_flags.append(run_mesh)
            gen_dir = workspace / "pimpleFoam_cylinder_2d_ogrid"
            return None, _sample_generation(gen_dir)

        real_pipeline.run.side_effect = capture_run
        mock_agent3.pipeline = real_pipeline
        mock_agent3._generate_case.side_effect = lambda ctx, out, ref: real_pipeline.run(
            ctx, out, run_mesh=False
        )[1]
        mock_agent3_cls.return_value = mock_agent3

        with patch("src.tools.case_tools._generate_case_files_only") as mock_gen_only:
            mock_gen_only.side_effect = lambda agent3, ctx, out, ref: agent3._generate_case(
                ctx, out, ref
            )
            case_scaffold(workspace, "test case", settings=FakeSettings())

        assert run_mesh_flags == [False]
        runner.run_block_mesh.assert_not_called()
        runner.run_check_mesh.assert_not_called()
        runner.run_solver.assert_not_called()
        runner.run_snappy_hex_mesh.assert_not_called()
        runner.run_surface_feature_extract.assert_not_called()
        runner.run_potential_foam.assert_not_called()

    @patch("src.tools.case_tools.OpenFOAMGPTAgent")
    @patch("src.tools.case_tools.PromptGenerationAgent")
    @patch("src.tools.case_tools.PreprocessingAgent")
    def test_uses_mini_model_for_preprocessing(
        self,
        mock_pre_cls,
        mock_agent2_cls,
        mock_agent3_cls,
        workspace: Path,
    ):
        spec = _sample_spec()
        context = _sample_context(spec)

        mock_agent1 = MagicMock()
        mock_agent1.extract.return_value = spec
        mock_agent1.complete_hearing.return_value = spec
        mock_pre_cls.return_value = mock_agent1

        mock_match = MagicMock()
        mock_match.context = context
        mock_agent2 = MagicMock()
        mock_agent2.retrieve_match.return_value = mock_match
        mock_agent2_cls.return_value = mock_agent2

        mock_agent3 = MagicMock()
        mock_agent3._generate_case.return_value = _sample_generation(workspace)
        mock_agent3_cls.return_value = mock_agent3

        settings = FakeSettings()
        case_scaffold(workspace, "test", settings=settings)

        pre_settings = mock_pre_cls.call_args[0][0]
        assert pre_settings.llm_model == "gpt-4o-mini"
        agent3_settings = mock_agent3_cls.call_args[0][0]
        assert agent3_settings.llm_model == "gpt-4o"

    @patch("src.tools.case_tools.OpenFOAMGPTAgent")
    @patch("src.tools.case_tools.PromptGenerationAgent")
    @patch("src.tools.case_tools.PreprocessingAgent")
    def test_generation_failure_returns_error(
        self,
        mock_pre_cls,
        mock_agent2_cls,
        mock_agent3_cls,
        workspace: Path,
    ):
        spec = _sample_spec()
        context = _sample_context(spec)

        mock_agent1 = MagicMock()
        mock_agent1.extract.return_value = spec
        mock_agent1.complete_hearing.return_value = spec
        mock_pre_cls.return_value = mock_agent1

        mock_match = MagicMock()
        mock_match.context = context
        mock_agent2 = MagicMock()
        mock_agent2.retrieve_match.return_value = mock_match
        mock_agent2_cls.return_value = mock_agent2

        mock_agent3 = MagicMock()
        mock_agent3._generate_case.side_effect = RuntimeError("boom")
        mock_agent3_cls.return_value = mock_agent3

        result = case_scaffold(workspace, "test", settings=FakeSettings())

        assert result.ok is False
        assert "Case generation failed" in result.content
        mock_agent3.run.assert_not_called()

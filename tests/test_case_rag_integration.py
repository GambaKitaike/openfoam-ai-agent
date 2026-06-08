"""ケース単位 RAG 統合テスト（OpenFOAM 実行なし）"""
from __future__ import annotations

import pytest

from src.case_applier import CaseApplier
from src.case_validator import CaseValidator
from src.models import EnrichedContext, SimulationSpec
from src.rag.case_catalog import TUTORIALS_ROOT, discover_cases, load_case_files


@pytest.mark.skipif(not TUTORIALS_ROOT.exists(), reason="OpenFOAM tutorials not installed")
def test_reference_case_bundle_consistency(tmp_path):
    """参照ケースの全ファイルが同一 case_path 由来であること。"""
    cases = discover_cases()
    case = next(
        c for c in cases
        if c.case_id == "incompressible/simpleFoam/pitzDaily"
        or (c.solver == "simpleFoam" and "pitzDaily" in c.case_id and not c.requires_preprocessing)
    )
    files = load_case_files(case.case_path)
    assert "system/fvSchemes" in files
    assert "system/fvSolution" in files
    assert "system/controlDict" in files

    spec = SimulationSpec(
        solver=case.solver,
        case_type="channel_2d",
        mesh_template="box_channel_2d",
        turbulence_model="kOmegaSST",
        steady_state=case.steady_state,
        inlet_velocity=1.0,
        dimensions=case.dimensions,
        nu=1.5e-5,
        description="integration test",
    )
    ctx = EnrichedContext(
        spec=spec,
        reference_case_id=case.case_id,
        reference_case_path=case.case_path,
        reference_files=files,
    )
    applier = CaseApplier()
    created = applier.apply(ctx, tmp_path)

    assert len(created) >= 5
    validator = CaseValidator()
    issues = validator.validate(tmp_path, spec)
    errors = [i for i in issues if i.severity == "error"]
    assert not any(i.check == "zero" for i in errors)

    # fvSchemes と fvSolution が同じケース由来（simpleFoam SIMPLE）
    fv = (tmp_path / "system" / "fvSolution").read_text()
    assert "SIMPLE" in fv

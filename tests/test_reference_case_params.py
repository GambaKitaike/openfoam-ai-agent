"""参照ケース典型パラメータ抽出のテスト"""
from __future__ import annotations

from pathlib import Path

from src.agents.spec_clarification import collect_reference_clarifications
from src.models import SimulationSpec
from src.rag.reference_case_params import ReferenceCaseParams, extract_reference_params


def test_extract_from_reference_files():
    ref = extract_reference_params(
        case_id="incompressible/simpleFoam/backwardFacingStep2D",
        case_path="/tmp/x",
        reference_files={
            "constant/transportProperties": "nu 1e-5;",
            "constant/turbulenceProperties": "simulationType RAS;\nRASModel kOmegaSST;",
            "system/controlDict": "application simpleFoam;\nendTime 1000;\ndeltaT 1;",
            "0/U": "internalField uniform (10 0 0);\nboundaryField { inlet { type fixedValue; value uniform (10 0 0); } }\n",
        },
        summary_ja="Re=36000 の後方ステップ定常流れ",
    )
    assert ref.inlet_velocity == 10.0
    assert ref.nu == 1e-5
    assert ref.re_from_summary == 36000
    assert ref.turbulence_model == "kOmegaSST"


def test_collect_reference_clarifications_diff():
    spec = SimulationSpec(
        solver="simpleFoam",
        case_type="channel_2d",
        mesh_template="box_channel_2d",
        turbulence_model="kOmegaSST",
        steady_state=True,
        inlet_velocity=20.0,
        dimensions=2,
        nu=1.5e-5,
        characteristic_length=1.0,
    )
    ref = ReferenceCaseParams(
        case_id="test/airFoil2D",
        title_ja="2次元翼",
        inlet_velocity=25.75,
        nu=1.5e-5,
        turbulence_model="SpalartAllmaras",
        velocity_note="internalField",
    )
    fields = collect_reference_clarifications(spec, ref)
    keys = {f.key for f in fields}
    assert "inlet_velocity" in keys
    assert "turbulence_model" in keys


def test_extract_airfoil_u_from_fixture(tmp_path: Path):
    case = tmp_path / "airFoil2D"
    (case / "0.orig").mkdir(parents=True)
    (case / "0.orig" / "U").write_text(
        "internalField   uniform (25.75 3.62 0);\n"
        "boundaryField { frontAndBack { type empty; } }\n"
    )
    (case / "constant").mkdir()
    (case / "constant" / "transportProperties").write_text("nu 1.5e-5;")
    ref = extract_reference_params(
        case_id="incompressible/simpleFoam/airFoil2D",
        case_path=case,
        summary_ja="2次元翼周りの定常流れ",
    )
    assert ref.inlet_velocity is not None
    assert 25 < ref.inlet_velocity < 27

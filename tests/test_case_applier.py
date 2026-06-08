"""CaseApplier ユニットテスト"""
from __future__ import annotations

from pathlib import Path

from src.case_applier import CaseApplier, copy_prebuilt_mesh
from src.models import EnrichedContext, SimulationSpec


def test_apply_substitutes_nu_and_velocity(tmp_path):
    spec = SimulationSpec(
        solver="pimpleFoam",
        case_type="external_snappy",
        mesh_template="box_snappy",
        turbulence_model="laminar",
        steady_state=False,
        inlet_velocity=0.1,
        dimensions=2,
        characteristic_length=0.1,
        nu=1e-4,
        description="test",
    )
    ref_files = {
        "constant/transportProperties": "nu 1.5e-05;\n",
        "constant/turbulenceProperties": "simulationType RAS;\n",
        "system/controlDict": (
            "application pimpleFoam;\nendTime 1;\ndeltaT 0.001;\nadjustTimeStep yes;\n"
        ),
        "system/fvSchemes": "ddtSchemes { default Euler; }\n",
        "system/fvSolution": "PIMPLE { nCorrectors 2; }\n",
        "0/U": "internalField uniform (1 0 0);\nboundaryField { inlet { type fixedValue; value uniform (1 0 0); } }\n",
        "0/p": "internalField uniform 0;\nboundaryField { inlet { type zeroGradient; } }\n",
    }
    ctx = EnrichedContext(
        spec=spec,
        reference_case_id="test/case",
        reference_case_path=str(tmp_path),
        reference_files=ref_files,
    )
    applier = CaseApplier()
    applier.apply(ctx, tmp_path)

    tr = (tmp_path / "constant" / "transportProperties").read_text()
    assert "1e-04" in tr or "0.0001" in tr

    u = (tmp_path / "0" / "U").read_text()
    assert "0.1" in u

    tp = (tmp_path / "constant" / "turbulenceProperties").read_text()
    assert "laminar" in tp

    cd = (tmp_path / "system" / "controlDict").read_text()
    assert "endTime         30" in cd


def test_copy_prebuilt_mesh_from_orig(tmp_path):
    source = tmp_path / "tutorial"
    dest = tmp_path / "output"
    (source / "constant" / "polyMesh.orig").mkdir(parents=True)
    (source / "constant" / "polyMesh.orig" / "points").write_text("FoamFile\n{}\n")
    assert copy_prebuilt_mesh(source, dest) is True
    assert (dest / "constant" / "polyMesh" / "points").exists()


def test_apply_mesh_prebuilt_skips_turbulence_override(tmp_path):
    source = tmp_path / "tutorial"
    (source / "constant" / "polyMesh.orig").mkdir(parents=True)
    (source / "constant" / "polyMesh.orig" / "boundary").write_text("empty\n")
    spec = SimulationSpec(
        solver="simpleFoam",
        case_type="snappy_2d",
        mesh_template="box_snappy_2d",
        turbulence_model="kOmegaSST",
        steady_state=True,
        inlet_velocity=10.0,
        dimensions=2,
        nu=1.5e-5,
        description="test",
    )
    ref_files = {
        "constant/turbulenceProperties": "simulationType RAS;\nRASModel SpalartAllmaras;\n",
        "system/controlDict": "application simpleFoam;\nendTime 100;\n",
        "0/U": "internalField uniform (1 0 0);\n",
        "0/p": "internalField uniform 0;\n",
    }
    ctx = EnrichedContext(
        spec=spec,
        reference_case_id="incompressible/simpleFoam/airFoil2D",
        reference_case_path=str(source),
        reference_files=ref_files,
        reference_mesh_prebuilt=True,
    )
    out = tmp_path / "case"
    CaseApplier().apply(ctx, out)
    tp = (out / "constant" / "turbulenceProperties").read_text()
    assert "SpalartAllmaras" in tp
    assert (out / "constant" / "polyMesh" / "boundary").exists()

"""CaseValidator ユニットテスト"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.case_validator import CaseValidator
from src.models import SimulationSpec


@pytest.fixture
def validator():
    return CaseValidator()


@pytest.fixture
def laminar_spec():
    return SimulationSpec(
        solver="pimpleFoam",
        case_type="cylinder_2d_ogrid",
        mesh_template="ogrid_cylinder_2d",
        turbulence_model="laminar",
        steady_state=False,
        inlet_velocity=0.1,
        dimensions=2,
        characteristic_length=0.1,
        nu=1e-4,
        description="test",
    )


def test_validate_missing_zero(tmp_path, validator, laminar_spec):
    (tmp_path / "system").mkdir()
    (tmp_path / "system" / "controlDict").write_text("application pimpleFoam;\n")
    (tmp_path / "system" / "fvSchemes").write_text("ddtSchemes { default Euler; }\n")
    (tmp_path / "system" / "fvSolution").write_text("PIMPLE { nCorrectors 2; }\n")
    issues = validator.validate(tmp_path, laminar_spec)
    assert any(i.check == "zero" for i in issues)


def test_validate_steady_state_mismatch(tmp_path, validator, laminar_spec):
    (tmp_path / "0").mkdir()
    (tmp_path / "0" / "U").write_text("internalField uniform (0.1 0 0);\nboundaryField { inlet { type fixedValue; value uniform (0.1 0 0); } }\n")
    (tmp_path / "0" / "p").write_text("internalField uniform 0;\nboundaryField { inlet { type zeroGradient; } }\n")
    (tmp_path / "system").mkdir()
    (tmp_path / "system" / "controlDict").write_text("application pimpleFoam;\n")
    (tmp_path / "system" / "fvSchemes").write_text("ddtSchemes { default steadyState; }\n")
    (tmp_path / "system" / "fvSolution").write_text("PIMPLE { nCorrectors 2; }\n")
    (tmp_path / "constant").mkdir()
    (tmp_path / "constant" / "transportProperties").write_text(f"nu {laminar_spec.nu};\n")
    (tmp_path / "constant" / "turbulenceProperties").write_text("simulationType laminar;\n")
    issues = validator.validate(tmp_path, laminar_spec)
    assert any(i.check == "ddt" for i in issues)


def test_validate_solver_mismatch(tmp_path, validator, laminar_spec):
    spec = laminar_spec
    spec.solver = "icoFoam"
    (tmp_path / "0").mkdir()
    (tmp_path / "0" / "U").write_text("x")
    (tmp_path / "0" / "p").write_text("x")
    (tmp_path / "system").mkdir()
    (tmp_path / "system" / "controlDict").write_text("application pimpleFoam;\n")
    (tmp_path / "system" / "fvSchemes").write_text("ddtSchemes { default Euler; }\n")
    (tmp_path / "system" / "fvSolution").write_text("PISO { nCorrectors 2; }\n")
    issues = validator.validate(tmp_path, spec)
    assert any(i.check == "solver" for i in issues)

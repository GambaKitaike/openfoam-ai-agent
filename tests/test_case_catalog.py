"""ケースカタログのユニットテスト"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.rag.case_catalog import (
    TUTORIALS_ROOT,
    _detect_mesh_prebuilt,
    _extract_run_commands,
    _has_blockmesh_in_allrun,
    _infer_dimensions,
    _infer_geometry,
    _parse_turbulence_model,
    discover_cases,
    load_case_files,
)


@pytest.mark.skipif(not TUTORIALS_ROOT.exists(), reason="OpenFOAM tutorials not installed")
def test_discover_cases_finds_tutorials():
    cases = discover_cases()
    assert len(cases) > 50


@pytest.mark.skipif(not TUTORIALS_ROOT.exists(), reason="OpenFOAM tutorials not installed")
def test_pitzdaily_case_metadata():
    cases = discover_cases()
    pitz = next((c for c in cases if c.case_id.endswith("simpleFoam/pitzDaily")), None)
    if pitz is None:
        pitz = next((c for c in cases if "pitzDaily" in c.case_id and "simpleFoam" in c.case_id), None)
    assert pitz is not None
    assert pitz.solver == "simpleFoam"
    assert pitz.steady_state is True
    assert pitz.has_blockmesh


@pytest.mark.skipif(not TUTORIALS_ROOT.exists(), reason="OpenFOAM tutorials not installed")
def test_load_case_files():
    cases = discover_cases()
    case = next(c for c in cases if "system/controlDict" in c.indexed_files or c.has_blockmesh)
    files = load_case_files(case.case_path)
    assert "system/controlDict" in files


def test_parse_turbulence_laminar():
    text = "simulationType      laminar;"
    assert _parse_turbulence_model(text) == "laminar"


def test_infer_geometry_cylinder():
    assert _infer_geometry("incompressible/pimpleFoam/LES/vortexShed", "vortexShed") == "cylinder"


def test_detect_mesh_prebuilt(tmp_path: Path):
    case_dir = tmp_path / "case"
    (case_dir / "constant" / "polyMesh").mkdir(parents=True)
    assert _detect_mesh_prebuilt(case_dir) is True
    case2 = tmp_path / "case2"
    (case2 / "system").mkdir(parents=True)
    assert _detect_mesh_prebuilt(case2) is False


def test_extract_run_commands(tmp_path: Path):
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    (case_dir / "Allrun").write_text(
        "runApplication blockMesh\nrunApplication simpleFoam\n"
    )
    cmds = _extract_run_commands(case_dir)
    assert "blockMesh" in cmds
    assert "simpleFoam" in cmds


def test_has_blockmesh_in_allrun(tmp_path: Path):
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    (case_dir / "Allrun").write_text("runApplication simpleFoam\n")
    assert _has_blockmesh_in_allrun(case_dir) is False
    (case_dir / "Allrun").write_text("blockMesh\nsimpleFoam\n")
    assert _has_blockmesh_in_allrun(case_dir) is True


def test_infer_dimensions_from_zero_orig_empty_type(tmp_path: Path):
    """0.orig/U の 'type            empty;' を 2D と判定する。"""
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    (case_dir / "0.orig").mkdir()
    (case_dir / "0.orig" / "U").write_text(
        "boundaryField\n{\n    frontAndBack\n    {\n        type            empty;\n    }\n}\n"
    )
    u = (case_dir / "0.orig" / "U").read_text()
    assert _infer_dimensions("", u) == 2
    assert _infer_dimensions("", u, case_dir) == 2


@pytest.mark.skipif(not TUTORIALS_ROOT.exists(), reason="OpenFOAM tutorials not installed")
def test_airfoil2d_discovered_as_2d():
    cases = discover_cases()
    airfoil = next(c for c in cases if c.case_id.endswith("simpleFoam/airFoil2D"))
    assert airfoil.dimensions == 2
    assert airfoil.mesh_prebuilt is True

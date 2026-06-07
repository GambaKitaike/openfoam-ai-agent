"""pytest 共通フィクスチャ"""
from __future__ import annotations

from pathlib import Path

import pytest

TEMPLATES_DIR = Path(__file__).parent.parent / "templates"
SRC_DIR = Path(__file__).parent.parent / "src"


@pytest.fixture
def templates_dir() -> Path:
    return TEMPLATES_DIR


@pytest.fixture
def base_spec_data() -> dict:
    """最小限の SimulationSpec 相当データ（LLM 呼び出しなしで構築）"""
    return {
        "solver": "simpleFoam",
        "case_type": "channel_2d",
        "dimensions": 2,
        "turbulence_model": "kOmegaSST",
        "steady_state": True,
        "inlet_velocity": 1.0,
        "characteristic_length": 1.0,
        "nu": 1.5e-5,
        "description": "テスト用",
        "boundary_conditions": {},
        "mesh_params": {},
        "defaults_applied": [],
    }

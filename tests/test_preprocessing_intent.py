"""preprocessing の phenomenon 正規化テスト"""
from __future__ import annotations

from src.agents.preprocessing import PreprocessingAgent


def test_normalize_phenomenon_karman():
    assert PreprocessingAgent._normalize_phenomenon("", "2D円柱周りのカルマン渦 Re=100") == "karman_vortex_shedding"


def test_normalize_phenomenon_airfoil():
    assert PreprocessingAgent._normalize_phenomenon("", "2D翼の定常流れと揚力") == "airfoil_steady"


def test_normalize_phenomenon_passthrough():
    assert PreprocessingAgent._normalize_phenomenon("backward_facing_step", "") == "backward_facing_step"

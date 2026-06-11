"""case_runtime / runner ユーティリティのテスト。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.case_runtime import (
    find_latest_time,
    list_time_dirs,
    patch_control_dict_for_continue,
    read_end_time,
    read_solver_name,
)


@pytest.fixture
def mini_case(tmp_path: Path) -> Path:
    case = tmp_path / "case"
    (case / "system").mkdir(parents=True)
    (case / "0").mkdir()
    (case / "25.0").mkdir()
    (case / "system" / "controlDict").write_text(
        "application     pimpleFoam;\n"
        "startFrom       startTime;\n"
        "endTime         125.0;\n"
        "writeInterval   0.25;\n"
    )
    return case


class TestCaseRuntime:
    def test_list_and_find_latest_time(self, mini_case: Path):
        assert list_time_dirs(mini_case) == [0.0, 25.0]
        assert find_latest_time(mini_case) == 25.0

    def test_read_solver_and_end_time(self, mini_case: Path):
        assert read_solver_name(mini_case) == "pimpleFoam"
        assert read_end_time(mini_case) == 125.0

    def test_patch_control_dict_for_continue(self, mini_case: Path):
        latest = patch_control_dict_for_continue(mini_case, 200.0, write_interval=1.0)
        assert latest == 25.0
        text = (mini_case / "system" / "controlDict").read_text()
        assert "startFrom       latestTime;" in text
        assert "endTime         200;" in text
        assert "writeInterval   1;" in text

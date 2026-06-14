"""build_system_prompt / workspace スナップショットの単体テスト。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.prompts import build_system_prompt
from src.agent.session import SessionState


def _write_minimal_mesh(workspace: Path) -> None:
    poly_mesh = workspace / "constant" / "polyMesh"
    poly_mesh.mkdir(parents=True)
    for name in ("points", "faces", "owner"):
        (poly_mesh / name).write_text(f"{name}\n", encoding="utf-8")


class TestWorkspaceSnapshot:
    def test_mesh_generated_status_in_prompt(self, tmp_path: Path) -> None:
        _write_minimal_mesh(tmp_path)
        state = SessionState(workspace=tmp_path)

        prompt = build_system_prompt(state)

        assert "メッシュ: 生成済み（constant/polyMesh あり）" in prompt

    def test_mesh_not_generated_status_in_prompt(self, tmp_path: Path) -> None:
        state = SessionState(workspace=tmp_path)

        prompt = build_system_prompt(state)

        assert "メッシュ: 未生成" in prompt

    def test_computation_status_when_solver_time_exists(self, tmp_path: Path) -> None:
        _write_minimal_mesh(tmp_path)
        (tmp_path / "0").mkdir()
        (tmp_path / "0.5").mkdir()
        (tmp_path / "2.5").mkdir()
        state = SessionState(workspace=tmp_path)

        prompt = build_system_prompt(state)

        assert "計算: 実行済み（最終時刻 2.5）" in prompt

    def test_computation_not_run_when_only_initial_time(self, tmp_path: Path) -> None:
        _write_minimal_mesh(tmp_path)
        (tmp_path / "0").mkdir()
        state = SessionState(workspace=tmp_path)

        prompt = build_system_prompt(state)

        assert "計算: 未実行" in prompt

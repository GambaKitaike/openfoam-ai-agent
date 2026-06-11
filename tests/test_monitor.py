"""SolverMonitor のユニットテスト。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.monitor import SolverMonitor, compute_progress_pct


class TestProgressPct:
    def test_halfway(self):
        assert compute_progress_pct(25.0, 50.0) == pytest.approx(50.0)

    def test_caps_at_100(self):
        assert compute_progress_pct(60.0, 50.0) == pytest.approx(100.0)

    def test_no_end_time(self):
        assert compute_progress_pct(10.0, None) == 0.0


class TestSolverMonitorModes:
    def test_steady_shows_converged_subtitle(self):
        monitor = SolverMonitor("dummy.log", steady_state=True)
        monitor._status.residuals = {"Ux": 1e-5, "p": 1e-6}
        monitor._status.time_step = 100.0
        monitor._status.converged = True
        panel = monitor._make_steady_panel()
        assert "収束しました" in str(panel.subtitle)

    def test_transient_shows_progress_not_converged(self):
        monitor = SolverMonitor("dummy.log", steady_state=False, end_time=50.0)
        monitor._status.residuals = {"Ux": 1e-5, "Uy": 1e-4}
        monitor._status.time_step = 25.0
        monitor._status.progress_pct = 50.0
        monitor._status.iterations = 1000
        panel = monitor._make_transient_panel()
        rendered = panel.subtitle
        assert "進捗 50.0%" in str(rendered)
        assert "収束しました" not in str(rendered)

    def test_transient_parse_does_not_mark_residual_convergence(self, tmp_path: Path):
        log = tmp_path / "log.pimpleFoam"
        log.write_text(
            "Time = 5\n"
            "Solving for Ux, Initial residual = 1e-05, Final residual = 1e-06\n"
            "Solving for p, Initial residual = 1e-06, Final residual = 1e-07\n"
        )
        monitor = SolverMonitor(str(log), steady_state=False, end_time=50.0)
        status = monitor.parse_log()
        assert status.converged is False
        assert status.progress_pct == pytest.approx(10.0)

"""foam_tools の単体テスト。"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tools.foam_tools import (
    ALLOWED_COMMANDS,
    foam_dict_check,
    read_log,
    run_openfoam,
)


SAMPLE_SOLVER_LOG = """\
/*---------------------------------------------------------------------------*\\
Starting solver
Time = 0.001
smoothSolver:  Solving for Ux, Initial residual = 1.000e-01, Final residual = 1.000e-02
smoothSolver:  Solving for Uy, Initial residual = 9.000e-02, Final residual = 9.000e-03
smoothSolver:  Solving for p, Initial residual = 5.000e-02, Final residual = 5.000e-03
Time = 0.002
smoothSolver:  Solving for Ux, Initial residual = 5.000e-02, Final residual = 5.000e-03
smoothSolver:  Solving for Uy, Initial residual = 4.000e-02, Final residual = 4.000e-03
smoothSolver:  Solving for p, Initial residual = 2.000e-02, Final residual = 2.000e-03
Time = 0.003
smoothSolver:  Solving for Ux, Initial residual = 1.000e-02, Final residual = 1.000e-03
smoothSolver:  Solving for Uy, Initial residual = 9.000e-03, Final residual = 9.000e-04
smoothSolver:  Solving for p, Initial residual = 5.000e-03, Final residual = 5.000e-04
End
"""

SAMPLE_ERROR_LOG = """\
line 01
line 02
line 03
line 04
line 05
line 06
line 07
line 08
line 09
line 10
line 11
line 12
line 13
line 14
line 15
line 16
line 17
line 18
line 19
line 20
line 21
line 22
line 23
line 24
line 25
--> FOAM FATAL ERROR:
Invalid boundary condition
    From function void check()
    in file boundary.C
line 26
line 27
line 28
line 29
line 30
line 31
line 32
line 33
line 34
line 35
line 36
line 37
line 38
line 39
line 40
line 41
line 42
line 43
line 44
line 45
FOAM exiting
line 46
"""


@pytest.fixture
def mock_subprocess(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []

    def fake_run(
        cmd: str,
        shell: bool = True,
        capture_output: bool = True,
        text: bool = True,
        timeout: int | None = None,
    ) -> subprocess.CompletedProcess[str]:
        calls.append({"cmd": cmd, "timeout": timeout})
        output = "blockMesh completed successfully\n"
        import re

        match = re.search(r"tee (\S+)", cmd)
        if match:
            log_path = Path(match.group(1))
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(output, encoding="utf-8")
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=output, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


class TestAllowedCommands:
    def test_contains_design_list(self) -> None:
        expected = {
            "blockMesh",
            "snappyHexMesh",
            "surfaceFeatureExtract",
            "checkMesh",
            "potentialFoam",
            "simpleFoam",
            "pimpleFoam",
            "foamToVTK",
            "foamDictionary",
            "postProcess",
            "decomposePar",
            "reconstructPar",
        }
        assert ALLOWED_COMMANDS == expected


class TestRunOpenfoam:
    def test_rejects_disallowed_command(self, tmp_path: Path, mock_subprocess: list[dict[str, object]]) -> None:
        result = run_openfoam(tmp_path, "rm")

        assert result.ok is False
        assert "not allowed" in result.content.lower()
        assert mock_subprocess == []

    def test_runs_allowed_command_and_saves_log(
        self,
        tmp_path: Path,
        mock_subprocess: list[dict[str, object]],
    ) -> None:
        result = run_openfoam(tmp_path, "blockMesh")

        assert result.ok is True
        assert result.data is not None
        assert result.data["exit_code"] == 0
        assert "log.blockMesh" in str(mock_subprocess[0]["cmd"])
        assert "exit_code: 0" in result.content
        assert "log tail" in result.content.lower()
        assert "blockMesh completed successfully" in result.content
        assert mock_subprocess[0]["timeout"] == 1800
        assert "blockMesh" in str(mock_subprocess[0]["cmd"])

    def test_summary_is_tail_only_not_full_log(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        many_lines = "\n".join(f"line {index}" for index in range(200))

        def fake_run(
            cmd: str,
            shell: bool = True,
            capture_output: bool = True,
            text: bool = True,
            timeout: int | None = None,
        ) -> subprocess.CompletedProcess[str]:
            import re

            match = re.search(r"tee (\S+)", cmd)
            if match:
                Path(match.group(1)).write_text(many_lines, encoding="utf-8")
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=many_lines, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)

        result = run_openfoam(tmp_path, "blockMesh")

        assert result.ok is True
        assert "line 0" not in result.content
        assert "line 199" in result.content
        assert "omitted" in result.content

    def test_foam_fatal_error_marks_failure(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        fatal_log = "Starting\n--> FOAM FATAL ERROR:\nbad mesh\nFOAM exiting\n"

        def fake_run(
            cmd: str,
            shell: bool = True,
            capture_output: bool = True,
            text: bool = True,
            timeout: int | None = None,
        ) -> subprocess.CompletedProcess[str]:
            import re

            match = re.search(r"tee (\S+)", cmd)
            if match:
                Path(match.group(1)).write_text(fatal_log, encoding="utf-8")
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=fatal_log, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)

        result = run_openfoam(tmp_path, "blockMesh")

        assert result.ok is False
        assert "exit_code: 1" in result.content


class TestReadLog:
    @pytest.fixture
    def workspace_with_logs(self, tmp_path: Path) -> Path:
        (tmp_path / "log.pimpleFoam").write_text(SAMPLE_SOLVER_LOG, encoding="utf-8")
        (tmp_path / "log.blockMesh").write_text(SAMPLE_ERROR_LOG, encoding="utf-8")
        return tmp_path

    def test_tail_mode_default_50_lines(self, workspace_with_logs: Path) -> None:
        long_log = "\n".join(f"row {index}" for index in range(80))
        log_path = workspace_with_logs / "log.long"
        log_path.write_text(long_log, encoding="utf-8")

        result = read_log(workspace_with_logs, "log.long", "tail")

        assert result.ok is True
        assert "row 0" not in result.content
        assert "row 79" in result.content
        assert "omitted" in result.content

    def test_tail_mode_custom_lines(self, workspace_with_logs: Path) -> None:
        long_log = "\n".join(f"row {index}" for index in range(20))
        log_path = workspace_with_logs / "log.short"
        log_path.write_text(long_log, encoding="utf-8")

        result = read_log(workspace_with_logs, "log.short", "tail", tail_lines=5)

        assert result.ok is True
        assert "row 15" in result.content
        assert "row 0" not in result.content

    def test_errors_mode_extracts_context(self, workspace_with_logs: Path) -> None:
        result = read_log(workspace_with_logs, "log.blockMesh", "errors")

        assert result.ok is True
        assert "FOAM FATAL ERROR" in result.content
        assert "line 06" in result.content
        assert "line 45" in result.content
        assert "FOAM exiting" in result.content
        assert "line 01" not in result.content

    def test_errors_mode_no_errors(self, workspace_with_logs: Path) -> None:
        result = read_log(workspace_with_logs, "log.pimpleFoam", "errors")

        assert result.ok is True
        assert "No errors found" in result.content

    def test_residuals_mode_parses_and_formats(self, workspace_with_logs: Path) -> None:
        result = read_log(workspace_with_logs, "log.pimpleFoam", "residuals")

        assert result.ok is True
        assert "time=0.001" in result.content
        assert "Ux=" in result.content
        assert "p=" in result.content
        assert "time=0.003" in result.content

    def test_residuals_mode_subsamples_large_series(self, workspace_with_logs: Path) -> None:
        lines = []
        for step in range(100):
            lines.append(f"Time = {step * 0.001}")
            lines.append(
                f"Solving for Ux, Initial residual = {1.0 / (step + 1):.3e}, Final residual = 1.000e-06"
            )
        log_path = workspace_with_logs / "log.big"
        log_path.write_text("\n".join(lines), encoding="utf-8")

        result = read_log(workspace_with_logs, "log.big", "residuals")

        assert result.ok is True
        assert "sampled from 100 time steps" in result.content
        assert result.content.count("time=") <= 30

    def test_unknown_mode(self, workspace_with_logs: Path) -> None:
        result = read_log(workspace_with_logs, "log.pimpleFoam", "full")

        assert result.ok is False
        assert "unknown mode" in result.content.lower()

    def test_rejects_path_escape(self, workspace_with_logs: Path) -> None:
        outside = workspace_with_logs.parent / "outside.log"
        outside.write_text("secret", encoding="utf-8")

        result = read_log(workspace_with_logs, "../outside.log", "tail")

        assert result.ok is False
        assert "escapes workspace" in result.content.lower()


class TestFoamDictCheck:
    def test_success(self, tmp_path: Path, mock_subprocess: list[dict[str, object]]) -> None:
        target = tmp_path / "system" / "controlDict"
        target.parent.mkdir(parents=True)
        target.write_text("application simpleFoam;\n", encoding="utf-8")

        result = foam_dict_check(tmp_path, "system/controlDict")

        assert result.ok is True
        assert "foamDictionary OK" in result.content
        assert "foamDictionary" in str(mock_subprocess[0]["cmd"])
        assert str(target.resolve()) in str(mock_subprocess[0]["cmd"])

    def test_failure(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        target = tmp_path / "system" / "controlDict"
        target.parent.mkdir(parents=True)
        target.write_text("bad syntax", encoding="utf-8")

        def fake_run(
            cmd: str,
            shell: bool = True,
            capture_output: bool = True,
            text: bool = True,
            timeout: int | None = None,
        ) -> subprocess.CompletedProcess[str]:
            output = "--> FOAM FATAL ERROR:\nparse error\nFOAM exiting\n"
            import re

            match = re.search(r"tee (\S+)", cmd)
            if match:
                Path(match.group(1)).write_text(output, encoding="utf-8")
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=output, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)

        result = foam_dict_check(tmp_path, "system/controlDict")

        assert result.ok is False
        assert "foamDictionary failed" in result.content

    def test_missing_file(self, tmp_path: Path, mock_subprocess: list[dict[str, object]]) -> None:
        result = foam_dict_check(tmp_path, "system/controlDict")

        assert result.ok is False
        assert "not found" in result.content.lower()
        assert mock_subprocess == []

"""registry の単体テスト。"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.session import SessionState
from src.tools.registry import dispatch, schemas


def _tool_call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    import json

    return {
        "id": "call_test",
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments),
        },
    }


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


class TestSchemas:
    def test_returns_openai_function_definitions(self) -> None:
        tool_schemas = schemas()
        names = {item["function"]["name"] for item in tool_schemas}

        assert all(item["type"] == "function" for item in tool_schemas)
        assert names == {
            "list_files",
            "read_file",
            "edit_file",
            "write_file",
            "run_openfoam",
            "read_log",
            "foam_dict_check",
        }


class TestConfirmGate:
    def test_edit_file_rejected_without_modifying_file(self, tmp_path: Path) -> None:
        target = tmp_path / "system" / "controlDict"
        target.parent.mkdir(parents=True)
        target.write_text("endTime 1;\n", encoding="utf-8")
        state = SessionState(workspace=tmp_path)
        prompts: list[str] = []

        result = dispatch(
            _tool_call("edit_file", {"path": "system/controlDict", "old_str": "endTime 1;", "new_str": "endTime 2;"}),
            state,
            confirm_fn=lambda prompt: prompts.append(prompt) or False,
        )

        assert result.ok is False
        assert "rejected" in result.content.lower()
        assert target.read_text() == "endTime 1;\n"
        assert len(prompts) == 1
        assert "edit_file: system/controlDict" in prompts[0]
        assert "---" in prompts[0]

    def test_edit_file_approved_applies_change(self, tmp_path: Path, mock_subprocess: list[dict[str, object]]) -> None:
        target = tmp_path / "note.txt"
        target.write_text("hello\n", encoding="utf-8")
        state = SessionState(workspace=tmp_path)

        result = dispatch(
            _tool_call("edit_file", {"path": "note.txt", "old_str": "hello", "new_str": "world"}),
            state,
            confirm_fn=lambda _: True,
        )

        assert result.ok is True
        assert target.read_text() == "world\n"
        assert mock_subprocess == []

    def test_write_file_rejected(self, tmp_path: Path) -> None:
        state = SessionState(workspace=tmp_path)

        result = dispatch(
            _tool_call("write_file", {"path": "new.txt", "content": "data"}),
            state,
            confirm_fn=lambda _: False,
        )

        assert result.ok is False
        assert not (tmp_path / "new.txt").exists()

    def test_write_file_approved(self, tmp_path: Path) -> None:
        state = SessionState(workspace=tmp_path)
        prompts: list[str] = []

        result = dispatch(
            _tool_call("write_file", {"path": "new.txt", "content": "data"}),
            state,
            confirm_fn=lambda prompt: prompts.append(prompt) or True,
        )

        assert result.ok is True
        assert (tmp_path / "new.txt").read_text() == "data"
        assert "write_file: new.txt" in prompts[0]

    def test_run_openfoam_rejected(self, tmp_path: Path, mock_subprocess: list[dict[str, object]]) -> None:
        state = SessionState(workspace=tmp_path)

        result = dispatch(
            _tool_call("run_openfoam", {"command": "blockMesh"}),
            state,
            confirm_fn=lambda _: False,
        )

        assert result.ok is False
        assert mock_subprocess == []
        assert state.run_records == []

    def test_run_openfoam_approved(self, tmp_path: Path, mock_subprocess: list[dict[str, object]]) -> None:
        state = SessionState(workspace=tmp_path)
        prompts: list[str] = []

        result = dispatch(
            _tool_call("run_openfoam", {"command": "blockMesh"}),
            state,
            confirm_fn=lambda prompt: prompts.append(prompt) or True,
        )

        assert result.ok is True
        assert mock_subprocess
        assert "run_openfoam: blockMesh" in prompts[0]


class TestEditFileAutoDictCheck:
    def test_system_dict_triggers_foam_dict_check(
        self,
        tmp_path: Path,
        mock_subprocess: list[dict[str, object]],
    ) -> None:
        target = tmp_path / "system" / "controlDict"
        target.parent.mkdir(parents=True)
        target.write_text("application simpleFoam;\n", encoding="utf-8")
        state = SessionState(workspace=tmp_path)

        result = dispatch(
            _tool_call(
                "edit_file",
                {
                    "path": "system/controlDict",
                    "old_str": "simpleFoam",
                    "new_str": "pimpleFoam",
                },
            ),
            state,
            confirm_fn=lambda _: True,
        )

        assert result.ok is True
        assert "Updated system/controlDict" in result.content
        assert "foamDictionary check" in result.content
        assert "foamDictionary OK" in result.content
        assert len(mock_subprocess) == 1
        assert "foamDictionary" in str(mock_subprocess[0]["cmd"])

    def test_non_dict_path_skips_auto_check(
        self,
        tmp_path: Path,
        mock_subprocess: list[dict[str, object]],
    ) -> None:
        target = tmp_path / "README.md"
        target.write_text("solver simpleFoam\n", encoding="utf-8")
        state = SessionState(workspace=tmp_path)

        result = dispatch(
            _tool_call(
                "edit_file",
                {"path": "README.md", "old_str": "simpleFoam", "new_str": "pimpleFoam"},
            ),
            state,
            confirm_fn=lambda _: True,
        )

        assert result.ok is True
        assert "foamDictionary" not in result.content
        assert mock_subprocess == []


class TestRunRecord:
    def test_run_openfoam_appends_run_record(
        self,
        tmp_path: Path,
        mock_subprocess: list[dict[str, object]],
    ) -> None:
        state = SessionState(workspace=tmp_path)

        result = dispatch(
            _tool_call("run_openfoam", {"command": "blockMesh"}),
            state,
            confirm_fn=lambda _: True,
        )

        assert result.ok is True
        assert len(state.run_records) == 1
        record = state.run_records[0]
        assert record.command == "blockMesh"
        assert record.log_path == Path("log.blockMesh")
        assert record.exit_code == 0
        assert record.started_at <= record.finished_at
        assert "blockMesh finished" in record.summary

    def test_read_file_does_not_require_confirmation(self, tmp_path: Path) -> None:
        target = tmp_path / "note.txt"
        target.write_text("hello\n", encoding="utf-8")
        state = SessionState(workspace=tmp_path)

        result = dispatch(
            _tool_call("read_file", {"path": "note.txt"}),
            state,
            confirm_fn=lambda _: False,
        )

        assert result.ok is True
        assert "hello" in result.content

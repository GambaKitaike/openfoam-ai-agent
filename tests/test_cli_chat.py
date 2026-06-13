"""cli_chat の単体テスト。"""
from __future__ import annotations

import sys
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console

from src.agent.session import SessionState, load, save
from src.cli_chat import (
    describe_workspace,
    format_tool_invocation,
    handle_slash_command,
    make_confirm_fn,
    print_startup_banner,
    run_chat,
)


class TestDescribeWorkspace:
    def test_empty_workspace(self, tmp_path: Path) -> None:
        assert describe_workspace(tmp_path) == "(新規ワークスペース)"

    def test_existing_case(self, tmp_path: Path) -> None:
        (tmp_path / "system").mkdir()
        (tmp_path / "system" / "controlDict").write_text(
            "application     pimpleFoam;\nendTime 10;\n",
            encoding="utf-8",
        )
        (tmp_path / "2.5").mkdir()

        assert describe_workspace(tmp_path) == "(既存ケースを検出: pimpleFoam, 最終時刻 2.5s)"

    def test_case_without_time_dirs(self, tmp_path: Path) -> None:
        (tmp_path / "system").mkdir()
        (tmp_path / "system" / "controlDict").write_text(
            "application simpleFoam;\n",
            encoding="utf-8",
        )

        assert describe_workspace(tmp_path) == "(既存ケースを検出: simpleFoam, タイムディレクトリなし)"


class TestFormatToolInvocation:
    def test_read_file(self) -> None:
        assert format_tool_invocation("read_file", {"path": "system/blockMeshDict"}) == (
            "read_file system/blockMeshDict"
        )

    def test_run_openfoam(self) -> None:
        assert format_tool_invocation(
            "run_openfoam",
            {"command": "blockMesh", "args": ["-case", "."]},
        ) == "run_openfoam blockMesh -case ."


class TestConfirmFn:
    def test_yolo_mode_auto_approves(self) -> None:
        console = Console(file=StringIO(), force_terminal=True, width=120)
        confirm = make_confirm_fn(console, yolo_mode=[True])
        assert confirm("edit_file: note.txt\n--- diff ---") is True

    def test_prompts_and_accepts(self) -> None:
        console = Console(file=StringIO(), force_terminal=True, width=120)
        confirm = make_confirm_fn(console, yolo_mode=[False])

        with patch.object(console, "input", side_effect=["y"]):
            assert confirm("run_openfoam: blockMesh\ntimeout: 1800s") is True

        with patch.object(console, "input", side_effect=[""]):
            assert confirm("run_openfoam: blockMesh\ntimeout: 1800s") is False


class TestSlashCommands:
    def test_status_and_yolo_toggle(self, tmp_path: Path) -> None:
        console = Console(file=StringIO(), force_terminal=True, width=120)
        state = SessionState(workspace=tmp_path)
        yolo_mode = [False]

        assert handle_slash_command("/status", state, console, yolo_mode) is False
        assert handle_slash_command("/yolo", state, console, yolo_mode) is False
        assert yolo_mode[0] is True
        assert handle_slash_command("/quit", state, console, yolo_mode) is True


class TestSessionPersistence:
    def test_run_chat_saves_on_turn_and_exit(self, tmp_path: Path) -> None:
        from src.agent.core import assistant_message, user_message

        console = Console(file=StringIO(), force_terminal=True, width=120)
        inputs = iter(["hello", "/quit"])
        console.input = lambda *args, **kwargs: next(inputs)  # type: ignore[method-assign, assignment]

        def fake_run_turn(user_input: str, state: SessionState) -> str:
            state.history.append(user_message(user_input))
            state.history.append(assistant_message("了解しました。"))
            return "了解しました。"

        with patch("src.cli_chat.AgentCore.run_turn", side_effect=fake_run_turn):
            with patch("src.cli_chat.LLMClient"):
                run_chat(tmp_path, console=console)

        session_file = tmp_path / ".ofagent" / "session.json"
        assert session_file.is_file()
        restored = load(tmp_path)
        assert restored.history[0]["content"] == "hello"
        assert restored.history[-1]["content"] == "了解しました。"


class TestStartupBanner:
    def test_prints_ofagent_prefix(self, tmp_path: Path) -> None:
        output = StringIO()
        console = Console(file=output, force_terminal=False, no_color=True, width=120)
        print_startup_banner(tmp_path, console)
        rendered = output.getvalue()
        assert "[ofagent]" in rendered
        assert tmp_path.as_posix() in rendered
        assert "新規ワークスペース" in rendered

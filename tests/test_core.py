"""AgentCore の単体テスト。"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.core import (
    AgentCore,
    MAX_STEPS,
    assistant_message,
    assistant_tool_call_message,
    tool_result_message,
    truncate_history_if_needed,
    user_message,
)
from src.agent.prompts import build_system_prompt
from src.agent.session import SessionState
from src.llm_client import ChatResponse
from src.tools.base import ToolResult


def _tool_call(name: str, arguments: dict[str, Any], call_id: str = "call_1") -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments),
        },
    }


class TestMessageHelpers:
    def test_user_and_assistant_messages(self) -> None:
        assert user_message("hello") == {"role": "user", "content": "hello"}
        assert assistant_message("done") == {"role": "assistant", "content": "done"}

    def test_tool_result_message(self) -> None:
        call = _tool_call("read_file", {"path": "note.txt"})
        result = ToolResult(ok=True, content="file body")
        message = tool_result_message(call, result)
        assert message["role"] == "tool"
        assert message["tool_call_id"] == "call_1"
        assert message["content"] == "file body"


class TestTruncateHistory:
    def test_truncates_oldest_tool_results_when_over_budget(self) -> None:
        from src.agent.core import CONTEXT_CHAR_BUDGET

        history = [
            {"role": "tool", "tool_call_id": "a", "content": "x" * (CONTEXT_CHAR_BUDGET + 100)},
            {"role": "tool", "tool_call_id": "b", "content": "keep-me"},
        ]
        truncate_history_if_needed(history)
        assert history[0]["content"] == "(truncated)"
        assert history[1]["content"] == "keep-me"


class TestBuildSystemPrompt:
    def test_includes_workspace_snapshot_and_spec(self, tmp_path: Path, base_spec_data: dict[str, Any]) -> None:
        from src.models import SimulationSpec

        (tmp_path / "system").mkdir()
        (tmp_path / "system" / "controlDict").write_text("application simpleFoam;\n", encoding="utf-8")
        spec = SimulationSpec(mesh_template="box_internal", **base_spec_data)
        state = SessionState(workspace=tmp_path, spec=spec)

        prompt = build_system_prompt(state)

        assert "OpenFOAM ケースの構築" in prompt
        assert "収束の鉄則" in prompt
        assert "case_scaffold" in prompt
        assert "最小差分" in prompt
        assert "controlDict" in prompt
        assert "simpleFoam" in prompt
        assert "channel_2d" in prompt
        assert tmp_path.as_posix() in prompt


class TestAgentCoreLoop:
    def test_tool_call_then_text_response(self, tmp_path: Path) -> None:
        target = tmp_path / "note.txt"
        target.write_text("hello\n", encoding="utf-8")
        state = SessionState(workspace=tmp_path)

        tool_response = ChatResponse(
            text=None,
            tool_calls=[_tool_call("read_file", {"path": "note.txt"})],
        )
        final_response = ChatResponse(text="ファイルを読みました。", tool_calls=[])

        message_lengths: list[int] = []

        def chat_side_effect(**kwargs: Any) -> ChatResponse:
            message_lengths.append(len(kwargs["messages"]))
            if len(message_lengths) == 1:
                return tool_response
            return final_response

        llm = MagicMock()
        llm.chat_with_tools.side_effect = chat_side_effect

        agent = AgentCore(llm=llm)
        result = agent.run_turn("note.txt を読んで", state)

        assert result == "ファイルを読みました。"
        assert llm.chat_with_tools.call_count == 2
        assert message_lengths == [1, 3]
        assert state.history[0] == user_message("note.txt を読んで")
        assert state.history[1]["role"] == "assistant"
        assert state.history[1]["tool_calls"]
        assert state.history[2]["role"] == "tool"
        assert "hello" in state.history[2]["content"]
        assert state.history[3] == assistant_message("ファイルを読みました。")

        first_call_kwargs = llm.chat_with_tools.call_args_list[0].kwargs
        assert "system" in first_call_kwargs
        assert first_call_kwargs["tools"]

    def test_immediate_text_response_without_tools(self, tmp_path: Path) -> None:
        state = SessionState(workspace=tmp_path)
        llm = MagicMock()
        llm.chat_with_tools.return_value = ChatResponse(text="了解しました。", tool_calls=[])

        agent = AgentCore(llm=llm)
        result = agent.run_turn("こんにちは", state)

        assert result == "了解しました。"
        assert llm.chat_with_tools.call_count == 1
        assert len(state.history) == 2

    def test_max_steps_returns_summary(self, tmp_path: Path) -> None:
        state = SessionState(workspace=tmp_path)
        always_tool = ChatResponse(
            text=None,
            tool_calls=[_tool_call("list_files", {}, call_id="loop")],
        )

        llm = MagicMock()
        llm.chat_with_tools.return_value = always_tool

        agent = AgentCore(llm=llm)
        result = agent.run_turn("一覧を見て", state)

        assert "ステップ上限に達しました" in result
        assert str(MAX_STEPS) in result
        assert llm.chat_with_tools.call_count == MAX_STEPS

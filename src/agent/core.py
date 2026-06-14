"""AgentCore — ツール使用ループ（DESIGN.md §5.1, §5.4）。"""
from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from src.agent.prompts import build_system_prompt
from src.agent.session import Message, SessionState, save, sanitize_utf8_text
from src.llm_client import ChatResponse, LLMClient
from src.tools import registry
from src.tools.base import ToolResult

MAX_STEPS = 15
# 概ね 80K トークン相当（1 トークン ≈ 4 文字）
CONTEXT_CHAR_BUDGET = 80_000 * 4
_TRUNCATED_MARKER = "(truncated)"


def user_message(content: str) -> Message:
    return {"role": "user", "content": content}


def assistant_message(content: str) -> Message:
    return {"role": "assistant", "content": content}


def assistant_tool_call_message(response: ChatResponse) -> Message:
    message: Message = {
        "role": "assistant",
        "content": response.text or "",
        "tool_calls": response.tool_calls,
    }
    return message


def tool_result_message(tool_call: dict[str, Any], result: ToolResult) -> Message:
    return {
        "role": "tool",
        "tool_call_id": tool_call["id"],
        "content": sanitize_utf8_text(result.content),
    }


def _message_char_size(message: Message) -> int:
    size = len(str(message.get("content", "")))
    tool_calls = message.get("tool_calls")
    if tool_calls:
        size += len(json.dumps(tool_calls, ensure_ascii=False))
    return size


def _estimate_history_chars(history: list[Message]) -> int:
    return sum(_message_char_size(message) for message in history)


def truncate_history_if_needed(history: list[Message]) -> None:
    """古い tool result の content を (truncated) に置換する（§5.4）。"""
    total = _estimate_history_chars(history)
    if total <= CONTEXT_CHAR_BUDGET:
        return

    for message in history:
        if message.get("role") != "tool":
            continue
        content = message.get("content", "")
        if content == _TRUNCATED_MARKER:
            continue
        message["content"] = _TRUNCATED_MARKER
        total = _estimate_history_chars(history)
        if total <= CONTEXT_CHAR_BUDGET:
            break


def _build_step_limit_summary(state: SessionState, steps: int) -> str:
    lines = [f"実行ステップ数: {steps}/{MAX_STEPS}"]

    if state.run_records:
        lines.append("実行履歴:")
        for record in state.run_records[-5:]:
            status = "OK" if record.exit_code == 0 else f"exit {record.exit_code}"
            lines.append(f"- {record.command} [{status}]: {record.summary}")

    tool_messages = [message for message in state.history if message.get("role") == "tool"]
    if tool_messages:
        lines.append("直近のツール結果:")
        for message in tool_messages[-3:]:
            content = str(message.get("content", ""))
            preview = content if len(content) <= 200 else content[:200] + "..."
            lines.append(f"- {preview}")

    assistant_messages = [
        message for message in state.history if message.get("role") == "assistant" and message.get("content")
    ]
    if assistant_messages:
        last = str(assistant_messages[-1].get("content", ""))
        if last:
            preview = last if len(last) <= 300 else last[:300] + "..."
            lines.append(f"直前のアシスタント応答: {preview}")

    return "\n".join(lines)


class AgentCore:
    """ツール使用ループを実行するエージェント本体。"""

    def __init__(
        self,
        llm: LLMClient,
        confirm_fn: Callable[[str], bool] | None = None,
    ):
        self.llm = llm
        self.confirm_fn = confirm_fn or (lambda _: True)

    def run_turn(self, user_input: str, state: SessionState) -> str:
        state.history.append(user_message(user_input))
        save(state)

        for step in range(1, MAX_STEPS + 1):
            truncate_history_if_needed(state.history)
            response = self.llm.chat_with_tools(
                system=build_system_prompt(state),
                messages=state.history,
                tools=registry.schemas(),
            )

            if response.has_tool_calls:
                state.history.append(assistant_tool_call_message(response))
                for call in response.tool_calls:
                    result = registry.dispatch(call, state, self.confirm_fn)
                    state.history.append(tool_result_message(call, result))
                    save(state)
                continue

            final_text = response.text or ""
            state.history.append(assistant_message(final_text))
            save(state)
            return final_text

        summary = _build_step_limit_summary(state, MAX_STEPS)
        return f"(ステップ上限に達しました。状況を整理します…)\n{summary}"

"""CLI チャット REPL（DESIGN.md §6 Phase 1）。"""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax

from src.agent.core import AgentCore
from src.agent.prompts import _format_run_records_summary, _format_spec_summary
from src.agent.session import SessionState, load, save
from src.case_runtime import find_latest_time, read_solver_name
from src.config import Settings
from src.llm_client import LLMClient
from src.tools import registry


def describe_workspace(workspace: Path) -> str:
    """ワークスペース走査結果のサフィックス文字列を返す。"""
    control_dict = workspace / "system" / "controlDict"
    if not control_dict.is_file():
        return "(新規ワークスペース)"

    solver = read_solver_name(workspace) or "不明"
    latest = find_latest_time(workspace)
    if latest is not None:
        time_part = f"最終時刻 {latest:g}s"
    else:
        time_part = "タイムディレクトリなし"
    return f"(既存ケースを検出: {solver}, {time_part})"


def print_startup_banner(workspace: Path, console: Console | None = None) -> None:
    """起動時バナーを §6 の表示例に合わせて出力する。"""
    console = console or Console()
    ws_display = workspace.resolve().as_posix()
    suffix = describe_workspace(workspace.resolve())
    console.print(
        f"[bold]\\[ofagent][/] ワークスペース: [cyan]{ws_display}[/] {suffix}"
    )


def _parse_tool_call_for_display(tool_call: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    function = tool_call.get("function") or {}
    name = function.get("name", "")
    raw_args = function.get("arguments", "{}")
    if isinstance(raw_args, dict):
        return name, raw_args
    if not raw_args:
        return name, {}
    return name, json.loads(raw_args)


def format_tool_invocation(name: str, args: dict[str, Any]) -> str:
    """§6 の [tool] 行用ラベルを組み立てる。"""
    if name == "read_file":
        return f"read_file {args.get('path', '')}"
    if name == "list_files":
        path = args.get("path", ".")
        depth = args.get("depth")
        if depth is not None:
            return f"list_files {path} depth={depth}"
        return f"list_files {path}"
    if name == "edit_file":
        return f"edit_file {args.get('path', '')}"
    if name == "write_file":
        return f"write_file {args.get('path', '')}"
    if name == "run_openfoam":
        command = args.get("command", "")
        cmd_args = args.get("args")
        if isinstance(cmd_args, list):
            suffix = " " + " ".join(cmd_args)
        elif isinstance(cmd_args, str) and cmd_args.strip():
            suffix = " " + cmd_args.strip()
        else:
            suffix = ""
        return f"run_openfoam {command}{suffix}"
    if name == "read_log":
        return f"read_log {args.get('log_path', '')} mode={args.get('mode', '')}"
    if name == "foam_dict_check":
        return f"foam_dict_check {args.get('path', '')}"
    if name == "case_scaffold":
        description = str(args.get("description", ""))
        preview = description if len(description) <= 40 else description[:37] + "..."
        return f"case_scaffold {preview!r}"
    if name == "rag_search":
        return f"rag_search {args.get('query', '')!r}"
    return name


def display_confirmation(console: Console, prompt: str) -> None:
    """確認ゲート用の diff / コマンドを rich で整形表示する。"""
    lines = prompt.split("\n", 1)
    header = lines[0]
    body = lines[1] if len(lines) > 1 else ""

    if header.startswith("edit_file:"):
        path = header.split(":", 1)[1].strip()
        diff_body = body or "(no textual changes)\n"
        console.print(
            Panel(
                Syntax(diff_body, "diff", theme="ansi_dark", line_numbers=False),
                title=f"edit_file: {path}",
                border_style="yellow",
            )
        )
        return

    if header.startswith("write_file:"):
        path = header.split(":", 1)[1].strip()
        console.print(
            Panel(
                body,
                title=f"write_file: {path}",
                border_style="yellow",
            )
        )
        return

    if header.startswith("run_openfoam:"):
        cmd_line = header.split(":", 1)[1].strip()
        content = f"command: {cmd_line}"
        if body.strip():
            content = f"{content}\n{body}"
        console.print(
            Panel(
                content,
                title="run_openfoam",
                border_style="red",
            )
        )
        return

    console.print(Panel(prompt, title="確認", border_style="yellow"))


def make_confirm_fn(
    console: Console,
    yolo_mode: list[bool],
) -> Callable[[str], bool]:
    """確認ゲート。yolo_mode[0] が True のときは常に承認する。"""

    def confirm(prompt: str) -> bool:
        if yolo_mode[0]:
            return True
        display_confirmation(console, prompt)
        while True:
            answer = console.input("[bold yellow]実行しますか?[/] [y/n]: ").strip().lower()
            if answer in ("y", "yes"):
                return True
            if answer in ("n", "no", ""):
                return False
            console.print("[red]y または n を入力してください[/]")

    return confirm


def print_status(state: SessionState, console: Console | None = None) -> None:
    """/status 用: spec と実行履歴を表示する。"""
    console = console or Console()
    console.print(
        Panel(
            f"[bold]SimulationSpec[/bold]\n{_format_spec_summary(state)}\n\n"
            f"[bold]実行履歴[/bold]\n{_format_run_records_summary(state)}",
            title="Session Status",
            border_style="cyan",
        )
    )


def handle_slash_command(
    command_line: str,
    state: SessionState,
    console: Console,
    yolo_mode: list[bool],
) -> bool:
    """スラッシュコマンドを処理する。True を返すと REPL を終了する。"""
    command = command_line.strip().split(maxsplit=1)[0].lower()

    if command == "/quit":
        return True

    if command == "/yolo":
        yolo_mode[0] = not yolo_mode[0]
        status = "ON（確認省略）" if yolo_mode[0] else "OFF（確認あり）"
        console.print(f"[yellow]yolo モード: {status}[/]")
        return False

    if command == "/status":
        print_status(state, console)
        return False

    console.print(
        "[red]未知のコマンドです。[/] 利用可能: /status, /yolo, /quit"
    )
    return False


def run_chat(
    workspace: Path,
    *,
    resume: bool = False,
    yolo: bool = False,
    console: Console | None = None,
) -> None:
    """rich REPL を起動し、AgentCore で対話ループを回す。"""
    console = console or Console()
    workspace = workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    state = load(workspace) if resume else SessionState(workspace=workspace)
    yolo_mode = [yolo]

    print_startup_banner(workspace, console)

    settings = Settings()
    llm = LLMClient(settings)
    agent = AgentCore(llm=llm, confirm_fn=make_confirm_fn(console, yolo_mode))

    original_dispatch = registry.dispatch

    def dispatch_with_tool_log(
        tool_call: dict[str, Any],
        session_state: SessionState,
        confirm_fn: Callable[[str], bool],
    ):
        name, args = _parse_tool_call_for_display(tool_call)
        console.print(f"[bold cyan]\\[tool][/] {format_tool_invocation(name, args)}", end="")
        if name == "foam_dict_check":
            console.print(" [dim](foamDictionary 構文チェック — 確認不要)[/]")
        else:
            console.print()
        return original_dispatch(tool_call, session_state, confirm_fn)

    registry.dispatch = dispatch_with_tool_log  # type: ignore[assignment]

    try:
        while True:
            try:
                user_input = console.input("[bold green]>[/] ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print()
                break

            if not user_input:
                continue

            if user_input.startswith("/"):
                if handle_slash_command(user_input, state, console, yolo_mode):
                    break
                continue

            console.print("[dim]エージェント実行中（編集・実行の前に y/n で確認します）...[/]")
            response = agent.run_turn(user_input, state)

            save(state)
            console.print()
            console.print(Markdown(response))
            console.print()
    finally:
        registry.dispatch = original_dispatch
        save(state)
        console.print("[dim]セッションを保存しました。[/]")

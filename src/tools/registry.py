"""ツール定義(JSON Schema)と dispatch（DESIGN.md §4, §5.2）。"""
from __future__ import annotations

import difflib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from src.agent.session import RunRecord, SessionState, _deserialize_spec, sanitize_utf8_text
from src.tools.base import ToolResult
from src.tools.case_tools import case_scaffold
from src.tools.foam_tools import foam_dict_check, read_log, run_openfoam
from src.tools.fs_tools import edit_file, list_files, read_file, write_file
from src.tools.rag_tools import rag_search

_CONFIRM_TOOLS = frozenset({"edit_file", "write_file", "run_openfoam", "case_scaffold"})
_FOAM_DICT_PREFIXES = ("0", "system", "constant")
_MESH_SOLVER_COMMANDS = frozenset({"checkMesh", "potentialFoam", "simpleFoam", "pimpleFoam"})

_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files and directories under the workspace as a tree.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path from workspace root. Default: '.'.",
                    },
                    "depth": {
                        "type": "integer",
                        "description": "Maximum directory depth to expand. Default: 2.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file from the workspace. Binary files return header/summary only.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path to the file.",
                    },
                    "line_range": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 2,
                        "maxItems": 2,
                        "description": "Optional inclusive 1-based line range [start, end].",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace exactly one occurrence of old_str with new_str in a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path to the file."},
                    "old_str": {"type": "string", "description": "Exact substring to replace once."},
                    "new_str": {"type": "string", "description": "Replacement text."},
                },
                "required": ["path", "old_str", "new_str"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create a new file. Existing files must be edited with edit_file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path for the new file."},
                    "content": {"type": "string", "description": "Full file content."},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_openfoam",
            "description": "Run an allowlisted OpenFOAM command in the workspace case directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "OpenFOAM command name (e.g. blockMesh, pimpleFoam).",
                    },
                    "args": {
                        "description": "Optional command arguments as string or string array.",
                        "oneOf": [
                            {"type": "string"},
                            {"type": "array", "items": {"type": "string"}},
                        ],
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds. Default: 1800.",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_log",
            "description": "Read an OpenFOAM log with errors, tail, or residual sampling modes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "log_path": {"type": "string", "description": "Relative path to the log file."},
                    "mode": {
                        "type": "string",
                        "enum": ["errors", "tail", "residuals"],
                        "description": "Reading mode.",
                    },
                },
                "required": ["log_path", "mode"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "foam_dict_check",
            "description": "Validate an OpenFOAM dictionary file using foamDictionary.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path to the dict file."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "case_scaffold",
            "description": (
                "Generate an OpenFOAM case from natural language (files only; no solver execution)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "Natural language description of the simulation.",
                    },
                    "stl_path": {
                        "type": "string",
                        "description": "Optional relative path to an STL geometry file.",
                    },
                },
                "required": ["description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rag_search",
            "description": "Search the ChromaDB knowledge base for similar cases or file examples.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query text.",
                    },
                    "scope": {
                        "type": "string",
                        "enum": ["case", "file"],
                        "description": "Collection to search. Default: 'case'.",
                    },
                    "filters": {
                        "type": "object",
                        "description": "Optional metadata filters (e.g. solver, steady_or_transient).",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (3-5). Default: 3.",
                        "minimum": 3,
                        "maximum": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
]


def schemas() -> list[dict[str, Any]]:
    """OpenAI function calling 形式のツール定義を返す。"""
    return list(_TOOL_SCHEMAS)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_tool_call(tool_call: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    function = tool_call.get("function") or {}
    name = function.get("name", "")
    raw_args = function.get("arguments", "{}")
    if isinstance(raw_args, dict):
        return name, raw_args
    if not raw_args:
        return name, {}
    return name, json.loads(raw_args)


def _resolve_in_workspace(workspace: Path, rel_path: str) -> tuple[Path | None, str | None]:
    workspace_resolved = workspace.resolve()
    target = (workspace_resolved / rel_path).resolve()
    try:
        target.relative_to(workspace_resolved)
    except ValueError:
        return None, f"Path escapes workspace: {rel_path}"
    return target, None


def _is_foam_dict_path(path: str) -> bool:
    parts = Path(path).parts
    return bool(parts) and parts[0] in _FOAM_DICT_PREFIXES


def _build_edit_confirmation(workspace: Path, path: str, old_str: str, new_str: str) -> str:
    target, err = _resolve_in_workspace(workspace, path)
    if err:
        return f"edit_file: {path}\n{err}"
    if target is None or not target.exists() or not target.is_file():
        return f"edit_file: {path}\n(preview unavailable: file not found)"

    text = target.read_text(encoding="utf-8", errors="replace")
    updated = text.replace(old_str, new_str, 1)
    diff_lines = difflib.unified_diff(
        text.splitlines(keepends=True),
        updated.splitlines(keepends=True),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
    )
    diff_text = "".join(diff_lines) or "(no textual changes)\n"
    return f"edit_file: {path}\n{diff_text}"


def _build_write_confirmation(path: str, content: str) -> str:
    preview_limit = 500
    preview = content if len(content) <= preview_limit else content[:preview_limit] + "\n...(truncated)"
    return f"write_file: {path}\n--- content preview ---\n{preview}"


def _build_case_scaffold_confirmation(description: str, stl_path: str | None) -> str:
    stl_text = stl_path if stl_path else "(none)"
    return f"case_scaffold\ndescription: {description}\nstl_path: {stl_text}"


def _build_run_confirmation(
    command: str,
    args: list[str] | str | None,
    timeout: int,
    state: SessionState,
) -> str:
    if isinstance(args, list):
        args_text = " ".join(args)
    elif isinstance(args, str):
        args_text = args.strip()
    else:
        args_text = ""
    cmd_line = f"{command}{f' {args_text}' if args_text else ''}"
    lines = [f"run_openfoam: {cmd_line}", f"timeout: {timeout}s"]
    if command in _MESH_SOLVER_COMMANDS and command != "checkMesh":
        if not any(record.command == "checkMesh" for record in state.run_records):
            lines.append(
                "warning: checkMesh が未実行です。"
                " blockMesh 後に run_openfoam checkMesh を先に実行してください。"
            )
    return "\n".join(lines)


def _build_confirmation(name: str, args: dict[str, Any], state: SessionState) -> str:
    if name == "edit_file":
        return _build_edit_confirmation(
            state.workspace,
            args["path"],
            args["old_str"],
            args["new_str"],
        )
    if name == "write_file":
        return _build_write_confirmation(args["path"], args["content"])
    if name == "run_openfoam":
        timeout = int(args.get("timeout", 1800))
        return _build_run_confirmation(args["command"], args.get("args"), timeout, state)
    if name == "case_scaffold":
        return _build_case_scaffold_confirmation(args["description"], args.get("stl_path"))
    return f"{name}: {json.dumps(args, ensure_ascii=False)}"


def _append_dict_check(result: ToolResult, workspace: Path, path: str) -> ToolResult:
    check = foam_dict_check(workspace, path)
    content = result.content
    if check.content:
        content = f"{content}\n\n--- foamDictionary check ---\n{check.content}"
    return ToolResult(ok=result.ok and check.ok, content=content, data=result.data)


def _sanitize_tool_result(result: ToolResult) -> ToolResult:
    content = sanitize_utf8_text(result.content)
    if content == result.content:
        return result
    return ToolResult(ok=result.ok, content=content, data=result.data)


def _build_run_summary(command: str, exit_code: int, finished_at: datetime, result: ToolResult) -> str:
    status = "OK" if exit_code == 0 else "FAILED"
    lines = [f"{command} finished at {finished_at.isoformat()} — {status} (exit {exit_code})"]
    for line in result.content.splitlines():
        if line.startswith("exit_code:") or "log tail" in line.lower():
            continue
        if line.strip() and not line.startswith("command:") and not line.startswith("log:"):
            lines.append(line.strip())
            if len(lines) >= 3:
                break
    return "\n".join(lines[:3])


def _execute_tool(name: str, args: dict[str, Any], state: SessionState) -> ToolResult:
    workspace = state.workspace

    if name == "list_files":
        return list_files(workspace, path=args.get("path", "."), depth=int(args.get("depth", 2)))

    if name == "read_file":
        line_range = args.get("line_range")
        parsed_range = tuple(line_range) if line_range is not None else None
        return read_file(workspace, args["path"], line_range=parsed_range)

    if name == "edit_file":
        result = edit_file(workspace, args["path"], args["old_str"], args["new_str"])
        if result.ok and _is_foam_dict_path(args["path"]):
            return _append_dict_check(result, workspace, args["path"])
        return result

    if name == "write_file":
        return write_file(workspace, args["path"], args["content"])

    if name == "run_openfoam":
        started_at = _utc_now()
        timeout = int(args.get("timeout", 1800))
        result = run_openfoam(
            workspace,
            args["command"],
            args=args.get("args"),
            timeout=timeout,
        )
        result = _sanitize_tool_result(result)
        finished_at = _utc_now()
        exit_code = int(result.data.get("exit_code", 1)) if result.data else (0 if result.ok else 1)
        log_rel = result.data.get("log_path", f"log.{args['command']}") if result.data else f"log.{args['command']}"
        summary = _build_run_summary(args["command"], exit_code, finished_at, result)
        state.run_records.append(
            RunRecord(
                command=args["command"],
                log_path=Path(log_rel),
                exit_code=exit_code,
                started_at=started_at,
                finished_at=finished_at,
                summary=summary,
            )
        )
        return result

    if name == "read_log":
        return _sanitize_tool_result(read_log(workspace, args["log_path"], args["mode"]))

    if name == "foam_dict_check":
        return foam_dict_check(workspace, args["path"])

    if name == "case_scaffold":
        return case_scaffold(
            workspace,
            args["description"],
            stl_path=args.get("stl_path"),
        )

    if name == "rag_search":
        return rag_search(
            args["query"],
            scope=args.get("scope", "case"),
            top_k=int(args.get("top_k", 3)),
            filters=args.get("filters"),
        )

    return ToolResult(ok=False, content=f"Unknown tool: {name}")


def dispatch(
    tool_call: dict[str, Any],
    state: SessionState,
    confirm_fn: Callable[[str], bool],
) -> ToolResult:
    """ツール呼び出しを実行する。破壊的操作は confirm_fn で確認する。"""
    name, args = _parse_tool_call(tool_call)

    if name in _CONFIRM_TOOLS:
        confirmation = _build_confirmation(name, args, state)
        if not confirm_fn(confirmation):
            return ToolResult(ok=False, content="Operation rejected by user.")

    result = _execute_tool(name, args, state)
    if name == "case_scaffold" and result.ok and result.data:
        spec_data = result.data.get("spec")
        if spec_data is not None:
            state.spec = _deserialize_spec(spec_data)
    return _sanitize_tool_result(result)

"""OpenFOAM 実行・ログ読取ツール。"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

from src.config import Settings
from src.monitor import _RESIDUAL_PATTERN
from src.runner import _of_command

from .base import ToolResult

ALLOWED_COMMANDS = {
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

_TAIL_MAX_LINES = 50
_RESIDUAL_MAX_POINTS = 30
_ERROR_CONTEXT_LINES = 20

_ERROR_TRIGGERS = (
    "FOAM FATAL ERROR",
    "FATAL ERROR",
    "FOAM exiting",
)


def _resolve_in_workspace(workspace: Path, rel_path: str) -> tuple[Path | None, str | None]:
    workspace_resolved = workspace.resolve()
    target = (workspace_resolved / rel_path).resolve()
    try:
        target.relative_to(workspace_resolved)
    except ValueError:
        return None, f"Path escapes workspace: {rel_path}"
    return target, None


def _error(message: str) -> ToolResult:
    return ToolResult(ok=False, content=message)


def _format_args(args: list[str] | str | None) -> str:
    if args is None:
        return ""
    if isinstance(args, str):
        return args.strip()
    return " ".join(args)


def _detect_foam_fatal(text: str) -> bool:
    return "FOAM FATAL ERROR" in text or "FOAM exiting" in text


def _format_log_tail(text: str, max_lines: int = _TAIL_MAX_LINES) -> str:
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return "\n".join(lines)
    omitted = len(lines) - max_lines
    return f"... ({omitted} lines omitted)\n" + "\n".join(lines[-max_lines:])


def _effective_returncode(returncode: int, log_text: str) -> int:
    if returncode != 0:
        return returncode
    if _detect_foam_fatal(log_text):
        return 1
    return 0


def run_openfoam(
    workspace: Path,
    command: str,
    args: list[str] | str | None = None,
    timeout: int = 1800,
) -> ToolResult:
    if command not in ALLOWED_COMMANDS:
        return _error(f"Command not allowed: {command}")

    case_dir = str(workspace.resolve())
    args_str = _format_args(args)
    log_path = workspace.resolve() / f"log.{command}"

    shell_cmd = (
        f"set -o pipefail; {command} -case {case_dir}"
        f"{f' {args_str}' if args_str else ''} 2>&1 | tee {log_path}"
    )
    cmd_str = _of_command(shell_cmd, Settings())

    try:
        result = subprocess.run(
            cmd_str,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return _error(f"Command timed out after {timeout}s: {command}")

    stdout = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else result.stdout
    exit_code = _effective_returncode(result.returncode, stdout)
    tail = _format_log_tail(stdout)

    content = (
        f"command: {command}\n"
        f"exit_code: {exit_code}\n"
        f"log: log.{command}\n"
        f"\n--- log tail ---\n"
        f"{tail}"
    )
    return ToolResult(
        ok=exit_code == 0,
        content=content,
        data={"exit_code": exit_code, "log_path": f"log.{command}"},
    )


def _is_error_line(line: str) -> bool:
    if any(trigger in line for trigger in _ERROR_TRIGGERS):
        return True
    return "Error" in line


def _extract_error_context(lines: list[str], context: int = _ERROR_CONTEXT_LINES) -> str:
    error_indices = [index for index, line in enumerate(lines) if _is_error_line(line)]
    if not error_indices:
        return "No errors found in log."

    ranges: list[tuple[int, int]] = []
    for index in error_indices:
        start = max(0, index - context)
        end = min(len(lines), index + context + 1)
        if ranges and start <= ranges[-1][1]:
            ranges[-1] = (ranges[-1][0], max(ranges[-1][1], end))
        else:
            ranges.append((start, end))

    chunks: list[str] = []
    for start, end in ranges:
        chunk_lines = [f"{start + offset + 1:6d}|{lines[start + offset]}" for offset in range(end - start)]
        chunks.append("\n".join(chunk_lines))
    return "\n\n---\n\n".join(chunks)


def _subsample_records(records: list[dict[str, object]], max_points: int) -> list[dict[str, object]]:
    if len(records) <= max_points:
        return records
    step = len(records) / max_points
    indices = sorted({min(len(records) - 1, int(index * step)) for index in range(max_points)})
    return [records[index] for index in indices]


def _parse_residual_series(text: str, max_points: int = _RESIDUAL_MAX_POINTS) -> str:
    time_matches = list(re.finditer(r"^Time = ([0-9.eE+\-]+)", text, re.MULTILINE))
    records: list[dict[str, object]] = []

    for index, match in enumerate(time_matches):
        block_start = match.start()
        block_end = time_matches[index + 1].start() if index + 1 < len(time_matches) else len(text)
        block = text[block_start:block_end]
        time_val = float(match.group(1))
        residuals: dict[str, float] = {}
        for residual_match in _RESIDUAL_PATTERN.finditer(block):
            residuals[residual_match.group(1)] = float(residual_match.group(2))
        if residuals:
            records.append({"time": time_val, "residuals": residuals})

    if not records:
        return "No residual data found in log."

    sampled = _subsample_records(records, max_points)
    lines = []
    for record in sampled:
        time_val = record["time"]
        residuals = record["residuals"]
        assert isinstance(time_val, float)
        assert isinstance(residuals, dict)
        fields = "  ".join(f"{name}={value:.3e}" for name, value in sorted(residuals.items()))
        lines.append(f"time={time_val:g}  {fields}")

    header = f"Residual time series ({len(sampled)} points"
    if len(records) > len(sampled):
        header += f", sampled from {len(records)} time steps"
    header += "):"
    return header + "\n" + "\n".join(lines)


def read_log(
    workspace: Path,
    log_path: str,
    mode: str,
    *,
    tail_lines: int = _TAIL_MAX_LINES,
) -> ToolResult:
    target, err = _resolve_in_workspace(workspace, log_path)
    if err:
        return _error(err)
    if not target.exists():
        return _error(f"Log not found: {log_path}")
    if not target.is_file():
        return _error(f"Not a file: {log_path}")

    text = target.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    if mode == "errors":
        content = _extract_error_context(lines)
        return ToolResult(ok=True, content=content)

    if mode == "tail":
        tail = _format_log_tail(text, max_lines=tail_lines)
        return ToolResult(ok=True, content=tail)

    if mode == "residuals":
        content = _parse_residual_series(text)
        return ToolResult(ok=True, content=content)

    return _error(f"Unknown mode: {mode}. Use errors, tail, or residuals.")


def foam_dict_check(workspace: Path, path: str) -> ToolResult:
    target, err = _resolve_in_workspace(workspace, path)
    if err:
        return _error(err)
    if not target.exists():
        return _error(f"Path not found: {path}")
    if not target.is_file():
        return _error(f"Not a file: {path}")

    # foamDictionary -case <dir> は相対パスを解決しないため絶対パスを渡す
    result = run_openfoam(workspace, "foamDictionary", args=[str(target)])
    if result.ok:
        return ToolResult(
            ok=True,
            content=f"foamDictionary OK: {path}",
            data=result.data,
        )
    return ToolResult(
        ok=False,
        content=f"foamDictionary failed: {path}\n{result.content}",
        data=result.data,
    )

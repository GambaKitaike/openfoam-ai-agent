"""ワークスペース配下のファイル操作ツール。"""
from __future__ import annotations

import struct
from pathlib import Path

from .base import ToolResult

_BINARY_EXTENSIONS = {
    ".stl",
    ".gz",
    ".zip",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".pdf",
    ".exe",
    ".bin",
}


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


def _is_processor_dir(name: str) -> bool:
    return name.startswith("processor")


def _is_time_dir(name: str) -> bool:
    try:
        float(name)
        return True
    except ValueError:
        return False


def _should_summarize_dir(name: str) -> bool:
    return _is_processor_dir(name) or name == "postProcessing" or _is_time_dir(name)


def _count_dir_items(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(1 for _ in path.iterdir())


def _format_tree_lines(root: Path, current: Path, prefix: str, depth: int, remaining: int) -> list[str]:
    if not current.is_dir():
        return []

    lines: list[str] = []
    entries = sorted(current.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))

    for index, entry in enumerate(entries):
        is_last = index == len(entries) - 1
        connector = "└── " if is_last else "├── "
        child_prefix = prefix + ("    " if is_last else "│   ")

        if entry.is_dir() and _should_summarize_dir(entry.name):
            count = _count_dir_items(entry)
            lines.append(f"{prefix}{connector}{entry.name}/  ({count} items)")
            continue

        if entry.is_dir():
            lines.append(f"{prefix}{connector}{entry.name}/")
            if remaining > 0:
                lines.extend(_format_tree_lines(root, entry, child_prefix, depth, remaining - 1))
        else:
            lines.append(f"{prefix}{connector}{entry.name}")

    return lines


def list_files(workspace: Path, path: str = ".", depth: int = 2) -> ToolResult:
    target, err = _resolve_in_workspace(workspace, path)
    if err:
        return _error(err)
    if not target.exists():
        return _error(f"Path not found: {path}")
    if not target.is_dir():
        return _error(f"Not a directory: {path}")

    rel = target.resolve().relative_to(workspace.resolve())
    header = f"{rel.as_posix() if str(rel) != '.' else '.'}/"
    body_lines = _format_tree_lines(target, target, "", depth, depth - 1)
    content = header + ("\n" + "\n".join(body_lines) if body_lines else "")
    return ToolResult(ok=True, content=content)


def _is_binary_file(path: Path) -> bool:
    if path.suffix.lower() in _BINARY_EXTENSIONS:
        return True
    with path.open("rb") as handle:
        chunk = handle.read(8192)
    return b"\x00" in chunk


def _read_stl_header(path: Path) -> str:
    size = path.stat().st_size
    with path.open("rb") as handle:
        header_bytes = handle.read(80)
        rest = handle.read(4)

    header_text = header_bytes.decode("ascii", errors="replace").strip()
    if header_text.lower().startswith("solid"):
        preview = header_text[:80]
        return (
            f"Binary STL: no (ASCII)\n"
            f"Size: {size} bytes\n"
            f"Header: {preview!r}"
        )

    triangle_count = struct.unpack("<I", rest)[0] if len(rest) == 4 else 0
    header_preview = header_bytes.decode("ascii", errors="replace").strip()
    if not header_preview:
        header_preview = "(empty 80-byte header)"
    return (
        f"Binary STL: yes\n"
        f"Size: {size} bytes\n"
        f"Header: {header_preview!r}\n"
        f"Triangle count (from header): {triangle_count}"
    )


def _read_binary_summary(path: Path) -> str:
    size = path.stat().st_size
    if path.suffix.lower() == ".stl":
        return _read_stl_header(path)
    with path.open("rb") as handle:
        preview = handle.read(64)
    hex_preview = preview.hex()
    return (
        f"Binary file: {path.name}\n"
        f"Size: {size} bytes\n"
        f"First 64 bytes (hex): {hex_preview}"
    )


def read_file(
    workspace: Path,
    path: str,
    line_range: tuple[int, int] | None = None,
) -> ToolResult:
    target, err = _resolve_in_workspace(workspace, path)
    if err:
        return _error(err)
    if not target.exists():
        return _error(f"Path not found: {path}")
    if not target.is_file():
        return _error(f"Not a file: {path}")

    if _is_binary_file(target):
        return ToolResult(ok=True, content=_read_binary_summary(target))

    text = target.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    if line_range is not None:
        start, end = line_range
        if start < 1 or end < start:
            return _error(f"Invalid line_range: {line_range}")
        selected = lines[start - 1 : end]
        numbered = [f"{start + index:6d}|{line}" for index, line in enumerate(selected)]
        content = "\n".join(numbered)
    else:
        numbered = [f"{index + 1:6d}|{line}" for index, line in enumerate(lines)]
        content = "\n".join(numbered)

    return ToolResult(ok=True, content=content)


def edit_file(workspace: Path, path: str, old_str: str, new_str: str) -> ToolResult:
    target, err = _resolve_in_workspace(workspace, path)
    if err:
        return _error(err)
    if not target.exists():
        return _error(f"Path not found: {path}")
    if not target.is_file():
        return _error(f"Not a file: {path}")

    text = target.read_text(encoding="utf-8", errors="replace")
    count = text.count(old_str)
    if count == 0:
        return _error(f"old_str not found in {path}")
    if count > 1:
        return _error(f"old_str is not unique in {path} ({count} occurrences)")

    updated = text.replace(old_str, new_str, 1)
    target.write_text(updated, encoding="utf-8")
    return ToolResult(ok=True, content=f"Updated {path}")


def write_file(workspace: Path, path: str, content: str) -> ToolResult:
    target, err = _resolve_in_workspace(workspace, path)
    if err:
        return _error(err)
    if target.exists():
        return _error(f"File already exists: {path}. Use edit_file to modify it.")

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return ToolResult(ok=True, content=f"Created {path}")

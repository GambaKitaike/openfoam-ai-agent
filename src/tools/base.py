"""全ツール共通の戻り値型。"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ToolResult:
    ok: bool
    content: str
    data: dict | None = None

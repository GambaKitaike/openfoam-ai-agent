"""fs_tools の単体テスト。"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tools.fs_tools import edit_file, list_files, read_file, write_file


def _write_binary_stl(path: Path, triangle_count: int = 1) -> None:
    with path.open("wb") as handle:
        handle.write(b"OpenFOAM mesh" + b"\x00" * 67)
        handle.write(struct.pack("<I", triangle_count))
        handle.write(b"\x00" * (50 * triangle_count))


class TestListFiles:
    def test_tree_with_summarized_dirs(self, tmp_path: Path) -> None:
        (tmp_path / "system").mkdir()
        (tmp_path / "system" / "controlDict").write_text("application simpleFoam;")
        (tmp_path / "0").mkdir()
        (tmp_path / "0" / "U").write_text("internalField uniform (1 0 0);")
        (tmp_path / "0" / "p").write_text("internalField uniform 0;")
        (tmp_path / "postProcessing").mkdir()
        (tmp_path / "postProcessing" / "forces").mkdir()
        (tmp_path / "postProcessing" / "probes").mkdir()
        (tmp_path / "processor0").mkdir()
        (tmp_path / "processor0" / "0").mkdir()
        (tmp_path / "processor0" / "0" / "U").write_text("data")

        result = list_files(tmp_path, depth=3)

        assert result.ok is True
        assert "postProcessing/  (2 items)" in result.content
        assert "0/  (2 items)" in result.content
        assert "processor0/  (1 items)" in result.content
        assert "controlDict" in result.content
        assert "0/U" not in result.content

    def test_path_not_found(self, tmp_path: Path) -> None:
        result = list_files(tmp_path, path="missing")
        assert result.ok is False
        assert "not found" in result.content.lower()


class TestReadFile:
    def test_text_file_with_line_range(self, tmp_path: Path) -> None:
        target = tmp_path / "system" / "controlDict"
        target.parent.mkdir()
        target.write_text("line1\nline2\nline3\n")

        result = read_file(tmp_path, "system/controlDict", line_range=(2, 3))

        assert result.ok is True
        assert "     2|line2" in result.content
        assert "     3|line3" in result.content
        assert "line1" not in result.content

    def test_binary_stl_returns_header_only(self, tmp_path: Path) -> None:
        stl = tmp_path / "body.stl"
        _write_binary_stl(stl, triangle_count=12)

        result = read_file(tmp_path, "body.stl")

        assert result.ok is True
        assert "Binary STL: yes" in result.content
        assert "Size:" in result.content
        assert "Triangle count (from header): 12" in result.content
        assert "internalField" not in result.content


class TestEditFile:
    def test_successful_replace(self, tmp_path: Path) -> None:
        target = tmp_path / "system" / "controlDict"
        target.parent.mkdir()
        target.write_text("endTime 1;\n")

        result = edit_file(tmp_path, "system/controlDict", "endTime 1;", "endTime 2;")

        assert result.ok is True
        assert target.read_text() == "endTime 2;\n"

    def test_old_str_not_found(self, tmp_path: Path) -> None:
        target = tmp_path / "note.txt"
        target.write_text("hello")

        result = edit_file(tmp_path, "note.txt", "missing", "new")

        assert result.ok is False
        assert "not found" in result.content.lower()

    def test_old_str_not_unique(self, tmp_path: Path) -> None:
        target = tmp_path / "note.txt"
        target.write_text("foo bar foo")

        result = edit_file(tmp_path, "note.txt", "foo", "baz")

        assert result.ok is False
        assert "not unique" in result.content.lower()
        assert target.read_text() == "foo bar foo"

    def test_reject_noop_edit(self, tmp_path: Path) -> None:
        target = tmp_path / "note.txt"
        original = "endTime 1;\n"
        target.write_text(original)

        result = edit_file(tmp_path, "note.txt", "endTime 1;", "endTime 1;")

        assert result.ok is False
        assert "同一" in result.content
        assert target.read_text() == original


class TestWriteFile:
    def test_create_new_file(self, tmp_path: Path) -> None:
        result = write_file(tmp_path, "constant/transportProperties", "nu 1e-5;")

        assert result.ok is True
        created = tmp_path / "constant" / "transportProperties"
        assert created.read_text() == "nu 1e-5;"

    def test_reject_existing_file(self, tmp_path: Path) -> None:
        target = tmp_path / "existing.txt"
        target.write_text("original")

        result = write_file(tmp_path, "existing.txt", "new content")

        assert result.ok is False
        assert "already exists" in result.content.lower()
        assert target.read_text() == "original"


class TestPathTraversal:
    def test_read_file_rejects_escape(self, tmp_path: Path) -> None:
        outside = tmp_path.parent / "outside.txt"
        outside.write_text("secret")

        result = read_file(tmp_path, "../outside.txt")

        assert result.ok is False
        assert "escapes workspace" in result.content.lower()

    def test_edit_file_rejects_escape(self, tmp_path: Path) -> None:
        outside = tmp_path.parent / "outside.txt"
        outside.write_text("secret")

        result = edit_file(tmp_path, "../outside.txt", "secret", "changed")

        assert result.ok is False
        assert outside.read_text() == "secret"

    def test_write_file_rejects_escape(self, tmp_path: Path) -> None:
        result = write_file(tmp_path, "../escaped.txt", "data")

        assert result.ok is False
        assert not (tmp_path.parent / "escaped.txt").exists()

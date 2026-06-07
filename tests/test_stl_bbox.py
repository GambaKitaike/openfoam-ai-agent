"""STL バウンディングボックス解析のユニットテスト"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.openfoam_gpt import OpenFOAMGPTAgent


def _write_binary_stl(path: Path, triangles: list) -> None:
    with open(path, "wb") as f:
        f.write(b"\x00" * 80)
        f.write(struct.pack("<I", len(triangles)))
        for normal, v1, v2, v3 in triangles:
            f.write(struct.pack("<3f", *normal))
            f.write(struct.pack("<3f", *v1))
            f.write(struct.pack("<3f", *v2))
            f.write(struct.pack("<3f", *v3))
            f.write(struct.pack("<H", 0))


def _write_ascii_stl(path: Path, triangles: list) -> None:
    lines = ["solid test"]
    for normal, v1, v2, v3 in triangles:
        lines += [
            f"  facet normal {normal[0]} {normal[1]} {normal[2]}",
            "    outer loop",
            f"      vertex {v1[0]} {v1[1]} {v1[2]}",
            f"      vertex {v2[0]} {v2[1]} {v2[2]}",
            f"      vertex {v3[0]} {v3[1]} {v3[2]}",
            "    endloop",
            "  endfacet",
        ]
    lines.append("endsolid test")
    path.write_text("\n".join(lines))


def _unit_cube_tris():
    """単位立方体（0〜1）の STL 三角形リスト（12三角形）"""
    return [
        ((0, 0, -1), (0, 0, 0), (1, 0, 0), (1, 1, 0)),
        ((0, 0, -1), (0, 0, 0), (1, 1, 0), (0, 1, 0)),
        ((0, 0, 1),  (0, 0, 1), (0, 1, 1), (1, 1, 1)),
        ((0, 0, 1),  (0, 0, 1), (1, 1, 1), (1, 0, 1)),
        ((-1, 0, 0), (0, 0, 0), (0, 1, 0), (0, 1, 1)),
        ((-1, 0, 0), (0, 0, 0), (0, 1, 1), (0, 0, 1)),
        ((1, 0, 0),  (1, 0, 0), (1, 0, 1), (1, 1, 1)),
        ((1, 0, 0),  (1, 0, 0), (1, 1, 1), (1, 1, 0)),
        ((0, -1, 0), (0, 0, 0), (1, 0, 0), (1, 0, 1)),
        ((0, -1, 0), (0, 0, 0), (1, 0, 1), (0, 0, 1)),
        ((0, 1, 0),  (0, 1, 0), (0, 1, 1), (1, 1, 1)),
        ((0, 1, 0),  (0, 1, 0), (1, 1, 1), (1, 1, 0)),
    ]


class TestAnalyzeStlBbox:
    def test_binary_stl_unit_cube(self, tmp_path):
        stl = tmp_path / "cube.stl"
        _write_binary_stl(stl, _unit_cube_tris())
        bbox = OpenFOAMGPTAgent._analyze_stl_bbox(stl)
        assert bbox["x_min"] == pytest.approx(0.0, abs=1e-5)
        assert bbox["x_max"] == pytest.approx(1.0, abs=1e-5)
        assert bbox["y_min"] == pytest.approx(0.0, abs=1e-5)
        assert bbox["y_max"] == pytest.approx(1.0, abs=1e-5)
        assert bbox["z_min"] == pytest.approx(0.0, abs=1e-5)
        assert bbox["z_max"] == pytest.approx(1.0, abs=1e-5)

    def test_ascii_stl_unit_cube(self, tmp_path):
        stl = tmp_path / "cube_ascii.stl"
        _write_ascii_stl(stl, _unit_cube_tris())
        bbox = OpenFOAMGPTAgent._analyze_stl_bbox(stl)
        assert bbox["x_min"] == pytest.approx(0.0, abs=1e-5)
        assert bbox["x_max"] == pytest.approx(1.0, abs=1e-5)

    def test_empty_stl_returns_default(self, tmp_path):
        """壊れた STL はデフォルト bbox を返す"""
        stl = tmp_path / "broken.stl"
        stl.write_bytes(b"\x00" * 10)  # 壊れたファイル
        bbox = OpenFOAMGPTAgent._analyze_stl_bbox(stl)
        assert "x_min" in bbox
        assert "x_max" in bbox

    def test_2d_cylinder_stl_thin_z(self, tmp_path):
        """2D 円柱 STL は z が薄い（0〜0.01m 程度）ことを確認"""
        # 実際の generate_cylinder_2d_stl.py が生成するファイルを使用
        real_stl = Path(__file__).parent.parent / "cylinder_2d.stl"
        if not real_stl.exists():
            pytest.skip("cylinder_2d.stl が見つかりません")
        bbox = OpenFOAMGPTAgent._analyze_stl_bbox(real_stl)
        z_thickness = bbox["z_max"] - bbox["z_min"]
        assert z_thickness < 0.1, f"2D STL の z 厚みが厚すぎます: {z_thickness}"
        # x, y は円柱直径 0.1m 程度
        x_size = bbox["x_max"] - bbox["x_min"]
        assert 0.05 < x_size < 0.2, f"円柱 x サイズが想定外: {x_size}"

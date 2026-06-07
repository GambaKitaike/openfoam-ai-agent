"""Jinja2 テンプレートのレンダリングテスト

LLM・OpenFOAM 不要。テンプレートが構文エラーなくレンダリングできることと
出力にキーワードが含まれることを確認する。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from jinja2 import Environment, FileSystemLoader, TemplateSyntaxError

sys.path.insert(0, str(Path(__file__).parent.parent))

TEMPLATES_DIR = Path(__file__).parent.parent / "templates"


@pytest.fixture(scope="module")
def jinja() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        trim_blocks=True,
        lstrip_blocks=True,
    )


# ─────────────────────────────────────────────
# 共通コンテキスト
# ─────────────────────────────────────────────

def _base_ctx(**overrides) -> dict:
    ctx = {
        "solver": "simpleFoam",
        "case_type": "channel_2d",
        "turbulence_model": "kOmegaSST",
        "steady_state": True,
        "dimensions": 2,
        "inlet_velocity": 1.0,
        "has_wall": True,
        "snappy_object_name": "cylinder",
        "is_snappy_2d": False,
        "end_time": 1000,
        "delta_t": 1,
        "write_interval": 100,
        # blockMeshDict パラメータ
        "x_min": -5.0, "x_max": 15.0, "ly": 5.0, "depth": 0.1,
        "nx": 40, "ny": 20,
        # snappy パラメータ（snappy テンプレート用）
        "stl_name": "cylinder.stl",
        "x_max_snappy": 4.5, "y_min": -1.0, "y_max": 1.0,
        "z_min": 0.0, "z_max": 0.01,
        "nz": 1,
        "domain_scale": 10.0,
    }
    ctx.update(overrides)
    return ctx


# ─────────────────────────────────────────────
# blockMeshDict テンプレート
# ─────────────────────────────────────────────

class TestBlockMeshDictTemplates:
    templates = [
        "system/blockMeshDict/box_channel_2d.j2",
        "system/blockMeshDict/box_channel_3d.j2",
        "system/blockMeshDict/box_snappy.j2",
        "system/blockMeshDict/box_snappy_2d.j2",
    ]

    def test_no_syntax_error(self, jinja):
        for tmpl_path in self.templates:
            try:
                jinja.get_template(tmpl_path)
            except TemplateSyntaxError as e:
                pytest.fail(f"{tmpl_path} に構文エラー: {e}")

    def test_box_channel_2d_renders(self, jinja):
        ctx = _base_ctx()
        out = jinja.get_template("system/blockMeshDict/box_channel_2d.j2").render(**ctx)
        assert "FoamFile" in out
        assert "vertices" in out
        assert "empty" in out  # 2D: front/back は empty

    def test_box_snappy_2d_renders(self, jinja):
        ctx = _base_ctx(stl_name="cylinder_2d.stl", nx=80, ny=40)
        out = jinja.get_template("system/blockMeshDict/box_snappy_2d.j2").render(**ctx)
        assert "FoamFile" in out
        assert "empty" in out       # 2D: front/back は empty
        assert "0.01" in out        # z=0.01 固定

    def test_box_snappy_3d_renders(self, jinja):
        ctx = _base_ctx(
            stl_name="body.stl", nx=30, ny=20, nz=20,
            x_min=-0.6, x_max=1.4, y_min=-0.5, y_max=0.5,
            z_min=-0.5, z_max=0.5,
        )
        out = jinja.get_template("system/blockMeshDict/box_snappy.j2").render(**ctx)
        assert "FoamFile" in out
        assert "symmetryPlane" in out   # 3D snappy: front/back は symmetryPlane

    def test_box_channel_3d_renders(self, jinja):
        ctx = _base_ctx(
            x_min=0.0, x_max=10.0, ly=1.0, lz=1.0,
            nx=60, ny=20, nz=20,
        )
        out = jinja.get_template("system/blockMeshDict/box_channel_3d.j2").render(**ctx)
        assert "FoamFile" in out
        assert "noSlip" in out or "wall" in out.lower()


# ─────────────────────────────────────────────
# 0/ テンプレート（境界条件）
# ─────────────────────────────────────────────

class TestZeroDirTemplates:
    fields = ["U", "p", "k", "omega", "nut"]

    @pytest.mark.parametrize("field", fields)
    def test_channel_2d_has_empty_patches(self, jinja, field):
        ctx = _base_ctx(case_type="channel_2d")
        out = jinja.get_template(f"0/{field}.j2").render(**ctx)
        assert "empty" in out, f"0/{field}: channel_2d に empty パッチがない"

    @pytest.mark.parametrize("field", fields)
    def test_snappy_2d_has_empty_front_back(self, jinja, field):
        ctx = _base_ctx(case_type="snappy_2d", is_snappy_2d=True)
        out = jinja.get_template(f"0/{field}.j2").render(**ctx)
        assert "empty" in out, f"0/{field}: snappy_2d に empty パッチがない"
        assert "cylinder" in out, f"0/{field}: snappy_2d にSTL物体パッチがない"

    @pytest.mark.parametrize("field", fields)
    def test_snappy_2d_has_symmetryplane_top_bottom(self, jinja, field):
        ctx = _base_ctx(case_type="snappy_2d", is_snappy_2d=True)
        out = jinja.get_template(f"0/{field}.j2").render(**ctx)
        assert "symmetryPlane" in out, f"0/{field}: snappy_2d に symmetryPlane がない"

    @pytest.mark.parametrize("field", fields)
    def test_external_snappy_no_empty(self, jinja, field):
        """3D external_snappy: front/back は symmetryPlane (empty ではない)"""
        ctx = _base_ctx(case_type="external_snappy", is_snappy_2d=False)
        out = jinja.get_template(f"0/{field}.j2").render(**ctx)
        assert "symmetryPlane" in out, f"0/{field}: external_snappy に symmetryPlane がない"


class TestUBoundaryConditions:
    def test_inlet_velocity(self, jinja):
        ctx = _base_ctx(inlet_velocity=5.0)
        out = jinja.get_template("0/U.j2").render(**ctx)
        assert "5" in out  # inlet_velocity が反映される

    def test_noSlip_on_wall(self, jinja):
        ctx = _base_ctx(case_type="channel_2d")
        out = jinja.get_template("0/U.j2").render(**ctx)
        assert "noSlip" in out

    def test_snappy_2d_cylinder_noSlip(self, jinja):
        ctx = _base_ctx(case_type="snappy_2d", is_snappy_2d=True, snappy_object_name="cylinder_2d")
        out = jinja.get_template("0/U.j2").render(**ctx)
        assert "cylinder_2d" in out
        assert "noSlip" in out


# ─────────────────────────────────────────────
# system/ テンプレート
# ─────────────────────────────────────────────

class TestSystemTemplates:
    def test_fvSchemes_steady_has_bounded(self, jinja):
        ctx = _base_ctx(steady_state=True)
        out = jinja.get_template("system/fvSchemes.j2").render(**ctx)
        assert "bounded" in out

    def test_fvSchemes_unsteady_no_bounded_in_div(self, jinja):
        ctx = _base_ctx(steady_state=False)
        out = jinja.get_template("system/fvSchemes.j2").render(**ctx)
        # ddtSchemes が Euler になっているか
        assert "Euler" in out or "backward" in out

    def test_fvSolution_steady_has_SIMPLE(self, jinja):
        ctx = _base_ctx(steady_state=True)
        out = jinja.get_template("system/fvSolution.j2").render(**ctx)
        assert "SIMPLE" in out
        assert "PIMPLE" not in out

    def test_fvSolution_unsteady_has_PIMPLE(self, jinja):
        ctx = _base_ctx(steady_state=False)
        out = jinja.get_template("system/fvSolution.j2").render(**ctx)
        assert "PIMPLE" in out

    def test_controlDict_steady_no_adjustTimeStep(self, jinja):
        ctx = _base_ctx(steady_state=True)
        out = jinja.get_template("system/controlDict.j2").render(**ctx)
        assert "adjustTimeStep" not in out

    def test_controlDict_unsteady_has_adjustTimeStep(self, jinja):
        ctx = _base_ctx(steady_state=False, end_time=10.0, delta_t=0.001, write_interval=0.1)
        out = jinja.get_template("system/controlDict.j2").render(**ctx)
        assert "adjustTimeStep" in out
        assert "maxCo" in out
        assert "purgeWrite" in out

    def test_controlDict_unsteady_has_purgeWrite(self, jinja):
        ctx = _base_ctx(steady_state=False, end_time=10.0, delta_t=0.001, write_interval=0.1)
        out = jinja.get_template("system/controlDict.j2").render(**ctx)
        assert "purgeWrite" in out

    def test_fvSolution_unsteady_has_pFinal(self, jinja):
        ctx = _base_ctx(steady_state=False)
        out = jinja.get_template("system/fvSolution.j2").render(**ctx)
        assert "pFinal" in out

    @pytest.mark.parametrize("tmpl", ["system/fvSchemes.j2", "system/fvSolution.j2", "system/controlDict.j2"])
    def test_no_syntax_error(self, jinja, tmpl):
        try:
            jinja.get_template(tmpl)
        except TemplateSyntaxError as e:
            pytest.fail(f"{tmpl} に構文エラー: {e}")

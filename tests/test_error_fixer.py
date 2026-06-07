"""ルールベースエラー修正モジュールのユニットテスト"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.error_fixer import apply_rule_based_fixes, _add_missing_patch_entry


def _make_case(tmp_path: Path, fields: dict[str, str]) -> Path:
    """テスト用 OpenFOAM ケースを作成する"""
    case = tmp_path / "test_case"
    zero = case / "0"
    system = case / "system"
    zero.mkdir(parents=True)
    system.mkdir()
    for fname, content in fields.items():
        (zero / fname).write_text(content)
    return case


# ─────────────────────────────────────────────
# Rule 1: patch type 不整合
# ─────────────────────────────────────────────

class TestPatchTypeMismatch:
    def _p_file_with_wrong_top(self) -> str:
        # front が zeroGradient になっているが、メッシュでは empty と定義されている（不整合）
        return """FoamFile { version 2.0; format ascii; class volScalarField; object p; }
dimensions [0 2 -2 0 0 0 0];
internalField uniform 0;
boundaryField
{
    inlet  { type zeroGradient; }
    outlet { type fixedValue; value uniform 0; }
    top    { type zeroGradient; }
    bottom { type zeroGradient; }
    front  { type zeroGradient; }
    back   { type zeroGradient; }
}"""

    def test_empty_patch_fixed_to_empty_type(self, tmp_path):
        """メッシュで empty のパッチが zeroGradient になっていたら empty に直す"""
        case = _make_case(tmp_path, {"p": self._p_file_with_wrong_top()})
        error_log = (
            "inconsistent patch and patchField types for\n"
            "    patch type empty and patchField type zeroGradient\n"
            "file: 0/p/boundaryField/front at line 12.\n"
        )
        fixed = apply_rule_based_fixes(str(case), error_log)
        assert fixed is True
        content = (case / "0" / "p").read_text()
        # front が empty になっているか確認
        assert re.search(r"front\s*\{\s*type\s+empty", content)

    def test_no_fix_when_no_match(self, tmp_path):
        """既知パターンに一致しないエラーは False を返す"""
        case = _make_case(tmp_path, {"p": "FoamFile{} boundaryField{}"})
        fixed = apply_rule_based_fixes(str(case), "some unrelated error message")
        assert fixed is False


import re


# ─────────────────────────────────────────────
# Rule 2: 不足パッチエントリの補完
# ─────────────────────────────────────────────

class TestMissingPatchEntry:
    def _p_without_cylinder(self) -> str:
        return """FoamFile { version 2.0; format ascii; class volScalarField; object p; }
dimensions [0 2 -2 0 0 0 0];
internalField uniform 0;
boundaryField
{
    inlet  { type zeroGradient; }
    outlet { type fixedValue; value uniform 0; }
    top    { type symmetryPlane; }
    bottom { type symmetryPlane; }
    front  { type empty; }
    back   { type empty; }
}"""

    def test_missing_patch_added(self, tmp_path):
        """Cannot find patchField entry for cylinder → 0/ に追加される"""
        case = _make_case(tmp_path, {"p": self._p_without_cylinder()})
        error_log = "Cannot find patchField entry for cylinder\n"
        fixed = apply_rule_based_fixes(str(case), error_log)
        assert fixed is True
        content = (case / "0" / "p").read_text()
        assert "cylinder" in content

    def test_no_duplicate_when_patch_exists(self, tmp_path):
        """すでにパッチがある場合は追加しない"""
        p_with_cylinder = self._p_without_cylinder().replace(
            "    front", "    cylinder  { type zeroGradient; }\n    front"
        )
        case = _make_case(tmp_path, {"p": p_with_cylinder})
        zero_path = case / "0"
        initial_count = p_with_cylinder.count("cylinder")
        _add_missing_patch_entry(zero_path, "cylinder")
        content = (case / "0" / "p").read_text()
        assert content.count("cylinder") == initial_count  # 増えていない


# ─────────────────────────────────────────────
# Rule 3: UFinal が fvSolution にない
# ─────────────────────────────────────────────

class TestUFinalMissing:
    def _fvsolution_without_ufinal(self) -> str:
        return """FoamFile { version 2.0; }
solvers
{
    p { solver GAMG; }
    U { solver PBiCGStab; }
}
PIMPLE
{
    nOuterCorrectors 3;
    nCorrectors 2;
}"""

    def test_ufinal_added_when_missing(self, tmp_path):
        case = _make_case(tmp_path, {})
        (case / "system").mkdir(exist_ok=True)
        (case / "system" / "fvSolution").write_text(self._fvsolution_without_ufinal())
        error_log = 'Entry \'UFinal\' not found in dictionary "system/fvSolution/solvers"\n'
        fixed = apply_rule_based_fixes(str(case), error_log)
        assert fixed is True
        content = (case / "system" / "fvSolution").read_text()
        assert "UFinal" in content or "pFinal" in content


# ─────────────────────────────────────────────
# Rule 4: Courant 爆発 → nOuterCorrectors 増加
# ─────────────────────────────────────────────

class TestCourantExplosion:
    def _fvsolution_low_correctors(self) -> str:
        return """FoamFile { version 2.0; }
solvers { p { solver GAMG; } }
PIMPLE
{
    nOuterCorrectors    3;
    nCorrectors         2;
}"""

    def test_nOuterCorrectors_increased(self, tmp_path):
        case = _make_case(tmp_path, {})
        (case / "system").mkdir(exist_ok=True)
        (case / "system" / "fvSolution").write_text(self._fvsolution_low_correctors())
        error_log = "Courant Number mean: 1.4 max: 87.5\n"
        fixed = apply_rule_based_fixes(str(case), error_log)
        assert fixed is True
        content = (case / "system" / "fvSolution").read_text()
        m = re.search(r"nOuterCorrectors\s+(\d+)", content)
        assert m and int(m.group(1)) >= 5

    def test_no_change_when_already_high(self, tmp_path):
        """すでに nOuterCorrectors が 5 以上なら変えない"""
        fvsol = self._fvsolution_low_correctors().replace(
            "nOuterCorrectors    3", "nOuterCorrectors    5"
        )
        case = _make_case(tmp_path, {})
        (case / "system").mkdir(exist_ok=True)
        (case / "system" / "fvSolution").write_text(fvsol)
        error_log = "Courant Number mean: 1.4 max: 87.5\n"
        apply_rule_based_fixes(str(case), error_log)
        content = (case / "system" / "fvSolution").read_text()
        m = re.search(r"nOuterCorrectors\s+(\d+)", content)
        assert m and int(m.group(1)) == 5  # 変わっていない

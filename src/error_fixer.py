"""
ルールベースエラー事前修正モジュール

LLM に頼る前に、よくある OpenFOAM エラーパターンをパターンマッチで修正する。
修正できた場合は True を返し、呼び出し元がリトライする。
修正できなかった場合は False を返し、呼び出し元が LLM に委譲する。
"""
from __future__ import annotations

import re
from pathlib import Path

from rich.console import Console

console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# patch type 不整合の自動修正
# ─────────────────────────────────────────────────────────────────────────────

# エラー例:
#   inconsistent patch and patchField types for
#       patch type empty and patchField type zeroGradient
_PATCH_MISMATCH_RE = re.compile(
    r"inconsistent patch and patchField types for\s+"
    r"patch type (\w+) and patchField type (\w+)",
    re.IGNORECASE,
)

# エラー例:
#   Cannot find patchField entry for <name>
_MISSING_PATCH_RE = re.compile(
    r"Cannot find patchField entry for (\w+)",
    re.IGNORECASE,
)

# patch type → 有効な patchField type のマッピング
_PATCH_TYPE_DEFAULTS: dict[str, dict[str, str]] = {
    "empty": {
        "U":     "empty",
        "p":     "empty",
        "k":     "empty",
        "omega": "empty",
        "nut":   "empty",
        "epsilon": "empty",
        "nuTilda": "empty",
        "default": "empty",
    },
    "symmetryPlane": {
        "U":     "symmetryPlane",
        "p":     "symmetryPlane",
        "k":     "symmetryPlane",
        "omega": "symmetryPlane",
        "nut":   "symmetryPlane",
        "epsilon": "symmetryPlane",
        "nuTilda": "symmetryPlane",
        "default": "symmetryPlane",
    },
    "wall": {
        "U":     "noSlip",
        "p":     "zeroGradient",
        "k":     "kqRWallFunction",
        "omega": "omegaWallFunction",
        "nut":   "nutkWallFunction",
        "epsilon": "epsilonWallFunction",
        "nuTilda": "zeroGradient",
        "default": "zeroGradient",
    },
}


def _fix_patch_field_type(zero_path: Path, patch_name: str, mesh_patch_type: str) -> bool:
    """0/ ファイルの特定パッチの patchField type をメッシュ定義に合わせて修正する。"""
    fixed_any = False
    for field_file in zero_path.glob("*"):
        if not field_file.is_file() or field_file.name.startswith("."):
            continue
        content = field_file.read_text(errors="ignore")
        if patch_name not in content:
            continue

        field_name = field_file.name  # "U", "p", "k", etc.
        correct_type = _PATCH_TYPE_DEFAULTS.get(mesh_patch_type, {}).get(
            field_name, _PATCH_TYPE_DEFAULTS.get(mesh_patch_type, {}).get("default", "zeroGradient")
        )

        # patch_name { type <something>; ... } を書き換える
        pattern = re.compile(
            rf"(\s*{re.escape(patch_name)}\s*\{{[^}}]*?type\s+)(\w+)(\s*;)",
            re.DOTALL,
        )
        new_content, count = pattern.subn(rf"\g<1>{correct_type}\g<3>", content)
        if count > 0 and new_content != content:
            field_file.write_text(new_content)
            console.print(
                f"  [dim][ルール修正] 0/{field_name}: {patch_name} → type {correct_type}[/dim]"
            )
            fixed_any = True

    return fixed_any


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def apply_rule_based_fixes(case_dir: str, error_log: str) -> bool:
    """
    エラーログを解析し、既知パターンを自動修正する。

    Parameters
    ----------
    case_dir : str
        OpenFOAM ケースディレクトリ
    error_log : str
        ソルバー / blockMesh の stdout + stderr

    Returns
    -------
    bool
        True: 修正を適用した（リトライを推奨）
        False: 既知パターンに一致しなかった（LLM に委譲）
    """
    case_path = Path(case_dir)
    zero_path = case_path / "0"
    system_path = case_path / "system"
    fixed = False

    # ── Rule 1: patch type と patchField type の不整合 ──────────────────
    for m in _PATCH_MISMATCH_RE.finditer(error_log):
        mesh_type = m.group(1).lower()   # "empty", "symmetryPlane", "wall", ...
        # どのパッチが問題か特定（エラー行の前後から patch 名を抽出）
        # "file: 0/p/boundaryField/top at line ..." の形式で patch 名が出る
        patch_name_match = re.search(r"boundaryField/(\w+)\s+at line", error_log)
        if patch_name_match:
            patch_name = patch_name_match.group(1)
            console.print(
                f"  [yellow][ルール検知] patch '{patch_name}' の型不整合 "
                f"(メッシュ: {mesh_type}) → 0/ を修正します[/yellow]"
            )
            if _fix_patch_field_type(zero_path, patch_name, mesh_type):
                fixed = True

    # ── Rule 2: patchField エントリが存在しない ──────────────────────────
    for m in _MISSING_PATCH_RE.finditer(error_log):
        patch_name = m.group(1)
        console.print(
            f"  [yellow][ルール検知] パッチ '{patch_name}' の BC エントリが 0/ にない → "
            f"noSlip/zeroGradient で補完します[/yellow]"
        )
        if _add_missing_patch_entry(zero_path, patch_name):
            fixed = True

    # ── Rule 3: UFinal / pFinal が fvSolution に存在しない ──────────────
    if "UFinal" in error_log and "not found in dictionary" in error_log:
        fvsol_path = system_path / "fvSolution"
        if fvsol_path.exists():
            content = fvsol_path.read_text()
            if "UFinal" not in content and "PIMPLE" in content:
                # pFinal / UFinal ブロックを solvers に追加
                insert = (
                    '\n    pFinal\n    {\n        $p;\n        relTol 0;\n    }\n'
                    '    "(U|k|omega|epsilon|nuTilda)Final"\n    {\n'
                    '        $U;\n        relTol 0;\n    }\n'
                )
                new_content = content.replace(
                    "solvers\n{",
                    "solvers\n{" + insert,
                    1,
                )
                if new_content != content:
                    fvsol_path.write_text(new_content)
                    console.print("  [dim][ルール修正] fvSolution: pFinal/UFinal ブロックを追加[/dim]")
                    fixed = True

    # ── Rule 4: Courant 爆発 → PIMPLE nOuterCorrectors を増やす ────────
    if re.search(r"Courant Number\s+\w+:\s+[0-9.e+]+\s+max:\s+[5-9][0-9]|max:\s+[0-9]{3,}", error_log):
        fvsol_path = system_path / "fvSolution"
        if fvsol_path.exists():
            content = fvsol_path.read_text()
            # nOuterCorrectors を 3 → 5 に増やす（既に5以上なら触らない）
            m_oc = re.search(r"nOuterCorrectors\s+(\d+)", content)
            if m_oc and int(m_oc.group(1)) < 5:
                new_content = content.replace(
                    m_oc.group(0),
                    f"nOuterCorrectors    5",
                )
                fvsol_path.write_text(new_content)
                console.print("  [dim][ルール修正] fvSolution: nOuterCorrectors を 5 に増加[/dim]")
                fixed = True

    return fixed


def _add_missing_patch_entry(zero_path: Path, patch_name: str) -> bool:
    """0/ の各フィールドファイルに不足しているパッチエントリを末尾前に追加する。"""
    fixed = False
    _defaults = {
        "U":       f"    {patch_name}  {{ type noSlip; }}",
        "p":       f"    {patch_name}  {{ type zeroGradient; }}",
        "k":       f"    {patch_name}  {{ type kqRWallFunction; value uniform 0.1; }}",
        "omega":   f"    {patch_name}  {{ type omegaWallFunction; value uniform 1; }}",
        "nut":     f"    {patch_name}  {{ type nutkWallFunction; value uniform 0; }}",
        "epsilon": f"    {patch_name}  {{ type epsilonWallFunction; value uniform 0.1; }}",
    }
    for field_file in zero_path.glob("*"):
        if not field_file.is_file() or field_file.name.startswith("."):
            continue
        fname = field_file.name
        entry = _defaults.get(fname)
        if not entry:
            continue
        content = field_file.read_text(errors="ignore")
        if patch_name in content:
            continue  # 既に存在する
        # boundaryField { ... } の閉じ括弧の直前に挿入
        new_content = re.sub(
            r"(boundaryField\s*\{)(.*?)(\})",
            lambda match: match.group(1) + match.group(2) + "\n" + entry + "\n" + match.group(3),
            content,
            count=1,
            flags=re.DOTALL,
        )
        if new_content != content:
            field_file.write_text(new_content)
            console.print(f"  [dim][ルール修正] 0/{fname}: {patch_name} エントリを追加[/dim]")
            fixed = True
    return fixed

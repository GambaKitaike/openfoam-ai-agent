"""Agent②: 現象別の必要十分条件プロファイル。"""
from __future__ import annotations

import re

from ..models import RequirementField, RequirementProfile, SimulationSpec, SpecReviewAlternative, SpecReviewIssue

LAMINAR_RE_LIMIT = 2300

PHENOMENON_REQUIREMENTS: dict[str, dict] = {
    "karman_vortex_shedding": {
        "fields": [
            ("phenomenon", True, "カルマン渦解析の現象タグ"),
            ("solver", True, "非定常外部流れソルバー (pimpleFoam 推奨)"),
            ("steady_state", True, "渦放出は非定常"),
            ("inlet_velocity", True, "来流速度または Re から決定"),
            ("characteristic_length", True, "円柱直径"),
            ("nu", True, "動粘度または Re から決定"),
            ("turbulence_model", True, "層流/乱流"),
        ],
        "constraints": [
            "case_type は cylinder_2d_ogrid を推奨",
            "karman + O-grid では pimpleFoam 固定",
            "steady_state は false",
        ],
        "defaults": {
            "solver": "pimpleFoam",
            "steady_state": False,
            "case_type": "cylinder_2d_ogrid",
            "turbulence_model": "laminar",
            "characteristic_length": 1.0,
            "inlet_velocity": 1.0,
        },
    },
    "airfoil_steady": {
        "fields": [
            ("inlet_velocity", True, "来流速度"),
            ("characteristic_length", True, "弦長"),
            ("turbulence_model", True, "乱流モデル"),
            ("solver", True, "定常ソルバー"),
        ],
        "constraints": ["定常解析が一般的"],
        "defaults": {"solver": "simpleFoam", "steady_state": True},
    },
    "channel_internal": {
        "fields": [
            ("inlet_velocity", True, "流入速度"),
            ("turbulence_model", True, "乱流モデル"),
        ],
        "constraints": [],
        "defaults": {"solver": "simpleFoam", "steady_state": True},
    },
}


def _parse_re_from_description(description: str) -> float | None:
    m = re.search(r"Re\s*[=:：]?\s*([\d.eE+\-]+)", description, re.IGNORECASE)
    if not m:
        m = re.search(r"レイノルズ数\s*[=:：]?\s*([\d.eE+\-]+)", description)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


def _field_satisfied(
    spec: SimulationSpec,
    key: str,
    defaults: set[str],
    target_re: float | None = None,
) -> bool:
    if key in defaults:
        return True
    if key == "phenomenon":
        return bool(spec.phenomenon)
    if key == "steady_state":
        if spec.phenomenon == "karman_vortex_shedding":
            return spec.steady_state is False
        return True
    if key == "nu" and target_re and spec.inlet_velocity and spec.characteristic_length:
        expected = spec.inlet_velocity * spec.characteristic_length / target_re
        if spec.nu and abs(spec.nu - expected) / max(expected, 1e-12) < 0.05:
            return True
    val = getattr(spec, key, None)
    if val is None:
        return False
    if isinstance(val, str):
        return bool(val.strip())
    if isinstance(val, (int, float)):
        return val != 0
    return True


def build_requirement_profile(
    spec: SimulationSpec,
    description: str = "",
    similar_case_ids: list[str] | None = None,
) -> RequirementProfile:
    """SimulationSpec と説明文から未充足項目を列挙。"""
    phenomenon = spec.phenomenon or "general"
    cfg = PHENOMENON_REQUIREMENTS.get(phenomenon, {})
    defaults_set = set(spec.defaults_applied or [])
    target_re = _parse_re_from_description(description)

    fields: list[RequirementField] = []
    for key, required, reason in cfg.get("fields", []):
        if _field_satisfied(spec, key, defaults_set, target_re):
            continue
        suggested = cfg.get("defaults", {}).get(key)
        if key == "steady_state":
            suggested = cfg.get("defaults", {}).get("steady_state", False)
        if key == "phenomenon" and not suggested:
            suggested = phenomenon
        if key == "inlet_velocity" and target_re and spec.nu and spec.characteristic_length:
            suggested = round(target_re * spec.nu / spec.characteristic_length, 6)
            reason = f"入力 Re={target_re:g} から逆算"
        if key == "nu":
            if target_re and spec.inlet_velocity and spec.characteristic_length:
                suggested = round(
                    spec.inlet_velocity * spec.characteristic_length / target_re, 8
                )
                reason = f"入力 Re={target_re:g} から nu を算出"
            elif spec.inlet_velocity and spec.characteristic_length and spec.re_number:
                suggested = round(
                    spec.inlet_velocity * spec.characteristic_length / spec.re_number, 8
                )
        fields.append(RequirementField(
            key=key,
            required=required,
            reason=reason,
            suggested=suggested,
            parser="bool" if key == "steady_state" else ("str" if key in ("solver", "turbulence_model", "phenomenon") else "float"),
        ))

    # Re 整合: 2 自由度のみ調整（nu 固定で U を逆算）
    if target_re and spec.nu and spec.characteristic_length:
        expected_u = target_re * spec.nu / spec.characteristic_length
        if abs(expected_u - spec.inlet_velocity) / max(spec.inlet_velocity, 1e-9) > 0.05:
            if not any(f.key == "inlet_velocity" for f in fields):
                fields.insert(0, RequirementField(
                    key="inlet_velocity",
                    required=True,
                    reason=f"入力 Re={target_re:g} との整合",
                    suggested=round(expected_u, 6),
                ))

    return RequirementProfile(
        phenomenon=phenomenon,
        fields=fields,
        constraints=list(cfg.get("constraints", [])),
        similar_case_ids=list(similar_case_ids or []),
    )


def apply_profile_defaults(spec: SimulationSpec, profile: RequirementProfile) -> SimulationSpec:
    """プロファイルの defaults を spec に適用（policy 層）。"""
    cfg = PHENOMENON_REQUIREMENTS.get(profile.phenomenon, {})
    for key, val in cfg.get("defaults", {}).items():
        if key == "case_type":
            if spec.case_type in ("channel_2d", "general", ""):
                spec.case_type = val
                spec.defaults_applied.append(f"profile_{key}")
        elif hasattr(spec, key):
            current = getattr(spec, key)
            if key in (spec.defaults_applied or []) or current in ("", 0, None, True):
                setattr(spec, key, val)
                if f"profile_{key}" not in spec.defaults_applied:
                    spec.defaults_applied.append(f"profile_{key}")
    if profile.phenomenon == "karman_vortex_shedding":
        spec.solver = "pimpleFoam"
        spec.steady_state = False
        if spec.case_type not in ("cylinder_2d_ogrid",):
            spec.case_type = "cylinder_2d_ogrid"
            spec.mesh_template = "ogrid_cylinder_2d"
    if spec.inlet_velocity and spec.nu and spec.characteristic_length:
        spec.re_number = (spec.inlet_velocity * spec.characteristic_length) / spec.nu
    return spec


def _mentioned_in(text: str, *keywords: str) -> bool:
    hay = text.lower()
    return any(k.lower() in hay for k in keywords)


def _velocity_for_re(spec: SimulationSpec, target_re: float) -> float | None:
    if not spec.nu or not spec.characteristic_length:
        return None
    return round(target_re * spec.nu / spec.characteristic_length, 6)


def review_spec(
    spec: SimulationSpec,
    profile: RequirementProfile,
    description: str = "",
) -> list:
    """Agent②: 完成 spec を物理・ケース知識でレビューし、修正提案を返す。"""
    from ..models import SpecReviewIssue

    issues: list[SpecReviewIssue] = []
    _recalc = spec.inlet_velocity and spec.nu and spec.characteristic_length
    if _recalc:
        spec.re_number = (spec.inlet_velocity * spec.characteristic_length) / spec.nu

    re_val = spec.re_number or 0.0
    user_laminar = _mentioned_in(description, "層流", "laminar")
    user_turbulent = _mentioned_in(description, "乱流", "turbulent", "komega", "ras")

    target_re = _parse_re_from_description(description)

    # Re と乱流モデルの整合
    if re_val > LAMINAR_RE_LIMIT and spec.turbulence_model == "laminar":
        hints = PHENOMENON_REQUIREMENTS.get(profile.phenomenon, {}).get("defaults", {})
        turb_suggested = hints.get("turbulence_model", "kOmegaSST")
        if turb_suggested == "laminar":
            turb_suggested = "kOmegaSST"
        capped_u = _velocity_for_re(spec, LAMINAR_RE_LIMIT)

        if user_laminar and target_re and target_re > LAMINAR_RE_LIMIT:
            issues.append(SpecReviewIssue(
                key="turbulence_model",
                message=(
                    f"Re={target_re:g} 指定と「層流」指定が矛盾しています。"
                    f"乱流モデルへの変更、または Re / 流速 / nu の見直しが必要です"
                ),
                suggested=turb_suggested,
                severity="warning",
                user_locked=True,
            ))
        elif user_laminar and capped_u is not None:
            issues.append(SpecReviewIssue(
                key="inlet_velocity",
                message=(
                    f"Re={re_val:,.0f} ですが「層流」指定のため、"
                    f"Re≈{LAMINAR_RE_LIMIT} になるよう U={capped_u:g} m/s を推奨"
                ),
                suggested=capped_u,
                severity="warning",
            ))
        else:
            alt_msg = ""
            alternatives: list[SpecReviewAlternative] = []
            if capped_u is not None:
                alt_msg = f"（層流のままなら U≈{capped_u:g} m/s で Re≈{LAMINAR_RE_LIMIT}）"
                alternatives.append(SpecReviewAlternative(
                    key="inlet_velocity",
                    suggested=capped_u,
                    label=f"層流維持: U={capped_u:g} m/s",
                ))
            msg = (
                f"Re={re_val:,.0f} は層流限界 (≈{LAMINAR_RE_LIMIT}) を超えています。"
                f"乱流モデル {turb_suggested} を推奨{alt_msg}"
            )
            if user_laminar:
                msg += "（説明文に「層流」とありますが、物理的には乱流域です）"
            issues.append(SpecReviewIssue(
                key="turbulence_model",
                message=msg,
                suggested=turb_suggested,
                severity="warning",
                user_locked=user_laminar,
                alternatives=alternatives,
            ))

    # カルマンは非定常固定
    if profile.phenomenon == "karman_vortex_shedding" and spec.steady_state:
        issues.append(SpecReviewIssue(
            key="steady_state",
            message="カルマン渦放出は非定常解析が必要です",
            suggested=False,
            severity="error",
        ))

    # 現象とソルバー
    if profile.phenomenon == "karman_vortex_shedding" and spec.solver not in (
        "pimpleFoam", "pisoFoam", "icoFoam",
    ):
        issues.append(SpecReviewIssue(
            key="solver",
            message=f"カルマン渦には非定常ソルバー (pimpleFoam) が必要 — 現在 {spec.solver}",
            suggested="pimpleFoam",
            severity="error",
        ))

    if spec.steady_state and spec.solver in ("pimpleFoam", "icoFoam") and not user_turbulent:
        if profile.phenomenon in ("channel_internal", "airfoil_steady", ""):
            issues.append(SpecReviewIssue(
                key="solver",
                message="定常解析には simpleFoam が一般的です",
                suggested="simpleFoam",
                severity="warning",
            ))

    # Re 明示入力との速度整合
    target_re = _parse_re_from_description(description)
    if target_re and spec.nu and spec.characteristic_length:
        expected_u = target_re * spec.nu / spec.characteristic_length
        if abs(expected_u - spec.inlet_velocity) / max(spec.inlet_velocity, 1e-9) > 0.05:
            issues.append(SpecReviewIssue(
                key="inlet_velocity",
                message=f"入力 Re={target_re:g} と U/nu/L が不一致（期待 U≈{expected_u:g} m/s）",
                suggested=round(expected_u, 6),
                severity="error",
            ))

    return issues


def apply_review_fixes(
    spec: SimulationSpec,
    issues: list,
    *,
    respect_user_lock: bool = True,
) -> SimulationSpec:
    """Agent② のレビュー指摘を spec に反映する。"""
    for issue in issues:
        if respect_user_lock and issue.user_locked:
            continue
        if issue.suggested is None or not hasattr(spec, issue.key):
            continue
        setattr(spec, issue.key, issue.suggested)
        tag = f"agent2_review_{issue.key}"
        if tag not in spec.defaults_applied:
            spec.defaults_applied.append(tag)
    if spec.inlet_velocity and spec.nu and spec.characteristic_length:
        spec.re_number = (spec.inlet_velocity * spec.characteristic_length) / spec.nu
    return spec

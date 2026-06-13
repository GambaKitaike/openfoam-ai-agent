"""
SimulationSpec の曖昧・未指定パラメータをユーザーに確認する
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from rich.console import Console
from rich.panel import Panel
from rich.prompt import FloatPrompt, Prompt

from ..models import EnrichedContext, SimulationSpec, SpecReviewIssue
from ..rag.reference_case_params import ReferenceCaseParams

console = Console()

# phenomenon ごとの妥当な代表値（チュートリアル寄り）
PHENOMENON_HINTS: dict[str, dict[str, float | str]] = {
    "airfoil_steady": {
        "inlet_velocity": 5.0,
        "characteristic_length": 0.1,
        "turbulence_model": "SpalartAllmaras",
        "re_target": 500_000,
    },
    "karman_vortex_shedding": {
        "inlet_velocity": 0.15,
        "characteristic_length": 0.1,
        "turbulence_model": "laminar",
        "re_target": 150,
    },
    "backward_facing_step": {
        "inlet_velocity": 10.0,
        "characteristic_length": 0.01,
        "turbulence_model": "kOmegaSST",
        "re_target": 10_000,
    },
    "channel_internal": {
        "inlet_velocity": 2.0,
        "characteristic_length": 0.1,
        "turbulence_model": "kOmegaSST",
        "re_target": 10_000,
    },
    "cavity_flow": {
        "inlet_velocity": 1.0,
        "characteristic_length": 0.1,
        "turbulence_model": "laminar",
        "re_target": 1_000,
    },
}

RE_WARN_STEADY = 100_000
RE_WARN_LAMINAR = 2_300


@dataclass
class ClarificationField:
    key: str
    label: str
    current: float | str
    suggested: float | str
    reason: str
    parser: str = "float"  # float | str
    warn_only: bool = False  # True のとき spec を書き換えず警告のみ


def _mentioned_in(text: str, *keywords: str) -> bool:
    hay = text.lower()
    return any(k.lower() in hay for k in keywords)


def _parse_re_from_description(description: str) -> float | None:
    from ..case_builder.policy import parse_re_from_description
    return parse_re_from_description(description)


def _recalc_re(spec: SimulationSpec) -> None:
    if spec.inlet_velocity and spec.nu and spec.characteristic_length:
        spec.re_number = (spec.inlet_velocity * spec.characteristic_length) / spec.nu
    else:
        spec.re_number = None


def collect_clarifications(
    spec: SimulationSpec,
    description: str,
) -> tuple[list[ClarificationField], list[str]]:
    """確認が必要なフィールド一覧と、書き換え不要の警告を返す。"""
    fields: list[ClarificationField] = []
    warnings: list[str] = []
    defaults = set(spec.defaults_applied or [])
    hints = PHENOMENON_HINTS.get(spec.phenomenon, {})
    target_re = _parse_re_from_description(description)
    from ..case_builder.policy import DEFAULT_RE_VELOCITY, nu_from_re, velocity_mentioned_in

    # Re のみ指定: U=1 m/s 固定 → ν を算出
    if target_re is not None and "inlet_velocity" in defaults and not velocity_mentioned_in(description):
        fields.append(ClarificationField(
            key="inlet_velocity",
            label="流速 (m/s)",
            current=spec.inlet_velocity,
            suggested=DEFAULT_RE_VELOCITY,
            reason=f"Re={target_re:g} 指定 — 流速を {DEFAULT_RE_VELOCITY:g} m/s に固定",
        ))
    elif target_re is not None and velocity_mentioned_in(description) and "nu" in defaults:
        nu_val = nu_from_re(spec.inlet_velocity, spec.characteristic_length, target_re)
        fields.append(ClarificationField(
            key="nu",
            label="動粘度 nu (m²/s)",
            current=spec.nu,
            suggested=round(nu_val, 8),
            reason=f"Re={target_re:g}, U={spec.inlet_velocity:g} m/s から算出",
        ))

    if "inlet_velocity" in defaults and not any(
        f.key == "inlet_velocity" for f in fields
    ):
        suggested = hints.get("inlet_velocity", spec.inlet_velocity)
        fields.append(ClarificationField(
            key="inlet_velocity",
            label="流速 (m/s)",
            current=spec.inlet_velocity,
            suggested=suggested,
            reason="入力に流速が指定されていません",
        ))

    if "characteristic_length" in defaults or (
        spec.phenomenon in PHENOMENON_HINTS
        and not _mentioned_in(description, "弦長", "直径", "長さ", "chord", "diameter", "length", "m")
    ):
        if spec.phenomenon == "airfoil_steady" and "characteristic_length" in defaults:
            fields.append(ClarificationField(
                key="characteristic_length",
                label="代表長さ・弦長 (m)",
                current=spec.characteristic_length,
                suggested=hints.get("characteristic_length", 0.1),
                reason="翼解析の代表長さ（弦長）が未指定です",
            ))
        elif spec.phenomenon == "karman_vortex_shedding" and "characteristic_length" in defaults:
            fields.append(ClarificationField(
                key="characteristic_length",
                label="円柱直径 (m)",
                current=spec.characteristic_length,
                suggested=0.1,
                reason="円柱直径が未指定です",
            ))

    if "turbulence_model" in defaults and not _mentioned_in(
        description, "層流", "乱流", "laminar", "turbulent", "RAS", "LES"
    ):
        suggested = hints.get("turbulence_model", spec.turbulence_model)
        fields.append(ClarificationField(
            key="turbulence_model",
            label="乱流モデル",
            current=spec.turbulence_model,
            suggested=suggested,
            reason="層流/乱流の指定がありません",
            parser="str",
        ))

    if "nu" in defaults and not _mentioned_in(description, "空気", "水", "air", "water", "粘度", "nu"):
        fields.append(ClarificationField(
            key="nu",
            label="動粘度 nu (m²/s)",
            current=spec.nu,
            suggested=spec.nu,
            reason="流体（空気/水）の指定がありません — 空気=1.5e-5, 水=1e-6",
        ))

    # Re が異常に高い / 層流なのに高 Re
    _recalc_re(spec)
    if spec.re_number:
        if spec.turbulence_model == "laminar" and spec.re_number > RE_WARN_LAMINAR:
            if not any(f.key == "inlet_velocity" for f in fields):
                if velocity_mentioned_in(description):
                    warnings.append(
                        f"層流指定ですが Re≈{spec.re_number:,.0f} です。"
                        f"発散する可能性があります（指定流速 {spec.inlet_velocity:g} m/s を維持）"
                    )
                else:
                    capped_v = RE_WARN_LAMINAR * spec.nu / spec.characteristic_length
                    fields.append(ClarificationField(
                        key="inlet_velocity",
                        label="流速 (m/s)",
                        current=spec.inlet_velocity,
                        suggested=round(capped_v, 4),
                        reason=f"層流指定ですが Re={spec.re_number:,.0f} と高すぎます",
                    ))
        elif spec.steady_state and spec.re_number > RE_WARN_STEADY:
            if not any(f.key == "inlet_velocity" for f in fields):
                if velocity_mentioned_in(description):
                    warnings.append(
                        f"定常解析で Re≈{spec.re_number:,.0f} と高めです。"
                        f"収束が困難な可能性があります（指定流速 {spec.inlet_velocity:g} m/s を維持）"
                    )
                else:
                    hint_re = hints.get("re_target", RE_WARN_STEADY)
                    capped_v = float(hint_re) * spec.nu / spec.characteristic_length
                    fields.append(ClarificationField(
                        key="inlet_velocity",
                        label="流速 (m/s)",
                        current=spec.inlet_velocity,
                        suggested=round(capped_v, 4),
                        reason=(
                            f"定常解析で Re={spec.re_number:,.0f} と高めです "
                            f"(目安 Re≈{hint_re:,.0f})"
                        ),
                    ))

    # 重複 key を除去（最初のものを残す）
    seen: set[str] = set()
    unique: list[ClarificationField] = []
    for f in fields:
        if f.key not in seen:
            seen.add(f.key)
            unique.append(f)
    return unique, warnings


def apply_auto_fixes(spec: SimulationSpec, fields: list[ClarificationField]) -> SimulationSpec:
    """非対話モード: 推奨値を自動適用（warn_only はスキップ）。"""
    for f in fields:
        if f.warn_only:
            continue
        if f.parser == "str":
            setattr(spec, f.key, str(f.suggested))
        else:
            setattr(spec, f.key, float(f.suggested))
        if f.key not in spec.defaults_applied:
            spec.defaults_applied.append(f.key)
    _recalc_re(spec)
    return spec


def clarify_spec(
    spec: SimulationSpec,
    description: str,
    interactive: bool = True,
) -> SimulationSpec:
    """
    未指定パラメータを確認し spec を更新する。

    interactive=False のときは推奨値を自動適用（ユーザー明示値は警告のみ）。
    """
    fields, warnings = collect_clarifications(spec, description)
    if not fields and not warnings:
        return spec

    if not interactive:
        if warnings:
            for msg in warnings:
                console.print(f"  [yellow]警告: {msg}[/yellow]")
        if fields:
            console.print(
                f"  [dim]未指定パラメータ {len(fields)} 件 — 推奨値を自動適用 "
                f"(対話する場合は --interactive)[/dim]"
            )
            return apply_auto_fixes(spec, fields)
        return spec

    if warnings:
        for msg in warnings:
            console.print(f"  [yellow]警告: {msg}[/yellow]")
    if not fields:
        return spec

    console.print(Panel(
        "入力に含まれていない（または LLM が補完した）パラメータがあります。\n"
        "Enter のみで [dim]括弧内の推奨値[/dim] を使用します。",
        title="[bold yellow]解析条件の確認[/bold yellow]",
        border_style="yellow",
    ))

    for f in fields:
        if f.parser == "str":
            choices_hint = "laminar / kOmegaSST / SpalartAllmaras 等"
            raw = Prompt.ask(
                f"{f.label} — {f.reason}\n  現在: {f.current}  推奨: [{f.suggested}] ({choices_hint})",
                default=str(f.suggested),
            )
            value: float | str = raw.strip() or str(f.suggested)
            setattr(spec, f.key, value)
        elif f.key == "nu":
            raw = Prompt.ask(
                f"{f.label} — {f.reason}\n  現在: {f.current:g}  推奨: [{f.suggested:g}]",
                default=f"{float(f.suggested):g}",
            )
            if raw.strip().lower() in ("空気", "air"):
                spec.nu = 1.5e-5
            elif raw.strip().lower() in ("水", "water"):
                spec.nu = 1e-6
            else:
                try:
                    spec.nu = float(raw)
                except ValueError:
                    spec.nu = float(f.suggested)
        else:
            raw = FloatPrompt.ask(
                f"{f.label} — {f.reason}\n  現在: {f.current}  推奨",
                default=float(f.suggested),
            )
            setattr(spec, f.key, float(raw))

        if f.key in spec.defaults_applied:
            spec.defaults_applied.remove(f.key)
        spec.defaults_applied.append(f"confirmed_{f.key}")

    _recalc_re(spec)
    return spec


def _apply_profile_field(spec: SimulationSpec, f: ClarificationField) -> None:
    """RequirementProfile の 1 項目を spec に反映する。"""
    if f.suggested is None:
        current = getattr(spec, f.key, None)
        if current not in (None, "", 0):
            return
        raise ValueError(f"推奨値がありません: {f.key}")

    if f.parser == "str":
        setattr(spec, f.key, str(f.suggested))
    elif f.parser == "bool":
        setattr(spec, f.key, bool(f.suggested))
    elif f.key == "nu":
        spec.nu = float(f.suggested)
    else:
        setattr(spec, f.key, float(f.suggested))


def clarify_with_profile(
    spec: SimulationSpec,
    profile,
    description: str = "",
    interactive: bool = True,
) -> SimulationSpec:
    """Agent② RequirementProfile に基づき spec を完成させる。"""
    from ..case_builder.policy import apply_solver_policy, reconcile_re
    from ..rag.requirement_profile import apply_profile_defaults

    target_re = _parse_re_from_description(description)
    apply_profile_defaults(spec, profile)

    if target_re is not None:
        reconcile_re(spec, target_re, description)

    fields = []
    for rf in profile.fields:
        fields.append(ClarificationField(
            key=rf.key,
            label=rf.key,
            current=getattr(spec, rf.key, ""),
            suggested=rf.suggested,
            reason=rf.reason,
            parser=rf.parser,
        ))

    if not fields:
        apply_solver_policy(spec)
        _recalc_re(spec)
        return spec

    if profile.reference_hints:
        hint_lines = []
        for hint in profile.reference_hints[:2]:
            parts = []
            if hint.inlet_velocity is not None:
                parts.append(f"U={hint.inlet_velocity:g}")
            if hint.nu is not None:
                parts.append(f"nu={hint.nu:g}")
            if hint.turbulence_model:
                parts.append(hint.turbulence_model)
            if parts:
                label = hint.title_ja or hint.case_id.split("/")[-1]
                hint_lines.append(f"  {label}: {', '.join(parts)}")
        if hint_lines:
            console.print(
                "[dim]類似チュートリアル典型値:\n"
                + "\n".join(hint_lines)
                + "[/dim]"
            )

    if not interactive:
        console.print(
            f"  [dim]要件プロファイル {len(fields)} 件 — 推奨値を自動適用[/dim]"
        )
        for f in fields:
            _apply_profile_field(spec, f)
            spec.defaults_applied.append(f"profile_{f.key}")
    else:
        console.print(Panel(
            "Agent② のケース知識に基づき、不足パラメータを確認します。",
            title="[bold yellow]解析条件の確認[/bold yellow]",
            border_style="yellow",
        ))
        for f in fields:
            if f.parser == "str":
                raw = Prompt.ask(
                    f"{f.label} — {f.reason}\n  現在: {f.current}  推奨: [{f.suggested}]",
                    default=str(f.suggested),
                )
                setattr(spec, f.key, raw.strip() or str(f.suggested))
            elif f.parser == "bool":
                setattr(spec, f.key, bool(f.suggested))
            elif f.key == "nu":
                raw = Prompt.ask(
                    f"{f.label} — {f.reason}\n  現在: {f.current:g}  推奨: [{float(f.suggested):g}]",
                    default=f"{float(f.suggested):g}",
                )
                try:
                    spec.nu = float(raw)
                except ValueError:
                    spec.nu = float(f.suggested)
            else:
                raw = FloatPrompt.ask(
                    f"{f.label} — {f.reason}\n  現在: {f.current}  推奨",
                    default=float(f.suggested),
                )
                setattr(spec, f.key, float(raw))
            spec.defaults_applied.append(f"confirmed_{f.key}")

    if target_re is not None:
        reconcile_re(spec, target_re, description)

    apply_solver_policy(spec)
    _recalc_re(spec)
    return spec


MAX_AGENT2_HEARING_ROUNDS = 3


def hearing_loop_with_agent2(
    spec: SimulationSpec,
    agent2,
    description: str = "",
    interactive: bool = True,
    trace: "AgentDialogueReport | None" = None,
) -> SimulationSpec:
    """
    Agent① ↔ Agent② 内部ループ: 充足 → レビュー → 修正を繰り返す。
    interactive=True なら Agent② の指摘をユーザーにも確認する。
    """
    from ..rag.requirement_profile import apply_review_fixes, review_spec

    if trace is not None:
        from ..agent_dialogue import profile_snapshot, issues_snapshot, spec_snapshot
        trace.add(
            round_num=0,
            from_agent="Agent①",
            to_agent="Agent②",
            kind="draft_spec",
            summary="draft SimulationSpec",
            spec=spec_snapshot(spec),
        )

    for round_num in range(1, MAX_AGENT2_HEARING_ROUNDS + 1):
        profile = agent2.get_requirement_profile(spec, description)
        if trace is not None:
            trace.add(
                round_num=round_num,
                from_agent="Agent②",
                to_agent="Agent①",
                kind="requirement_profile",
                summary=f"必要項目 {sum(1 for f in profile.fields if f.required)} 件",
                profile=profile_snapshot(profile),
            )
        spec = clarify_with_profile(spec, profile, description, interactive=interactive)
        issues = agent2.review_spec(spec, profile, description)
        if trace is not None:
            trace.add(
                round_num=round_num,
                from_agent="Agent①",
                to_agent="Agent②",
                kind="clarified_spec",
                summary="ヒアリング後の SimulationSpec",
                spec=spec_snapshot(spec),
            )
        if issues and trace is not None:
            trace.add(
                round_num=round_num,
                from_agent="Agent②",
                to_agent="Agent①",
                kind="review_issues",
                summary=f"レビュー指摘 {len(issues)} 件",
                issues=issues_snapshot(issues),
            )
        if not issues:
            if round_num > 1:
                console.print(f"  [green]Agent② レビュー OK（ラウンド {round_num}）[/green]")
            break

        console.print(Panel(
            "\n".join(f"• [{i.severity}] {i.message}" for i in issues),
            title=f"[bold yellow]Agent② レビュー (ラウンド {round_num})[/bold yellow]",
            border_style="yellow",
        ))

        if interactive:
            for issue in issues:
                if issue.alternatives and not issue.user_locked:
                    opts = f"1) {issue.key} → {issue.suggested}"
                    for idx, alt in enumerate(issue.alternatives, start=2):
                        opts += f"  {idx}) {alt.label}"
                    choice = Prompt.ask(
                        f"{issue.message}\n  {opts}\n  番号を選択",
                        default="1",
                    ).strip()
                    if choice == "1" and issue.suggested is not None:
                        setattr(spec, issue.key, issue.suggested)
                        spec.defaults_applied.append(f"agent2_review_{issue.key}")
                    else:
                        try:
                            alt_idx = int(choice) - 2
                            alt = issue.alternatives[alt_idx]
                            setattr(spec, alt.key, alt.suggested)
                            spec.defaults_applied.append(f"agent2_review_{alt.key}")
                        except (ValueError, IndexError):
                            if issue.suggested is not None:
                                setattr(spec, issue.key, issue.suggested)
                                spec.defaults_applied.append(f"agent2_review_{issue.key}")
                elif issue.user_locked:
                    ans = Prompt.ask(
                        f"{issue.message}\n  Agent② 推奨: [{issue.suggested}] — この設定のまま続行しますか？ (y/N)",
                        default="n",
                    )
                    if ans.strip().lower() not in ("y", "yes", "はい"):
                        if issue.suggested is not None and hasattr(spec, issue.key):
                            setattr(spec, issue.key, issue.suggested)
                            spec.defaults_applied.append(f"agent2_review_{issue.key}")
                elif issue.severity == "error" or issue.suggested is not None:
                    if issue.suggested is not None and hasattr(spec, issue.key):
                        raw = Prompt.ask(
                            f"{issue.message}\n  推奨 [{issue.suggested}] を採用しますか？ (Y/n)",
                            default="y",
                        )
                        if raw.strip().lower() not in ("n", "no", "いいえ"):
                            setattr(spec, issue.key, issue.suggested)
                            spec.defaults_applied.append(f"agent2_review_{issue.key}")
        else:
            auto = [i for i in issues if not i.user_locked]
            locked = [i for i in issues if i.user_locked]
            from ..case_builder.policy import velocity_mentioned_in
            skipped: list = []
            if auto:
                applicable = []
                for issue in auto:
                    if (
                        issue.key == "inlet_velocity"
                        and velocity_mentioned_in(description)
                    ):
                        skipped.append(issue)
                        continue
                    applicable.append(issue)
                if applicable:
                    apply_review_fixes(spec, applicable, respect_user_lock=True)
                    if trace is not None:
                        trace.add(
                            round_num=round_num,
                            from_agent="Agent①",
                            to_agent="Agent②",
                            kind="review_fixes",
                            summary=f"自動修正 {len(applicable)} 件",
                            spec=spec_snapshot(spec),
                        )
                    console.print(
                        f"  [dim]Agent② が {len(applicable)} 件を自動修正"
                        f"{f'（{len(locked)} 件は説明文の指定を維持）' if locked else ''}"
                        f"{f'（{len(skipped)} 件は指定流速を維持）' if skipped else ''}[/dim]"
                    )
                elif locked or skipped:
                    parts = []
                    if locked:
                        parts.append(f"{len(locked)} 件は説明文の指定を維持")
                    if skipped:
                        parts.append(f"{len(skipped)} 件は指定流速を維持")
                    console.print(
                        f"  [dim]Agent② が {len(issues)} 件の矛盾を検出 — "
                        f"{'、'.join(parts)}[/dim]"
                    )
                    break
            elif locked:
                console.print(
                    f"  [dim]Agent② が {len(locked)} 件の矛盾を検出 — "
                    f"説明文の指定を維持[/dim]"
                )
                break

        from ..case_builder.policy import apply_solver_policy
        apply_solver_policy(spec)
        _recalc_re(spec)

    return spec


def collect_reference_clarifications(
    spec: SimulationSpec,
    ref: ReferenceCaseParams,
) -> list[ClarificationField]:
    """参照ケース典型値と spec の差分から確認項目を生成。"""
    fields: list[ClarificationField] = []
    title = ref.title_ja or ref.case_id

    if ref.inlet_velocity is not None:
        if abs(ref.inlet_velocity - spec.inlet_velocity) / max(spec.inlet_velocity, 1e-9) > 0.05:
            fields.append(ClarificationField(
                key="inlet_velocity",
                label="流速 (m/s)",
                current=spec.inlet_velocity,
                suggested=round(ref.inlet_velocity, 4),
                reason=f"参照ケース [{title}] の典型値 ({ref.velocity_note or 'U場'})",
            ))

    if ref.nu is not None and abs(ref.nu - spec.nu) / spec.nu > 0.05:
        fields.append(ClarificationField(
            key="nu",
            label="動粘度 nu (m²/s)",
            current=spec.nu,
            suggested=ref.nu,
            reason=f"参照ケース [{title}] の transportProperties",
        ))

    if (
        ref.turbulence_model
        and ref.turbulence_model not in ("unknown", "RAS")
        and ref.turbulence_model != spec.turbulence_model
    ):
        fields.append(ClarificationField(
            key="turbulence_model",
            label="乱流モデル",
            current=spec.turbulence_model,
            suggested=ref.turbulence_model,
            reason=f"参照ケース [{title}] の turbulenceProperties",
            parser="str",
        ))

    if ref.characteristic_length is not None:
        if abs(ref.characteristic_length - spec.characteristic_length) / max(
            spec.characteristic_length, 1e-9
        ) > 0.05:
            fields.append(ClarificationField(
                key="characteristic_length",
                label="代表長さ (m)",
                current=spec.characteristic_length,
                suggested=round(ref.characteristic_length, 6),
                reason=f"参照ケース [{title}] の Re 要約から逆算",
            ))

    return fields


def _apply_clarification_fields(spec: SimulationSpec, fields: list[ClarificationField]) -> None:
    for f in fields:
        if f.parser == "str":
            setattr(spec, f.key, str(f.suggested))
        else:
            setattr(spec, f.key, float(f.suggested))
        spec.defaults_applied.append(f"from_reference_{f.key}")
    _recalc_re(spec)


def _reference_clarification_warnings(fields: list[ClarificationField]) -> list[str]:
    """参照ケースとの差分を警告文に変換（spec は書き換えない）。"""
    warnings: list[str] = []
    for f in fields:
        warnings.append(
            f"{f.label}: {f.reason} — 現値 {f.current}、参照典型値 {f.suggested}"
        )
    return warnings


def clarify_from_reference(
    spec: SimulationSpec,
    context: EnrichedContext,
    interactive: bool = True,
) -> tuple[SimulationSpec, list[str]]:
    """
    Phase A: RAG 選定後、参照ケースの典型条件をユーザーに提案する。
    """
    if not context.reference_case_id or context.reference_typical_params is None:
        return spec, []

    # 現象タグが一致しない参照ケース（general 等）の典型値は上書きしない
    ref_phenomenon = context.reference_phenomenon or "general"
    if (
        spec.phenomenon
        and spec.phenomenon != "general"
        and ref_phenomenon != spec.phenomenon
    ):
        console.print(
            f"  [dim]参照ケースの現象 ({ref_phenomenon}) が "
            f"{spec.phenomenon} と異なるため典型条件の自動適用をスキップ[/dim]"
        )
        return spec, []

    ref = context.reference_typical_params
    fields = collect_reference_clarifications(spec, ref)
    if not fields:
        return spec, []

    ref_lines = ref.to_display_lines()
    body = "\n".join(ref_lines) if ref_lines else ref.summary_ja[:300]
    if ref.summary_ja and body != ref.summary_ja[:300]:
        body += f"\n\n{ref.summary_ja[:200]}"

    if not interactive:
        warnings = _reference_clarification_warnings(fields)
        console.print(
            f"  [dim]参照ケース {ref.case_id} の典型条件 {len(fields)} 件 — "
            f"自動適用せず警告のみ[/dim]"
        )
        return spec, warnings

    console.print(Panel(
        body or "(典型条件を抽出できませんでした)",
        title=f"[bold cyan]参照ケースの典型条件[/bold cyan] — {ref.title_ja or ref.case_id}",
        border_style="cyan",
    ))

    use_ref = Prompt.ask(
        "参照ケースの条件に合わせますか？ (個別項目も確認できます)",
        choices=["はい", "いいえ", "個別"],
        default="はい",
    )

    if use_ref == "いいえ":
        return spec, []

    if use_ref == "はい":
        _apply_clarification_fields(spec, fields)
        console.print("  [green]参照ケースの典型条件を適用しました[/green]")
        return spec, []

    for f in fields:
        if f.parser == "str":
            raw = Prompt.ask(
                f"{f.label} — {f.reason}\n  現在: {f.current}  参照: [{f.suggested}]",
                default=str(f.suggested),
            )
            setattr(spec, f.key, raw.strip() or str(f.suggested))
        elif f.key == "nu":
            raw = Prompt.ask(
                f"{f.label} — {f.reason}\n  現在: {f.current:g}  参照: [{f.suggested:g}]",
                default=f"{float(f.suggested):g}",
            )
            try:
                spec.nu = float(raw)
            except ValueError:
                spec.nu = float(f.suggested)
        else:
            raw = FloatPrompt.ask(
                f"{f.label} — {f.reason}\n  現在: {f.current}  参照",
                default=float(f.suggested),
            )
            setattr(spec, f.key, float(raw))
        spec.defaults_applied.append(f"from_reference_{f.key}")

    _recalc_re(spec)
    return spec, []

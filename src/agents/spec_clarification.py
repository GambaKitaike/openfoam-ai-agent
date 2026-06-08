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

from ..models import EnrichedContext, SimulationSpec
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


def _mentioned_in(text: str, *keywords: str) -> bool:
    hay = text.lower()
    return any(k.lower() in hay for k in keywords)


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


def _recalc_re(spec: SimulationSpec) -> None:
    if spec.inlet_velocity and spec.nu and spec.characteristic_length:
        spec.re_number = (spec.inlet_velocity * spec.characteristic_length) / spec.nu
    else:
        spec.re_number = None


def collect_clarifications(
    spec: SimulationSpec,
    description: str,
) -> list[ClarificationField]:
    """確認が必要なフィールド一覧を返す。"""
    fields: list[ClarificationField] = []
    defaults = set(spec.defaults_applied or [])
    hints = PHENOMENON_HINTS.get(spec.phenomenon, {})
    target_re = _parse_re_from_description(description)

    # 入力文に Re が書いてあれば速度を逆算できる（代表長さ・nu が分かる場合）
    if target_re is not None and "inlet_velocity" in defaults:
        vel = target_re * spec.nu / spec.characteristic_length
        fields.append(ClarificationField(
            key="inlet_velocity",
            label="流速 (m/s)",
            current=spec.inlet_velocity,
            suggested=round(vel, 4),
            reason=f"入力の Re={target_re:g} から逆算",
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
    return unique


def apply_auto_fixes(spec: SimulationSpec, fields: list[ClarificationField]) -> SimulationSpec:
    """非対話モード: 推奨値を自動適用。"""
    for f in fields:
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

    interactive=False のときは推奨値を自動適用。
    """
    fields = collect_clarifications(spec, description)
    if not fields:
        return spec

    if not interactive:
        console.print(
            f"  [dim]未指定パラメータ {len(fields)} 件 — 推奨値を自動適用 "
            f"(対話する場合は --interactive)[/dim]"
        )
        return apply_auto_fixes(spec, fields)

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


def clarify_from_reference(
    spec: SimulationSpec,
    context: EnrichedContext,
    interactive: bool = True,
) -> SimulationSpec:
    """
    Phase A: RAG 選定後、参照ケースの典型条件をユーザーに提案する。
    """
    if not context.reference_case_id or context.reference_typical_params is None:
        return spec

    ref = context.reference_typical_params
    fields = collect_reference_clarifications(spec, ref)
    if not fields:
        return spec

    ref_lines = ref.to_display_lines()
    body = "\n".join(ref_lines) if ref_lines else ref.summary_ja[:300]
    if ref.summary_ja and body != ref.summary_ja[:300]:
        body += f"\n\n{ref.summary_ja[:200]}"

    if not interactive:
        console.print(
            f"  [dim]参照ケース {ref.case_id} の典型条件 {len(fields)} 件を自動適用[/dim]"
        )
        _apply_clarification_fields(spec, fields)
        return spec

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
        return spec

    if use_ref == "はい":
        _apply_clarification_fields(spec, fields)
        console.print("  [green]参照ケースの典型条件を適用しました[/green]")
        return spec

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
    return spec

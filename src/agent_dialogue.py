"""Agent 間通信のトレースとテスト用レポート。"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .models import RequirementProfile, ReferenceMatch, SimulationSpec, SpecReviewIssue

console = Console()


@dataclass
class AgentMessage:
    """1 回の Agent 間メッセージ。"""
    round_num: int
    from_agent: str
    to_agent: str
    kind: str
    summary: str
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentDialogueReport:
    description: str
    messages: list[AgentMessage] = field(default_factory=list)
    final_spec: SimulationSpec | None = None
    reference_match: ReferenceMatch | None = None

    def add(
        self,
        *,
        round_num: int,
        from_agent: str,
        to_agent: str,
        kind: str,
        summary: str,
        **detail: Any,
    ) -> None:
        self.messages.append(AgentMessage(
            round_num=round_num,
            from_agent=from_agent,
            to_agent=to_agent,
            kind=kind,
            summary=summary,
            detail=detail,
        ))


def spec_snapshot(spec: SimulationSpec) -> dict[str, Any]:
    return {
        "solver": spec.solver,
        "case_type": spec.case_type,
        "phenomenon": spec.phenomenon,
        "steady_state": spec.steady_state,
        "turbulence_model": spec.turbulence_model,
        "inlet_velocity": spec.inlet_velocity,
        "nu": spec.nu,
        "re_number": spec.re_number,
        "defaults_applied": list(spec.defaults_applied),
    }


def profile_snapshot(profile: RequirementProfile) -> dict[str, Any]:
    return {
        "phenomenon": profile.phenomenon,
        "required_fields": [f.key for f in profile.fields if f.required],
        "suggested_fields": {
            f.key: f.suggested for f in profile.fields if f.suggested is not None
        },
        "constraints": list(profile.constraints),
    }


def issues_snapshot(issues: list[SpecReviewIssue]) -> list[dict[str, Any]]:
    return [
        {
            "key": i.key,
            "severity": i.severity,
            "message": i.message,
            "suggested": i.suggested,
            "user_locked": i.user_locked,
            "alternatives": [
                {"key": a.key, "suggested": a.suggested, "label": a.label}
                for a in i.alternatives
            ],
        }
        for i in issues
    ]


def print_dialogue_report(report: AgentDialogueReport) -> None:
    console.print(Panel.fit(
        "[bold cyan]Agent 間通信テスト[/bold cyan]\n"
        f"入力: {report.description}",
        border_style="cyan",
    ))

    table = Table(title="通信ログ", show_lines=True)
    table.add_column("R", style="dim", width=3)
    table.add_column("From", style="cyan", width=8)
    table.add_column("To", style="green", width=8)
    table.add_column("Kind", style="yellow", width=14)
    table.add_column("Summary")

    for msg in report.messages:
        table.add_row(
            str(msg.round_num),
            msg.from_agent,
            msg.to_agent,
            msg.kind,
            msg.summary,
        )
    console.print(table)

    if report.final_spec:
        snap = spec_snapshot(report.final_spec)
        console.print(Panel(
            "\n".join(f"  {k}: {v}" for k, v in snap.items()),
            title="[bold]最終 Spec（Agent② → Agent③）[/bold]",
            border_style="blue",
        ))

    if report.reference_match:
        ctx = report.reference_match.context
        route = "fast path" if report.reference_match.use_fast_path else "staged"
        console.print(Panel(
            f"  reference_case_id: {ctx.reference_case_id or '(なし)'}\n"
            f"  match_score: {report.reference_match.score:.2f}\n"
            f"  route: {route}\n"
            f"  mesh_template: {ctx.mesh_template_name or ctx.spec.mesh_template}",
            title="[bold]Agent② → Agent③ ReferenceMatch[/bold]",
            border_style="magenta",
        ))

    rounds_with_review = {
        m.round_num for m in report.messages if m.kind == "review_issues"
    }
    has_profile = any(m.kind == "requirement_profile" for m in report.messages)
    has_match = report.reference_match is not None

    checks = [
        ("Agent① → Agent② (draft spec)", any(m.kind == "draft_spec" for m in report.messages)),
        ("Agent② → Agent① (RequirementProfile)", has_profile),
        (
            "Agent② → Agent① (review_spec)",
            any(m.kind == "review_issues" for m in report.messages)
            or (report.final_spec is not None),
        ),
        ("Agent② → Agent③ (ReferenceMatch)", has_match),
    ]
    ok = all(passed for _, passed in checks)

    summary_table = Table(title="接続チェック", show_header=False)
    for label, passed in checks:
        mark = "[green]✓[/green]" if passed else "[red]✗[/red]"
        summary_table.add_row(mark, label)
    console.print(summary_table)

    if ok:
        console.print("\n[bold green]Agent① ↔ Agent② の内部ループは動作しています。[/bold green]")
    else:
        console.print("\n[bold red]一部の Agent 間通信が記録されていません。[/bold red]")

    review_msgs = [m for m in report.messages if m.kind == "review_issues"]
    if not review_msgs and report.final_spec:
        console.print(
            "[dim]注: この入力では Agent② レビュー指摘なし（spec が既に整合）。[/dim]"
        )

    if not report.reference_match or not report.reference_match.context.reference_case_id:
        console.print(
            "[dim]注: RAG 参照ケース未ヒットはインデックス/条件次第で正常。"
            " Agent③ への staged フォールバックは引き続き可能。[/dim]"
        )

    console.print(
        "[dim]未実装: Agent③ → Agent② のファイル単位 syntax 問い合わせ (get_file_guidance)[/dim]"
    )


def report_to_dict(report: AgentDialogueReport) -> dict[str, Any]:
    return {
        "description": report.description,
        "messages": [asdict(m) for m in report.messages],
        "final_spec": spec_snapshot(report.final_spec) if report.final_spec else None,
        "reference_match": {
            "score": report.reference_match.score,
            "use_fast_path": report.reference_match.use_fast_path,
            "reference_case_id": report.reference_match.context.reference_case_id,
        } if report.reference_match else None,
    }


BUILTIN_SCENARIOS: dict[str, str] = {
    "karman": "2D円柱周りのカルマン渦 Re=100 層流 流入速度1m/s",
    "channel_conflict": (
        "直方体の部屋に障害物なし、左から右へ風が1m/sで吹く2D定常解析 simpleFoam"
    ),
    "channel_laminar": (
        "直方体の部屋に障害物なし、左から右へ風が1m/sで吹く2D定常解析 simpleFoam 層流"
    ),
}


def offline_draft_spec(scenario: str, description: str) -> SimulationSpec:
    """LLM なしテスト用: シナリオ別 draft spec（Agent① extract の代わり）。"""
    drafts = {
        "karman": SimulationSpec(
            solver="pimpleFoam",
            case_type="cylinder_2d_ogrid",
            mesh_template="ogrid_cylinder_2d",
            turbulence_model="laminar",
            steady_state=False,
            inlet_velocity=1.0,
            dimensions=2,
            characteristic_length=1.0,
            nu=0.01,
            re_number=100.0,
            phenomenon="karman_vortex_shedding",
            description=description,
        ),
        "channel_conflict": SimulationSpec(
            solver="simpleFoam",
            case_type="channel_2d",
            mesh_template="box_channel_2d",
            turbulence_model="laminar",
            steady_state=True,
            inlet_velocity=1.0,
            dimensions=2,
            characteristic_length=1.0,
            nu=1.5e-5,
            phenomenon="channel_internal",
            description=description,
        ),
        "channel_laminar": SimulationSpec(
            solver="simpleFoam",
            case_type="channel_2d",
            mesh_template="box_channel_2d",
            turbulence_model="laminar",
            steady_state=True,
            inlet_velocity=1.0,
            dimensions=2,
            characteristic_length=1.0,
            nu=1.5e-5,
            phenomenon="channel_internal",
            description=description,
        ),
    }
    if scenario in drafts:
        return drafts[scenario]
    return drafts["channel_conflict"]

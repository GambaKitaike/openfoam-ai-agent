"""
オーケストレーター
4 エージェントを順番に呼び出してパイプラインを統括する
"""
from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

from .case_validator import CaseValidator
from .config import Settings
from .models import AnalysisReport
from .agents.preprocessing import PreprocessingAgent
from .agents.prompt_generation import PromptGenerationAgent
from .agents.openfoam_gpt import OpenFOAMGPTAgent
from .agents.postprocessing import PostprocessingAgent
from .agents.spec_clarification import clarify_from_reference
from .agent_dialogue import (
    AgentDialogueReport,
    offline_draft_spec,
    print_dialogue_report,
    spec_snapshot,
)

console = Console()

MAX_CASE_RETRIES = 3


class OpenFOAMOrchestrator:
    """
    4 エージェントを統括するオーケストレーター。

    パイプライン:
      Agent②(要件) → Agent①(ヒアリング) → Agent②(参照) → Agent③(生成・実行) → Agent④
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.agent1 = PreprocessingAgent(settings)
        self.agent2 = PromptGenerationAgent(settings)
        self.agent3 = OpenFOAMGPTAgent(settings)
        self.agent4 = PostprocessingAgent(settings)
        self.validator = CaseValidator()

    def run(
        self,
        description: str,
        output_dir: str = "./output",
        convergence_threshold: float = 1e-4,
        stl_path: str = "",
        interactive: bool = True,
        parallel: bool = False,
        n_procs: int = 4,
        demo: bool = False,
        periods: int | None = None,
    ) -> AnalysisReport:
        """
        自然言語入力からParaView案内まで全工程を実行する。

        Args:
            description: ユーザーの自然言語入力
            output_dir: ケース出力先ディレクトリ
            convergence_threshold: 収束判定閾値
            stl_path: STLファイルパス（指定時はsnappyHexMeshを使用）

        Returns:
            AnalysisReport
        """
        console.print(Panel.fit(
            "[bold cyan]OpenFOAM AI Agent[/bold cyan]  4-Agent Pipeline\n"
            "自然言語 → メッシュ生成 → 計算 → 物理妥当性チェック",
            border_style="cyan",
        ))
        console.print(f"\n[bold]入力:[/bold] {description}")
        if stl_path:
            console.print(f"[bold]STL:[/bold] {stl_path}")
        console.print(f"[bold]出力先:[/bold] {output_dir}")
        if parallel:
            console.print(f"[bold]並列:[/bold] mpirun -np {n_procs}")
        if demo:
            console.print("[bold]デモ:[/bold] 短時間設定 (GIF/プレビュー向け)")
        if periods is not None:
            console.print(f"[bold]周期数:[/bold] {periods} (カルマン渦 endTime)")
        console.print()

        # ── Agent①: draft 抽出 ────────────────────────────────────────
        console.print(Rule("[bold]Agent①  Pre-processing (extract)[/bold]"))
        spec = self.agent1.extract(description, stl_path=stl_path)

        # ── Agent② + Agent①: 内部ループ（充足 → レビュー）────────────────
        console.print(Rule("[bold]Agent① ↔ Agent②  Spec Completion[/bold]"))
        spec = self.agent1.complete_hearing(
            spec, self.agent2, description, interactive=interactive
        )
        if periods is not None:
            spec.mesh_params["karman_periods"] = periods
        elif demo:
            spec.mesh_params["demo_mode"] = True

        guidance_fn = lambda rel, s, ctx, patch_names=None: self.agent2.get_file_guidance(
            rel, s, ctx, patch_names=patch_names
        )

        # ── Agent② + Agent③（ケース再選定ループ）────────────────────
        exclude_case_ids: list[str] = []
        match = None
        artifacts = None

        for attempt in range(1, MAX_CASE_RETRIES + 1):
            console.print(Rule(
                f"[bold]Agent②  Reference Match (試行 {attempt}/{MAX_CASE_RETRIES})[/bold]"
            ))
            match = self.agent2.retrieve_match(spec, exclude_case_ids=exclude_case_ids)

            if match.context.reference_case_id:
                spec, _ = clarify_from_reference(spec, match.context, interactive=interactive)
                match.context.spec = spec

            console.print(Rule("[bold]Agent③  OpenFOAMGPT[/bold]"))
            artifacts = self.agent3.run(
                context=match.context,
                output_dir=output_dir,
                convergence_threshold=convergence_threshold,
                reference_match=match,
                parallel=parallel,
                n_procs=n_procs,
                file_guidance_fn=guidance_fn,
            )

            if artifacts.solver_success:
                break

            if match.context.reference_case_id:
                console.print(
                    f"[yellow]  参照ケース {match.context.reference_case_id} で失敗。"
                    f"別ケースを試します...[/yellow]"
                )
                exclude_case_ids.append(match.context.reference_case_id)
            else:
                break

        # ── Agent④ ───────────────────────────────────────────────────
        console.print(Rule("[bold]Agent④  Post-processing[/bold]"))
        report = self.agent4.run(artifacts)

        return report

    def test_agent_dialogue(
        self,
        description: str,
        *,
        interactive: bool = False,
        include_reference_match: bool = True,
        offline: bool = False,
        scenario: str = "",
    ) -> AgentDialogueReport:
        """
        Agent① ↔ Agent②（＋任意で Agent② → Agent③）の通信を記録して返す。
        ソルバーは実行しない。
        """
        trace = AgentDialogueReport(description=description)

        if offline:
            console.print(Rule("[bold]Agent①  draft spec (offline)[/bold]"))
            spec = offline_draft_spec(scenario or "channel_conflict", description)
        else:
            console.print(Rule("[bold]Agent①  Pre-processing (extract)[/bold]"))
            spec = self.agent1.extract(description)

        console.print(Rule("[bold]Agent① ↔ Agent②  Spec Completion[/bold]"))
        spec = self.agent1.complete_hearing(
            spec, self.agent2, description, interactive=interactive, trace=trace
        )
        trace.final_spec = spec

        if include_reference_match:
            console.print(Rule("[bold]Agent②  Reference Match[/bold]"))
            match = self.agent2.retrieve_match(spec)
            trace.reference_match = match
            trace.add(
                round_num=0,
                from_agent="Agent②",
                to_agent="Agent③",
                kind="reference_match",
                summary=(
                    f"score={match.score:.2f}, "
                    f"route={'fast' if match.use_fast_path else 'staged'}, "
                    f"ref={match.context.reference_case_id or 'none'}"
                ),
                reference_case_id=match.context.reference_case_id,
                score=match.score,
                use_fast_path=match.use_fast_path,
            )

        print_dialogue_report(trace)
        return trace

"""Agent②: Knowledge & Reference エージェント。"""
from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from ..config import Settings
from ..models import RequirementProfile, ReferenceMatch, SimulationSpec, EnrichedContext, SpecReviewIssue
from ..rag.retriever import OpenFOAMRetriever

console = Console()


class PromptGenerationAgent:
    """Agent②: 要件プロファイル + 参照ケース選定。"""

    def __init__(self, settings: Settings):
        self.settings = settings
        db_path = Path(__file__).parent.parent.parent / "knowledge_base" / "chroma_db"
        self.retriever = OpenFOAMRetriever(
            db_path=str(db_path),
            openai_api_key=settings.openai_api_key,
        )

    def get_requirement_profile(
        self,
        spec: SimulationSpec,
        description: str = "",
    ) -> RequirementProfile:
        """Agent① ヒアリング用: 必要十分条件を返す。"""
        return self.retriever.get_requirement_profile(spec, description)

    def review_spec(
        self,
        spec: SimulationSpec,
        profile: RequirementProfile,
        description: str = "",
    ) -> list[SpecReviewIssue]:
        """Agent① 向け: 完成 spec の物理整合性レビュー。"""
        from ..rag.requirement_profile import review_spec as _review
        return _review(spec, profile, description)

    def retrieve_match(
        self,
        spec: SimulationSpec,
        exclude_case_ids: list[str] | None = None,
    ) -> ReferenceMatch:
        """Agent③ 用: 参照ケース + match_score。"""
        match = self.retriever.retrieve_match(spec, exclude_case_ids=exclude_case_ids)
        self._print_rag_summary(match.context, match.score, match.use_fast_path)
        return match

    def run(
        self,
        spec: SimulationSpec,
        exclude_case_ids: list[str] | None = None,
    ) -> EnrichedContext:
        """後方互換: EnrichedContext のみ返す。"""
        return self.retrieve_match(spec, exclude_case_ids).context

    def _print_rag_summary(
        self,
        context: EnrichedContext,
        score: float = 0.0,
        use_fast_path: bool = False,
    ):
        if not context.rag_available and not context.reference_case_id:
            if context.mesh_template_name:
                console.print(Panel(
                    f"フォールバック: [bold]{context.mesh_template_name}[/bold]",
                    title="[yellow]参照ケース未選択[/yellow]",
                    border_style="yellow",
                ))
            return

        if context.reference_case_id:
            title = context.reference_title_ja or context.reference_case_id
            path_label = "fast path" if use_fast_path else "staged"
            body = (
                f"[bold]{title}[/bold]\n"
                f"ケースID: {context.reference_case_id}\n"
                f"一致度: {score:.2f} → [bold]{path_label}[/bold]"
            )
            console.print(Panel(
                body,
                title="[bold green]ケース単位 RAG[/bold green]",
                border_style="green",
            ))
        elif context.mesh_template_name:
            console.print(Panel(
                f"フォールバック: [bold]{context.mesh_template_name}[/bold]",
                title="[yellow]参照ケース未選択[/yellow]",
                border_style="yellow",
            ))

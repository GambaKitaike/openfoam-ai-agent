"""
Agent② Prompt Generation Agent (RAG)
ChromaDB から参照チュートリアルケースを選定し EnrichedContext を生成する
"""
from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from ..config import Settings
from ..models import SimulationSpec, EnrichedContext
from ..rag.retriever import OpenFOAMRetriever

console = Console()


class PromptGenerationAgent:
    """Agent②: SimulationSpec + ケース単位 RAG → EnrichedContext 生成エージェント。"""

    def __init__(self, settings: Settings):
        self.settings = settings
        db_path = Path(__file__).parent.parent.parent / "knowledge_base" / "chroma_db"
        self.retriever = OpenFOAMRetriever(
            db_path=str(db_path),
            openai_api_key=settings.openai_api_key,
        )

    def run(
        self,
        spec: SimulationSpec,
        exclude_case_ids: list[str] | None = None,
    ) -> EnrichedContext:
        """
        SimulationSpec に基づいて参照ケースを選定し EnrichedContext を返す。

        Args:
            spec: Agent① が生成した解析仕様
            exclude_case_ids: 再選定時に除外する case_id リスト

        Returns:
            EnrichedContext
        """
        if self.retriever.is_available:
            count = self.retriever.collection.count()
            console.print(f"  RAGケースカタログ: [green]{count:,} ケース[/green] から検索中...")
            context = self.retriever.retrieve(spec, exclude_case_ids=exclude_case_ids)
            self._print_rag_summary(context)
        else:
            console.print("  [yellow]RAGインデックス未構築 - テンプレートベースで続行[/yellow]")
            console.print("  [dim]（python -m src.main build-index で構築できます）[/dim]")
            context = self.retriever._fallback_context(spec)

        return context

    def _print_rag_summary(self, context: EnrichedContext):
        """RAG 検索結果のサマリーを表示する。"""
        if not context.rag_available:
            return

        if context.reference_case_id:
            n_files = len(context.reference_files)
            title = context.reference_title_ja or context.reference_case_id
            summary = context.reference_summary_ja
            body = f"[bold]{title}[/bold]\n"
            if summary:
                body += f"{summary}\n\n"
            body += (
                f"ケースID: {context.reference_case_id}\n"
                f"ファイル数: {n_files}\n"
                f"パス: {context.reference_case_path}"
            )
            if context.reference_phenomenon:
                body += f"\n現象タグ: {context.reference_phenomenon}"
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

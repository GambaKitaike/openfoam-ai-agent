"""
Agent② Prompt Generation Agent (RAG)
ChromaDB からチュートリアル・ドキュメントを検索して
EnrichedContext（RAG 拡張コンテキスト）を生成する
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
    """Agent②: SimulationSpec + RAG → EnrichedContext 生成エージェント。"""

    def __init__(self, settings: Settings):
        self.settings = settings
        db_path = Path(__file__).parent.parent.parent / "knowledge_base" / "chroma_db"
        self.retriever = OpenFOAMRetriever(
            db_path=str(db_path),
            openai_api_key=settings.openai_api_key,
        )

    def run(self, spec: SimulationSpec) -> EnrichedContext:
        """
        SimulationSpec に基づいて RAG 検索を実行し EnrichedContext を返す。

        Args:
            spec: Agent① が生成した解析仕様

        Returns:
            EnrichedContext
        """
        if self.retriever.is_available:
            count = self.retriever.collection.count()
            console.print(f"  RAGインデックス: [green]{count:,} チャンク[/green] から検索中...")
            context = self.retriever.retrieve(spec)
            self._print_rag_summary(context)
        else:
            console.print("  [yellow]RAGインデックス未構築 - テンプレートベースで続行[/yellow]")
            console.print("  [dim]（python -m src.main build-index で構築できます）[/dim]")
            context = self.retriever._fallback_context(spec)

        return context

    def _print_rag_summary(self, context: EnrichedContext):
        """RAG 検索結果のサマリーを表示する。"""
        if not context.rag_available or not context.rag_sources:
            return

        sources_display = []
        seen = set()
        for src in context.rag_sources[:5]:
            # ソースパスを短く表示
            short = Path(src).name if "/" in src or "\\" in src else src
            if short not in seen:
                sources_display.append(short)
                seen.add(short)

        console.print(Panel(
            f"メッシュテンプレート: [bold]{context.mesh_template_name}[/bold]\n"
            f"参照ソース: {', '.join(sources_display)}",
            title="[bold green]RAG 検索結果[/bold green]",
            border_style="green",
        ))

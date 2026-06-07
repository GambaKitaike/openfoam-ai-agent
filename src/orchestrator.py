"""
オーケストレーター
4 エージェントを順番に呼び出してパイプラインを統括する
"""
from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

from .config import Settings
from .models import AnalysisReport
from .agents.preprocessing import PreprocessingAgent
from .agents.prompt_generation import PromptGenerationAgent
from .agents.openfoam_gpt import OpenFOAMGPTAgent
from .agents.postprocessing import PostprocessingAgent

console = Console()


class OpenFOAMOrchestrator:
    """
    4 エージェントを統括するオーケストレーター。

    パイプライン:
      Agent① Pre-processing   : 自然言語 → SimulationSpec
      Agent② Prompt Generation: SimulationSpec + RAG → EnrichedContext
      Agent③ OpenFOAMGPT      : EnrichedContext → 生成・実行・自己修正 → CaseArtifacts
      Agent④ Post-processing  : CaseArtifacts → 妥当性チェック → AnalysisReport
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.agent1 = PreprocessingAgent(settings)
        self.agent2 = PromptGenerationAgent(settings)
        self.agent3 = OpenFOAMGPTAgent(settings)
        self.agent4 = PostprocessingAgent(settings)

    def run(
        self,
        description: str,
        output_dir: str = "./output",
        convergence_threshold: float = 1e-4,
        stl_path: str = "",
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
        console.print(f"[bold]出力先:[/bold] {output_dir}\n")

        # ── Agent① ───────────────────────────────────────────────────
        console.print(Rule("[bold]Agent①  Pre-processing[/bold]"))
        spec = self.agent1.run(description, stl_path=stl_path)

        # ── Agent② ───────────────────────────────────────────────────
        console.print(Rule("[bold]Agent②  Prompt Generation (RAG)[/bold]"))
        context = self.agent2.run(spec)

        # ── Agent③ ───────────────────────────────────────────────────
        console.print(Rule("[bold]Agent③  OpenFOAMGPT[/bold]"))
        artifacts = self.agent3.run(
            context=context,
            output_dir=output_dir,
            convergence_threshold=convergence_threshold,
        )

        # ── Agent④ ───────────────────────────────────────────────────
        console.print(Rule("[bold]Agent④  Post-processing[/bold]"))
        report = self.agent4.run(artifacts)

        return report
